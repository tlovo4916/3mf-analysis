"""Bounded 3MF/OPC geometry reader; no mesh repair, appearance or CAD inference.

The default remains strict: only core/production required extensions and model
objects are accepted. ``geometry_only=True`` also permits the *verified* 2015/02
Materials namespace for geometry analysis, not editing/manufacturing, and valid
core non-model types. Their purpose and reconstruction blockers are reported.
Unreachable object structures are deferred only in this compatibility mode.
Unknown required extensions are never accepted, even with build-error isolation.

One MeshInstance represents one included entry build item. Components stay in
that instance but NEVER weld to another leaf occurrence, including coincident or
touching parts. Within each leaf, only exactly equal computed float64 positions
weld; first-source-vertex order, unused vertices and all faces survive. Meshes use
process=False. Reflections reverse triangle indices, not source winding defects.

Matrices use column vectors. Reference translations use the parent model's unit;
``source_to_world_mm`` maps raw leaf coordinates to mm, also for mixed-unit parts.
Vertex/face ranges are half-open output-mesh ranges. Each component's
``source_vertex_to_instance_vertex_ids`` maps EVERY original vertex to its output
ID. Raw coordinates remain in the source model part, not duplicated per occurrence
or approximated by inverse-transforming output. Metadata IDs address the bounded
``document.metadata['source_metadata']`` table, avoiding repeated metadata blobs.

Build isolation reports incomplete coverage; package/XML/security and cumulative
budget failures remain fatal. No archive member is extracted or external URI
fetched. Namespace-less legacy XML remains accepted with a warning.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
import stat
from typing import Any
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
import zipfile
import zlib

import numpy as np
import trimesh

from slicer_metadata import SlicerMetadata, is_role_config_member

CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
PRODUCTION_NS = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
# Verified against 3MFConsortium/spec_materials, Appendix E.3 / XSD. Do not
# invent aliases for other dates or call this extension fully supported.
MATERIAL_NS = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
# Core specification Appendix B.1.2: optional segmentation, not new geometry.
TRIANGLESETS_NS = "http://schemas.microsoft.com/3dmanufacturing/trianglesets/2021/07"
OBJECT_TYPES = frozenset({"model", "solidsupport", "support", "surface", "other"})
XML_NS = "http://www.w3.org/XML/1998/namespace"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
MODEL_REL = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
UNITS = {
    "micron": 0.001, "millimeter": 1.0, "centimeter": 10.0,
    "meter": 1000.0, "inch": 25.4, "foot": 304.8,
}


class ThreeMFError(ValueError):
    """Malformed, unsafe, unsupported, or over-budget input."""


class _PackageError(ThreeMFError):
    """Fatal security/XML/required-extension/budget error, never item-isolated."""


@dataclass(frozen=True)
class ThreeMFLimits:
    """Per-document budgets; optional ``limits=`` permits deliberate tuning.

    XML element, source geometry and expanded geometry counts are cumulative.
    Expanded geometry counts are charged *before welding*, across all instances.
    Component depth includes the build object's level; occurrences count every
    visited object, including assemblies, so a small shared DAG cannot explode.
    """
    max_archive_bytes: int = 256 * 1024 * 1024
    max_zip_members: int = 4096
    max_member_bytes: int = 128 * 1024 * 1024
    max_total_uncompressed_bytes: int = 256 * 1024 * 1024
    max_xml_bytes: int = 128 * 1024 * 1024
    max_total_xml_bytes: int = 192 * 1024 * 1024
    max_xml_parts: int = 256
    max_xml_elements: int = 2_000_000
    max_xml_depth: int = 64
    max_models: int = 128
    max_objects: int = 100_000
    max_source_vertices: int = 1_000_000
    max_source_faces: int = 2_000_000
    max_build_items: int = 10_000
    max_instances: int = 10_000
    max_component_depth: int = 64
    max_component_occurrences: int = 100_000
    max_output_vertices: int = 2_000_000
    max_output_faces: int = 4_000_000
    # Source metadata is bounded and referenced by IDs, not copied per component.
    # Excess optional metadata is explicitly reported as omitted/truncated.
    max_metadata_entries: int = 4096
    max_metadata_text_chars: int = 4096
    max_metadata_total_chars: int = 256 * 1024
    max_metadata_attributes: int = 16
    max_ignored_content_summaries: int = 1024

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("All ThreeMFLimits must be positive integers")


DEFAULT_LIMITS = ThreeMFLimits()


@dataclass
class MeshInstance:
    mesh: trimesh.Trimesh
    metadata: dict[str, Any]


@dataclass
class ThreeMFDocument:
    instances: list[MeshInstance]
    metadata: dict[str, Any]


def _validate_affine(matrix: np.ndarray, context: str) -> None:
    if not np.isfinite(matrix).all():
        raise ThreeMFError(f"Non-finite transform: {context}")
    try:
        # slogdet avoids determinant overflow/underflow on valid scales. Do not
        # classify small but invertible scales as singular using a fixed epsilon.
        with np.errstate(all="ignore"):
            sign, logdet = np.linalg.slogdet(matrix[:3, :3])
            inverse = np.linalg.inv(matrix)
        if sign == 0 or not np.isfinite(logdet) or not np.isfinite(inverse).all():
            raise np.linalg.LinAlgError("singular or unrepresentable inverse")
    except np.linalg.LinAlgError as exc:
        raise ThreeMFError(f"Non-invertible transform: {context}") from exc


def transform(text: str | None) -> np.ndarray:
    """Decode 3MF's row-vector 12-number serialization as a column-vector 4x4.

    Missing/blank means identity. NaN, infinity, singular linear parts and
    transforms whose inverse cannot be represented as float64 are rejected.
    Translation is still in the owning model's native unit here, not mm.
    """
    if text is None or (isinstance(text, str) and not text.strip()):
        return np.eye(4)
    if not isinstance(text, str):
        raise ThreeMFError("Invalid 3MF transform: expected text")
    tokens = text.split(maxsplit=12)
    if len(tokens) != 12:
        raise ThreeMFError("Invalid 3MF transform: expected 12 numbers")
    try:
        values = np.asarray([float(v) for v in tokens], dtype=np.float64)
    except (ValueError, OverflowError) as exc:
        raise ThreeMFError("Invalid 3MF transform: expected 12 finite numbers") from exc
    matrix = np.eye(4)
    matrix[:3, :3] = values[:9].reshape(3, 3).T
    matrix[:3, 3] = values[9:]
    _validate_affine(matrix, "3MF transform")
    return matrix


def _integer(value: str | None, context: str, minimum: int = 0) -> int:
    text = value.strip() if value is not None else ""
    if not re.fullmatch(r"[+-]?[0-9]+", text):
        raise ThreeMFError(f"Invalid integer {context}: {value!r}")
    try:
        result = int(text)
    except ValueError as exc:
        raise ThreeMFError(f"Invalid integer {context}") from exc
    if result < minimum:
        raise ThreeMFError(f"Invalid integer {context}: must be >= {minimum}")
    return result


def _printable(value: str) -> bool:
    value = value.strip()
    if value not in ("0", "1", "false", "true"):
        raise ThreeMFError(f"Invalid build printable boolean: {value!r}")
    return value in ("1", "true")


def _decoded_path(value: str) -> str:
    # urlsplit silently strips some control characters; reject them beforehand.
    if (not value or value != value.strip()
            or any(ord(c) < 32 or ord(c) == 127 for c in value) or "\\" in value):
        raise _PackageError(f"Unsafe package path: {value!r}")
    if "?" in value or "#" in value or value.startswith("//"):
        raise _PackageError(f"Unsupported package URI: {value!r}")
    try:
        uri = urlsplit(value)
        if uri.scheme or uri.netloc or re.search(r"%(?![0-9a-fA-F]{2})", value):
            raise ValueError("external or malformed URI")
        decoded = unquote(uri.path, encoding="utf-8", errors="strict")
    except (ValueError, UnicodeError) as exc:
        raise _PackageError(f"Unsafe package URI: {value!r}") from exc
    # Nested escapes are ambiguous between package producers. Reject rather
    # than trying multiple decodings; URI escaping otherwise happens once.
    if ("%" in decoded or ":" in decoded or "\\" in decoded
            or any(ord(c) < 32 or ord(c) == 127 for c in decoded)
            or decoded.startswith("//")):
        raise _PackageError(f"Unsafe package path: {value!r}")
    return decoded


def _member_name(value: str, is_directory: bool) -> str:
    decoded = _decoded_path(value)
    if is_directory:
        decoded = decoded.removesuffix("/")
    if decoded.startswith("/") or any(p in ("", ".", "..") for p in decoded.split("/")):
        raise _PackageError(f"Unsafe ZIP member path: {value!r}")
    return decoded


def _part_path(value: str, parent: str = "") -> str:
    decoded = _decoded_path(value)
    if decoded.endswith("/"):
        raise _PackageError(f"Package reference is a directory: {value!r}")
    parts = [] if decoded.startswith("/") else parent.split("/")[:-1]
    for segment in decoded.lstrip("/").split("/"):
        if segment == ".":
            continue
        if segment == "..":
            if not parts:
                raise _PackageError(f"Package path escapes archive root: {value!r}")
            parts.pop()
        elif not segment:
            raise _PackageError(f"Unsafe package reference: {value!r}")
        else:
            parts.append(segment)
    if not parts:
        raise _PackageError(f"Empty package reference: {value!r}")
    return "/".join(parts)


def _reference_model_path(element: ET.Element, parent_path: str) -> str:
    for attribute in element.attrib:
        if attribute.rsplit("}", 1)[-1] == "path" and attribute != f"{{{PRODUCTION_NS}}}path":
            raise _PackageError(f"Unsupported non-production path attribute in {parent_path}")
    path = element.get(f"{{{PRODUCTION_NS}}}path")
    return parent_path if path is None else _part_path(path, parent_path)


class _SafeTreeBuilder(ET.TreeBuilder):
    """Parser callbacks reject DTDs before entity expansion, also for UTF-16 XML."""
    def __init__(self, package: _Package, name: str):
        super().__init__()
        self.package = package
        self.name = name
        self.depth = 0
        self.namespaces: dict[str, str] = {}
        self.root_namespaces: dict[str, str] = {}

    def doctype(self, name, pubid, system):
        raise _PackageError(f"XML DTD/entity declarations are forbidden: {self.name}")

    def start_ns(self, prefix, uri):
        self.namespaces[prefix or ""] = uri

    def start(self, tag, attrs):
        self.depth += 1
        self.package.xml_elements += 1
        if self.depth > self.package.limits.max_xml_depth:
            raise _PackageError(f"XML depth limit exceeded: {self.name}")
        if self.package.xml_elements > self.package.limits.max_xml_elements:
            raise _PackageError("Total XML element count limit exceeded")
        if self.depth == 1:
            self.root_namespaces = self.namespaces.copy()
        return super().start(tag, attrs)

    def end(self, tag):
        result = super().end(tag)
        self.depth -= 1
        return result


@dataclass
class _SourceMesh:
    vertices: np.ndarray
    faces: np.ndarray


@dataclass
class _Reference:
    model_path: str
    object_id: int
    native_transform: np.ndarray
    mm_transform: np.ndarray
    transform_text: str | None
    metadata_ids: list[int] = field(default_factory=list)
    part_number: str = ""


@dataclass
class _Object:
    object_id: int
    name: str
    object_type: str
    element: ET.Element
    metadata_ids: list[int] = field(default_factory=list)
    geometry_element: ET.Element | None = None
    geometry: _SourceMesh | tuple[_Reference, ...] | None = None
    geometry_error: str | None = None


@dataclass
class _Model:
    path: str
    namespace: str
    unit: str
    scale: float
    objects: dict[int, _Object]
    build: ET.Element | None
    required_extensions: list[str] = field(default_factory=list)
    namespaces: dict[str, str] = field(default_factory=dict)
    metadata_ids: list[int] = field(default_factory=list)

    def tag(self, local: str) -> str:
        return f"{{{self.namespace}}}{local}" if self.namespace else local


class _Package:
    def __init__(self, archive: zipfile.ZipFile, limits: ThreeMFLimits, *,
                 source_path: str, geometry_only: bool, isolate_build_errors: bool,
                 include_nonprintable: bool):
        self.archive = archive
        self.limits = limits
        self.source_path = source_path
        self.geometry_only = geometry_only
        self.isolate_build_errors = isolate_build_errors
        self.include_nonprintable = include_nonprintable
        self.members: dict[str, zipfile.ZipInfo] = {}
        self.models: dict[str, _Model] = {}
        self.model_errors: dict[str, str] = {}
        self.warnings: list[str] = []
        self._warnings_seen: set[str] = set()
        self.xml_parts = self.xml_bytes = self.xml_elements = 0
        self.objects_indexed = self.objects_parsed = self.meshes_parsed = 0
        self.source_vertices = self.source_faces = 0
        self.occurrences = self.output_vertices = self.output_faces = 0
        self.reached_objects: set[tuple[str, int]] = set()
        self.source_metadata: list[dict[str, Any]] = []
        self._metadata_seen: dict[ET.Element, int | None] = {}
        self.metadata_chars = self.metadata_omitted = self.metadata_truncated = 0
        self.slicer_metadata_entries = self.slicer_metadata_chars = 0
        self.slicer_config_expected = False
        self.ignored_content: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        self.ignored_summaries_omitted = 0
        self._validate_archive()

    def warn(self, text: str) -> None:
        if text not in self._warnings_seen:
            self._warnings_seen.add(text)
            self.warnings.append(text)

    def ignore(self, model: _Model, scope: str, name: str, kind: str = "element") -> None:
        """Aggregate ignored content by source scope, not by millions of faces."""
        key = (model.path, scope, name, kind)
        if key in self.ignored_content:
            self.ignored_content[key]["count"] += 1
            return
        if len(self.ignored_content) >= self.limits.max_ignored_content_summaries:
            self.ignored_summaries_omitted += 1
            self.warn("Ignored-content summary limit reached; additional occurrences not listed")
            return
        namespace = name[1:].split("}", 1)[0] if name.startswith("{") else model.namespace
        self.ignored_content[key] = {
            "model_path": model.path, "scope": scope, "kind": kind,
            "name": name, "namespace": namespace, "count": 1,
            "handling": "not_interpreted_not_recovered",
            "includes_descendants": kind == "element",
        }

    def inspect_attributes(self, element: ET.Element, model: _Model, scope: str) -> None:
        for name in element.attrib:
            if name in ("pid", "pindex", "p1", "p2", "p3"):
                self.ignore(model, scope, name, "appearance_attribute")
            elif name.startswith("{") and name not in (
                    f"{{{PRODUCTION_NS}}}path", f"{{{XML_NS}}}lang"):
                self.ignore(model, scope, name, "extension_attribute")

    def metadata_ids(self, element: ET.Element, model: _Model, scope: str) -> list[int]:
        """Preserve bounded source strings verbatim; no interpretation of values."""
        result = []
        for child in element:
            if child.tag == model.tag("metadata"):
                candidates = (child,)
            elif child.tag == model.tag("metadatagroup"):
                candidates = child
            else:
                continue
            for node in candidates:
                if node.tag != model.tag("metadata"):
                    self.ignore(model, scope + "/metadatagroup", node.tag)
                    continue
                if node not in self._metadata_seen:
                    self._metadata_seen[node] = None
                    if len(self.source_metadata) + self.slicer_metadata_entries >= self.limits.max_metadata_entries:
                        self.metadata_omitted += 1
                        self.warn("Source metadata entry limit reached; excess metadata omitted")
                        continue
                    truncated = False

                    def bounded(text: str) -> str:
                        nonlocal truncated
                        remaining = max(0, self.limits.max_metadata_total_chars - self.metadata_chars)
                        value = text[:min(self.limits.max_metadata_text_chars, remaining)]
                        self.metadata_chars += len(value)
                        if len(value) != len(text):
                            truncated = True
                        return value

                    attributes = {}
                    for index, (name, value) in enumerate(node.attrib.items()):
                        if index >= self.limits.max_metadata_attributes:
                            truncated = True
                            break
                        attributes[bounded(name)] = bounded(value)
                    value = bounded(node.text or "")
                    # CT_Metadata has text, not nested XML. Do not silently claim
                    # that nested vendor data was recovered as a plain value.
                    for nested in node:
                        self.ignore(model, scope + "/metadata", nested.tag)
                        truncated = True
                    metadata_id = len(self.source_metadata)
                    name = attributes.get("name", "")
                    prefix = name.split(":", 1)[0] if ":" in name else ""
                    self.source_metadata.append({
                        "metadata_id": metadata_id, "model_path": model.path,
                        "scope": scope, "name": name, "value": value,
                        "attributes": attributes,
                        "name_namespace": model.namespaces.get(prefix) if prefix else None,
                        "truncated": truncated,
                    })
                    self._metadata_seen[node] = metadata_id
                    if truncated:
                        self.metadata_truncated += 1
                        self.warn("Source metadata truncated; consult original 3MF for full values")
                metadata_id = self._metadata_seen[node]
                if metadata_id is not None:
                    result.append(metadata_id)
        return result

    def ignorable_geometry_annotation(self, child: ET.Element, model: _Model, scope: str) -> bool:
        # Unknown object/mesh structures remain unsupported even when optional:
        # geometry-only is NOT permission to discard an unknown shape encoding.
        namespace = child.tag[1:].split("}", 1)[0] if child.tag.startswith("{") else ""
        if self.geometry_only and namespace in (MATERIAL_NS, TRIANGLESETS_NS):
            self.ignore(model, scope, child.tag)
            return True
        return False

    def validate_object(self, model: _Model, obj: _Object) -> ET.Element:
        if obj.geometry_element is not None:
            return obj.geometry_element
        context = f"{model.path}:{obj.object_id}"
        if obj.object_type not in OBJECT_TYPES:
            raise ThreeMFError(f"Unknown/unsupported object type {obj.object_type!r}: {context}")
        if not self.geometry_only and obj.object_type != "model":
            raise ThreeMFError(f"Unsupported non-model object type {obj.object_type!r}: {context}")
        geometry = [e for e in obj.element if e.tag in (model.tag("mesh"), model.tag("components"))]
        if len(geometry) != 1:
            raise ThreeMFError(f"Object must contain exactly one mesh or components: {context}")
        for child in obj.element:
            if child is geometry[0] or child.tag in (model.tag("metadata"), model.tag("metadatagroup")):
                continue
            if not self.ignorable_geometry_annotation(child, model, "object"):
                raise ThreeMFError(f"Unsupported object geometry/element: {context}: {child.tag}")
        obj.geometry_element = geometry[0]
        return geometry[0]

    def _validate_archive(self) -> None:
        infos = self.archive.infolist()
        if len(infos) > self.limits.max_zip_members:
            raise _PackageError("ZIP member count limit exceeded")
        names: set[str] = set()
        total = xml_total = xml_count = 0
        for info in infos:
            if info.orig_filename != info.filename:
                raise _PackageError("ZIP member has a truncated/unsafe filename")
            name = _member_name(info.filename, info.is_dir())
            if name in names:
                raise _PackageError(f"Duplicate ZIP member (including URI aliases): {name}")
            names.add(name)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise _PackageError(f"ZIP symlink member is unsupported: {name}")
            if info.flag_bits & 1:
                raise _PackageError(f"Encrypted ZIP member is unsupported: {name}")
            if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise _PackageError(f"Unsupported ZIP compression: {name}")
            if not 0 <= info.file_size <= self.limits.max_member_bytes:
                raise _PackageError(f"ZIP member size limit exceeded: {name}")
            total += info.file_size
            if total > self.limits.max_total_uncompressed_bytes:
                raise _PackageError("Total ZIP uncompressed size limit exceeded")
            if info.is_dir():
                continue
            self.members[name] = info
            if name.lower().endswith((".xml", ".model", ".rels")) or is_role_config_member(name):
                xml_count += 1
                xml_total += info.file_size
                if info.file_size > self.limits.max_xml_bytes:
                    raise _PackageError(f"XML part size limit exceeded: {name}")
        if xml_count > self.limits.max_xml_parts:
            raise _PackageError("XML part count limit exceeded")
        if xml_total > self.limits.max_total_xml_bytes:
            raise _PackageError("Total XML size limit exceeded")

    def xml(self, name: str) -> tuple[ET.Element, dict[str, str]]:
        info = self.members.get(name)
        if info is None:
            raise ThreeMFError(f"Missing package part: {name}")
        self.xml_parts += 1
        self.xml_bytes += info.file_size
        if self.xml_parts > self.limits.max_xml_parts:
            raise _PackageError("Parsed XML part count limit exceeded")
        if info.file_size > self.limits.max_xml_bytes:
            raise _PackageError(f"XML part size limit exceeded: {name}")
        if self.xml_bytes > self.limits.max_total_xml_bytes:
            raise _PackageError("Total parsed XML size limit exceeded")
        target = _SafeTreeBuilder(self, name)
        parser = ET.XMLParser(target=target)
        consumed = 0
        try:
            with self.archive.open(info) as source:
                while True:
                    data = source.read(min(64 * 1024, self.limits.max_xml_bytes - consumed + 1))
                    if not data:
                        break
                    consumed += len(data)
                    if consumed > self.limits.max_xml_bytes or consumed > info.file_size:
                        raise _PackageError(f"XML actual size limit exceeded: {name}")
                    parser.feed(data)
            root = parser.close()
        except ThreeMFError as exc:
            raise _PackageError(str(exc)) from exc
        except (ET.ParseError, ValueError, LookupError, UnicodeError) as exc:
            raise _PackageError(f"Invalid XML in {name}: {exc}") from exc
        if consumed != info.file_size:
            raise _PackageError(f"Truncated XML part: {name}")
        return root, target.root_namespaces

    def entry_model(self) -> tuple[str, str]:
        if "_rels/.rels" not in self.members:
            main = "3D/3dmodel.model"
            if main not in self.members:
                raise ThreeMFError("No OPC root relationships or conventional entry model")
            self.warn("OPC root relationships absent; using 3D/3dmodel.model")
            return main, "conventional_path"
        root, _ = self.xml("_rels/.rels")
        if root.tag not in (f"{{{REL_NS}}}Relationships", "Relationships"):
            raise ThreeMFError("Invalid OPC root relationships element")
        prefix = f"{{{REL_NS}}}" if root.tag.startswith("{") else ""
        ids: set[str] = set()
        candidates = []
        for relation in root:
            if relation.tag != prefix + "Relationship":
                raise ThreeMFError("Unsupported element in OPC root relationships")
            rid = relation.get("Id")
            if not rid or rid in ids:
                raise ThreeMFError("Missing/duplicate OPC relationship Id")
            ids.add(rid)
            if relation.get("Type") == MODEL_REL:
                candidates.append(relation)
        if len(candidates) != 1:
            raise ThreeMFError("OPC root relationships must select exactly one entry model")
        relation = candidates[0]
        if relation.get("TargetMode", "Internal") != "Internal":
            raise ThreeMFError("External/unsupported OPC entry model TargetMode")
        target = relation.get("Target")
        if target is None:
            raise ThreeMFError("Missing OPC entry model Target")
        return _part_path(target), "opc_root_relationship"

    def model(self, path: str) -> _Model:
        if path in self.models:
            return self.models[path]
        if path in self.model_errors:
            raise ThreeMFError(self.model_errors[path])
        if len(self.models) + len(self.model_errors) >= self.limits.max_models:
            raise _PackageError("Model part count limit exceeded")
        try:
            return self.parse_model(path)
        except _PackageError:
            raise
        except ThreeMFError as exc:
            self.model_errors[path] = str(exc)
            raise

    def parse_model(self, path: str) -> _Model:
        root, namespaces = self.xml(path)
        if root.tag == f"{{{CORE_NS}}}model":
            namespace = CORE_NS
        elif root.tag == "model":
            namespace = ""
            self.warn(f"Namespace-less legacy model XML: {path}")
        else:
            raise ThreeMFError(f"Unsupported model root/namespace: {path}: {root.tag}")
        required = root.get("requiredextensions", "").split()
        for prefix in required:
            uri = namespaces.get(prefix)
            if uri in (CORE_NS, PRODUCTION_NS):
                continue
            if self.geometry_only and uri == MATERIAL_NS:
                self.warn(f"Required material/appearance extension NOT recovered; geometry analysis only in {path}: {uri}")
                continue
            raise _PackageError(f"Unsupported/undeclared requiredextensions prefix {prefix!r} ({uri!r}) in {path}")
        for uri in sorted(set(namespaces.values()) - {CORE_NS, PRODUCTION_NS, XML_NS}):
            if not any(namespaces.get(prefix) == uri for prefix in required):
                self.warn(f"Optional XML namespace not interpreted in {path}: {uri}")
        tag = lambda local: f"{{{namespace}}}{local}" if namespace else local
        # Inspect source XML before optional metadata truncation: removing a
        # Bambu config must not turn a slicer project into an ordinary core part.
        # This recognizes exporter evidence only; no vendor extension is enabled.
        if (any("bambustudio" in uri.casefold() or "bambulab.com" in uri.casefold()
                for uri in namespaces.values())
                or any(node.get("name", "").casefold() == "application"
                       and re.search(r"bambu\s*studio", node.text or "", re.IGNORECASE)
                       for node in root.iter(tag("metadata")))):
            self.slicer_config_expected = True
        # Security checks precede semantic item isolation and reachability. An
        # invalid printable flag or earlier bad component must not hide an
        # unsafe URI elsewhere in an already parsed model part.
        reference_tags = (tag("item"), tag("component"))
        for element in root.iter():
            if element.tag in reference_tags:
                _reference_model_path(element, path)
        unit = root.get("unit", "millimeter")
        if unit not in UNITS:
            raise ThreeMFError(f"Unknown/unsupported unit {unit!r} in {path}")
        resources = root.findall(tag("resources"))
        builds = root.findall(tag("build"))
        if len(resources) != 1 or len(builds) > 1:
            raise ThreeMFError(f"Expected one resources and at most one build in {path}")
        model = _Model(path, namespace, unit, UNITS[unit], {}, builds[0] if builds else None,
                       required, namespaces)
        model.metadata_ids = self.metadata_ids(root, model, "model")
        self.inspect_attributes(root, model, "model")
        for child in root:
            if child.tag not in (tag("resources"), tag("build"), tag("metadata"), tag("metadatagroup")):
                self.ignore(model, "model", child.tag)
        self.inspect_attributes(resources[0], model, "resources")
        if model.build is not None:
            self.inspect_attributes(model.build, model, "build")
        resource_ids: set[int] = set()
        for element in resources[0]:
            if element.get("id") is not None:
                resource_id = _integer(element.get("id"), f"resource id in {path}", 1)
                if resource_id in resource_ids:
                    raise ThreeMFError(f"Duplicate object/resource ID {resource_id} in {path}")
                resource_ids.add(resource_id)
            if element.tag != tag("object"):
                self.warn(f"Non-object resource not interpreted in {path}: {element.tag}")
                self.ignore(model, "resources", element.tag)
                continue
            object_id = _integer(element.get("id"), f"object id in {path}", 1)
            obj = _Object(object_id, element.get("name", ""), element.get("type", "model"), element,
                          self.metadata_ids(element, model, f"object:{object_id}"))
            # Preserve legacy eager structural rejection by default. Compatibility
            # mode indexes IDs but validates only objects that are actually used.
            if not self.geometry_only:
                try:
                    self.validate_object(model, obj)
                except ThreeMFError as exc:
                    if not self.isolate_build_errors:
                        raise
                    obj.geometry_error = str(exc)
            self.objects_indexed += 1
            if self.objects_indexed > self.limits.max_objects:
                raise _PackageError("Total object count limit exceeded")
            model.objects[object_id] = obj
            self.inspect_attributes(element, model, "object")
            if any(k in element.attrib for k in ("pid", "pindex")):
                self.warn(f"Material properties not interpreted in {path}")
        if self.geometry_only or self.isolate_build_errors:
            # Preserve reference metadata even when a non-printable item or
            # unreachable assembly is not expanded. It is source-only data;
            # parsing it does not mark that object's geometry as validated.
            if model.build is not None:
                for index, item in enumerate(model.build):
                    self.metadata_ids(item, model, f"build_item:{index}")
            for obj in model.objects.values():
                for group in obj.element.findall(model.tag("components")):
                    for index, child in enumerate(group):
                        if child.tag == model.tag("component"):
                            self.metadata_ids(child, model, f"object:{obj.object_id}/component:{index}")
        self.models[path] = model
        return model

    def object(self, path: str, object_id: int) -> tuple[_Model, _Object]:
        model = self.model(path)
        obj = model.objects.get(object_id)
        if obj is None:
            raise ThreeMFError(f"Missing object {path}:{object_id}")
        return model, obj

    def reference(self, element: ET.Element, parent: _Model, scope: str) -> _Reference:
        for child in element:
            if child.tag == parent.tag("metadatagroup"):
                continue
            if not self.ignorable_geometry_annotation(child, parent, scope):
                raise ThreeMFError(f"Unsupported nested content in build item/component: {parent.path}: {child.tag}")
        metadata_ids = self.metadata_ids(element, parent, scope)
        self.inspect_attributes(element, parent, "reference")
        model_path = _reference_model_path(element, parent.path)
        object_id = _integer(element.get("objectid"), f"object reference in {parent.path}", 1)
        text = element.get("transform")
        native = transform(text)
        mm = native.copy()
        with np.errstate(all="ignore"):
            mm[:3, 3] *= parent.scale
        _validate_affine(mm, f"unit-converted reference in {parent.path}")
        return _Reference(model_path, object_id, native, mm, text, metadata_ids,
                          element.get("partnumber", ""))

    def geometry(self, model: _Model, obj: _Object) -> _SourceMesh | tuple[_Reference, ...]:
        self.reached_objects.add((model.path, obj.object_id))
        if obj.geometry is not None:
            return obj.geometry
        if obj.geometry_error is not None:
            raise ThreeMFError(obj.geometry_error)
        self.objects_parsed += 1
        try:
            return self.parse_geometry(model, obj)
        except _PackageError:
            raise
        except ThreeMFError as exc:
            # Cache reasons, not exception tracebacks that retain failed arrays.
            # Repeated items must not re-charge geometry or duplicate metadata.
            obj.geometry_error = str(exc)
            raise

    def parse_geometry(self, model: _Model, obj: _Object) -> _SourceMesh | tuple[_Reference, ...]:
        element = self.validate_object(model, obj)
        context = f"{model.path}:{obj.object_id}"
        if element.tag == model.tag("components"):
            self.inspect_attributes(element, model, "components")
            if not len(element):
                raise ThreeMFError(f"Empty components object: {context}")
            if any(child.tag != model.tag("component") for child in element):
                raise ThreeMFError(f"Unsupported components element: {context}")
            obj.geometry = tuple(self.reference(child, model, f"object:{obj.object_id}/component:{index}")
                                 for index, child in enumerate(element))
            return obj.geometry
        if (len(element.findall(model.tag("vertices"))) != 1
                or len(element.findall(model.tag("triangles"))) != 1):
            raise ThreeMFError(f"Unsupported/malformed mesh geometry: {context}")
        for child in element:
            if child.tag in (model.tag("vertices"), model.tag("triangles")):
                continue
            if not self.ignorable_geometry_annotation(child, model, "mesh"):
                raise ThreeMFError(f"Unsupported/malformed mesh geometry: {context}: {child.tag}")
        self.inspect_attributes(element, model, "mesh")
        vertices_element = element.find(model.tag("vertices"))
        triangles_element = element.find(model.tag("triangles"))
        self.inspect_attributes(vertices_element, model, "vertices")
        self.inspect_attributes(triangles_element, model, "triangles")
        if not len(vertices_element) or not len(triangles_element):
            raise ThreeMFError(f"Empty mesh: {context}")
        self.source_vertices += len(vertices_element)
        self.source_faces += len(triangles_element)
        if self.source_vertices > self.limits.max_source_vertices:
            raise _PackageError("Total source vertex count limit exceeded")
        if self.source_faces > self.limits.max_source_faces:
            raise _PackageError("Total source face count limit exceeded")
        vertices = np.empty((len(vertices_element), 3), dtype=np.float64)
        for i, vertex in enumerate(vertices_element):
            if vertex.tag != model.tag("vertex") or len(vertex):
                raise ThreeMFError(f"Unsupported vertex element: {context}")
            self.inspect_attributes(vertex, model, "vertex")
            try:
                vertices[i] = [float(vertex.attrib[k]) for k in ("x", "y", "z")]
            except (KeyError, ValueError, OverflowError) as exc:
                raise ThreeMFError(f"Invalid vertex {i}: {context}") from exc
        if not np.isfinite(vertices).all():
            raise ThreeMFError(f"Non-finite vertex coordinates: {context}")
        faces = np.empty((len(triangles_element), 3), dtype=np.int64)
        for i, triangle in enumerate(triangles_element):
            if triangle.tag != model.tag("triangle") or len(triangle):
                raise ThreeMFError(f"Unsupported triangle element: {context}")
            indices = [_integer(triangle.get(k), f"triangle index {i} in {context}") for k in ("v1", "v2", "v3")]
            if any(index >= len(vertices) for index in indices):
                raise ThreeMFError(f"Triangle index out of bounds: {context}, face {i}")
            faces[i] = indices
            self.inspect_attributes(triangle, model, "triangle")
            if any(k in triangle.attrib for k in ("pid", "p1", "p2", "p3")):
                self.warn(f"Triangle material/color properties not interpreted in {model.path}")
        vertices.setflags(write=False)
        faces.setflags(write=False)
        obj.geometry = _SourceMesh(vertices, faces)
        self.meshes_parsed += 1
        return obj.geometry

    def instance(self, build_index: int, reference: _Reference, entry: _Model,
                 printable: bool) -> MeshInstance:
        model, obj = self.object(reference.model_path, reference.object_id)
        vertices: list[np.ndarray] = []
        faces: list[np.ndarray] = []
        components: list[dict[str, Any]] = []
        vertex_offset = face_offset = 0

        def expand(path: str, object_id: int, world: np.ndarray,
                   stack: tuple[tuple[str, int], ...], chain: list[dict[str, Any]]) -> None:
            nonlocal vertex_offset, face_offset
            key = (path, object_id)
            if key in stack:
                raise ThreeMFError(f"Cyclic components: {stack + (key,)}")
            if len(stack) >= self.limits.max_component_depth:
                raise _PackageError("Component depth limit exceeded")
            self.occurrences += 1
            if self.occurrences > self.limits.max_component_occurrences:
                raise _PackageError("Expanded component occurrence limit exceeded")
            source_model, source_obj = self.object(path, object_id)
            geometry = self.geometry(source_model, source_obj)
            if isinstance(geometry, tuple):
                for index, child in enumerate(geometry):
                    with np.errstate(all="ignore"):
                        child_world = world @ child.mm_transform
                    _validate_affine(child_world, f"composed component {path}:{object_id}")
                    step = {
                        "parent_model_path": path, "parent_object_id": object_id,
                        "component_index": index, "model_path": child.model_path,
                        "object_id": child.object_id, "transform_unit": source_model.unit,
                        "transform": child.native_transform.tolist(),
                        "transform_mm": child.mm_transform.tolist(),
                        "transform_text": child.transform_text,
                        "metadata_ids": child.metadata_ids.copy(),
                        "part_number": child.part_number,
                        "parent_object_type": source_obj.object_type,
                        "parent_object_metadata_ids": source_obj.metadata_ids.copy(),
                    }
                    expand(child.model_path, child.object_id, child_world, stack + (key,), chain + [step])
                return
            nv, nf = len(geometry.vertices), len(geometry.faces)
            self.output_vertices += nv
            self.output_faces += nf
            if self.output_vertices > self.limits.max_output_vertices:
                raise _PackageError("Expanded output vertex count limit exceeded")
            if self.output_faces > self.limits.max_output_faces:
                raise _PackageError("Expanded output face count limit exceeded")
            unit_matrix = np.diag([source_model.scale] * 3 + [1.0])
            with np.errstate(all="ignore"):
                source_to_world = world @ unit_matrix
            _validate_affine(source_to_world, f"source-to-world {path}:{object_id}")
            with np.errstate(all="ignore"):
                transformed = geometry.vertices @ source_to_world[:3, :3].T + source_to_world[:3, 3]
            if not np.isfinite(transformed).all():
                raise ThreeMFError(f"Non-finite transformed vertices: {path}:{object_id}")
            # Equivalent to apply_transform's reflection handling, without its
            # near-identity shortcut (which can discard tiny, real transforms).
            mirrored = bool(np.linalg.slogdet(source_to_world[:3, :3])[0] < 0)
            source_faces = geometry.faces[:, ::-1] if mirrored else geometry.faces
            # Weld ONLY this leaf occurrence. Identical coordinates across two
            # components are not evidence that their topology should be united.
            _, first, inverse = np.unique(transformed, axis=0, return_index=True, return_inverse=True)
            order = np.argsort(first)
            remap = np.empty(len(first), dtype=np.int64)
            remap[order] = np.arange(len(first))
            local_map = remap[inverse.reshape(-1)]  # also handles NumPy 2.0's inverse shape
            welded = transformed[first[order]]
            vertices.append(welded)
            faces.append(local_map[source_faces] + vertex_offset)
            components.append({
                "component_index": len(components), "model_path": path,
                "object_id": object_id, "name": source_obj.name,
                "object_type": source_obj.object_type,
                "metadata_ids": source_obj.metadata_ids.copy(),
                "part_number": source_obj.element.get("partnumber", ""),
                "original_unit": source_model.unit, "unit_scale_mm": source_model.scale,
                "source_vertex_count": nv, "source_face_count": nf,
                "source_face_start": 0, "source_face_end": nf,
                "vertex_start": vertex_offset, "vertex_end": vertex_offset + len(welded),
                "source_vertex_to_instance_vertex_ids": (local_map + vertex_offset).tolist(),
                "source_coordinate_storage": "source_3mf:model/resources/object/mesh/vertices",
                "face_start": face_offset, "face_end": face_offset + nf,
                "source_to_world_mm": source_to_world.tolist(),
                "winding_reversed_for_reflection": mirrored, "component_path": chain,
            })
            if source_obj.object_type != "model":
                self.warn(f"Non-model {source_obj.object_type!r} leaf included for analysis only: {path}:{object_id}")
            vertex_offset += len(welded)
            face_offset += nf

        expand(reference.model_path, reference.object_id, reference.mm_transform, (), [])
        if not faces:
            raise ThreeMFError(f"Build item {build_index} expands to an empty mesh")
        welded_vertices = np.concatenate(vertices)
        welded_faces = np.concatenate(faces)
        source_vertex_count = sum(c["source_vertex_count"] for c in components)
        cleanup = {
            "operation": "merge_coincident_vertices", "comparison": "exact_float64_equality",
            "scope": "leaf_occurrence_only", "cross_component_welding": False,
            "precision": "No decimal rounding or tolerance; only equal computed mm coordinates within one leaf occurrence merge",
            "vertices_before": source_vertex_count, "vertices_after": len(welded_vertices),
            "vertices_merged": source_vertex_count - len(welded_vertices),
            "max_vertex_displacement_mm": 0.0,
            "faces_before": len(welded_faces), "faces_after": len(welded_faces),
            "unreferenced_vertices_removed": 0, "faces_removed": 0,
            "normals_fixed": False, "holes_filled": False,
        }
        effective_types = list(dict.fromkeys(c["object_type"] for c in components))
        blockers = [f"non_model_object_type:{kind}" for kind in effective_types if kind != "model"]
        if not printable:
            blockers.append("non_printable_build_item")
        violations = []
        if "other" in effective_types:
            violations.append("Core 3MF forbids build references to type 'other', directly or recursively")
            self.warn(f"Build item {build_index} references type 'other'; forbidden by core, analysis only")
        metadata = {
            "build_index": build_index, "object_id": reference.object_id,
            "model_path": reference.model_path, "name": obj.name, "printable": printable,
            "object_type": obj.object_type,
            "effective_object_type": effective_types[0] if len(effective_types) == 1 else "mixed",
            "effective_object_types": effective_types,
            "reconstruction_eligible": not blockers,
            "reconstruction_eligibility_basis": "object_type_and_printability_only_not_geometric_validation",
            "reconstruction_blockers": blockers, "core_spec_violations": violations,
            "geometry_only": self.geometry_only, "appearance_not_recovered": True,
            "requires_review": True, "source_path": self.source_path,
            "metadata_ids": obj.metadata_ids.copy(), "build_metadata_ids": reference.metadata_ids.copy(),
            "part_number": obj.element.get("partnumber", ""), "build_part_number": reference.part_number,
            "original_unit": model.unit, "output_unit": "millimeter",
            "build_transform": reference.native_transform.tolist(),
            "build_transform_mm": reference.mm_transform.tolist(),
            "build_transform_text": reference.transform_text,
            "build_transform_unit": entry.unit, "entry_model": entry.path,
            "components": components, "mesh_cleanup": [cleanup],
        }
        mesh = trimesh.Trimesh(vertices=welded_vertices, faces=welded_faces, process=False, metadata=metadata)
        return MeshInstance(mesh, metadata)

    def preflight_reference_xml(self, entry: _Model) -> None:
        """Do not let an earlier item/graph error conceal unsafe referenced XML.

        This visits source references, not expanded instances, ignoring transform
        and object-ID errors until normal per-item validation. Whole parsed XML,
        required extensions and path safety remain fatal. Unused resources and
        child-model builds are not additional sources of reachable geometry.
        """
        pending = [(entry.path, item) for item in entry.build]
        visited: set[tuple[str, int]] = set()
        while pending:
            parent_path, element = pending.pop()
            path = _reference_model_path(element, parent_path)
            try:
                model = self.model(path)
            except _PackageError:
                raise
            except ThreeMFError:
                continue  # cached semantic model failure is reported by its item
            try:
                object_id = _integer(element.get("objectid"), f"object reference in {parent_path}", 1)
            except ThreeMFError:
                continue
            key = (path, object_id)
            if key in visited:
                continue
            visited.add(key)
            obj = model.objects.get(object_id)
            if obj is not None:
                for group in obj.element.findall(model.tag("components")):
                    pending.extend((path, child) for child in group if child.tag == model.tag("component"))

    def load(self) -> ThreeMFDocument:
        entry_path, selection = self.entry_model()
        # Private role configs use the package XML parser and cumulative budgets,
        # outside build-item isolation. They never broaden strict geometry rules.
        slicer = SlicerMetadata(self)
        entry = self.model(entry_path)
        if entry.build is None:
            raise ThreeMFError(f"Entry model has no build: {entry_path}")
        if any(item.tag != entry.tag("item") for item in entry.build):
            raise ThreeMFError("Unsupported entry build element")
        if len(entry.build) > self.limits.max_build_items:
            raise ThreeMFError("Build item count limit exceeded")
        if self.isolate_build_errors:
            self.preflight_reference_xml(entry)
        instances = []
        skipped = []
        failed = []
        for index, item in enumerate(entry.build):
            reference = None
            printable = None
            try:
                printable = _printable(item.get("printable", "1"))
                reference = self.reference(item, entry, f"build_item:{index}")
                # Even skipped items must resolve to a resource; compatibility
                # mode does not validate that unexpanded resource's structure.
                _, obj = self.object(reference.model_path, reference.object_id)
                if not self.geometry_only and obj.geometry_error is not None:
                    self.reached_objects.add((reference.model_path, reference.object_id))
                    raise ThreeMFError(obj.geometry_error)
                if not printable and not self.include_nonprintable:
                    skipped.append({
                        "build_index": index, "object_id": reference.object_id,
                        "model_path": reference.model_path, "printable": False,
                        "object_type": obj.object_type, "metadata_ids": reference.metadata_ids.copy(),
                        "reason": "non_printable_not_requested",
                    })
                    self.warn(f"Non-printable build item skipped: {index}")
                    continue
                if len(instances) >= self.limits.max_instances:
                    raise _PackageError("Printable/included instance count limit exceeded")
                instances.append(self.instance(index, reference, entry, printable))
            except _PackageError:
                raise
            except ThreeMFError as exc:
                if not self.isolate_build_errors:
                    raise
                failed.append({
                    "build_index": index,
                    "object_id": reference.object_id if reference else item.get("objectid"),
                    "model_path": reference.model_path if reference else entry.path,
                    "printable": printable, "reason": str(exc), "error_type": type(exc).__name__,
                    "metadata_ids": self.metadata_ids(item, entry, f"build_item:{index}"),
                    "reference_path_text": item.get(f"{{{PRODUCTION_NS}}}path"),
                    "build_transform_text": item.get("transform"),
                })
                self.warn(f"Build item {index} failed and was not included: {exc}")
        if not self.geometry_only:
            # Isolation is not permission to ignore unsupported structures that
            # belong to no build item. Strict default's whole-index rule remains.
            for model in self.models.values():
                for obj in model.objects.values():
                    if obj.geometry_error is not None and (model.path, obj.object_id) not in self.reached_objects:
                        raise ThreeMFError(obj.geometry_error)
        if not instances:
            label = "valid" if self.include_nonprintable else "printable"
            detail = ""
            if failed:
                reasons = "; ".join(f"item {e['build_index']}: {e['reason']}" for e in failed[:3])
                suffix = f"; {len(failed) - 3} more failures" if len(failed) > 3 else ""
                detail = f"; {len(failed)} failed build items: {reasons}{suffix}"
            raise ThreeMFError(f"3MF has no {label} build instances{detail}")
        skipped_objects = []
        if self.geometry_only:
            for model in self.models.values():
                for obj in model.objects.values():
                    if (model.path, obj.object_id) not in self.reached_objects:
                        skipped_objects.append({
                            "model_path": model.path, "object_id": obj.object_id, "name": obj.name,
                            "object_type": obj.object_type, "metadata_ids": obj.metadata_ids.copy(),
                            "reason": "not_reached_by_included_build_items",
                            "structure_validated": False,
                            "child_elements": list(dict.fromkeys(child.tag for child in obj.element[:16])),
                            "child_element_count": len(obj.element),
                            "child_element_summary_complete": len(obj.element) <= 16,
                        })
            if skipped_objects:
                self.warn(f"Unreached object structures not validated in geometry-only mode: {len(skipped_objects)}")
        slicer_metadata = slicer.attach(entry, instances, coverage_complete=not failed and not skipped)
        if slicer_metadata["config_issues"] or slicer_metadata["role_mapping_status"] == "unresolved":
            self.warn("Slicer role/config evidence unresolved or unsupported; reconstruction vetoed (no CSG/pose applied)")
        metadata = {
            "entry_model": entry_path, "entry_model_selection": selection,
            "slicer_metadata": slicer_metadata,
            **{key: slicer_metadata[key] for key in (
                "material_semantics_complete", "material_semantics_scope", "role_mapping_status",
                "reconstruction_veto", "reconstruction_blockers", "geometry_frame",
                "slicer_assembly_transform_applied", "slicer_private_transforms_applied",
                "printed_material_geometry_recovered")},
            "reconstruction_authorized_by_metadata": False,
            "output_unit": "millimeter", "warnings": self.warnings,
            "build_item_count": len(entry.build), "instance_count": len(instances),
            "geometry_only": self.geometry_only, "appearance_not_recovered": True,
            "isolate_build_errors": self.isolate_build_errors,
            "include_nonprintable": self.include_nonprintable,
            "document_coverage_complete": not failed and not skipped,
            "document_coverage_scope": "entry_build_item_geometry_not_unused_resources_or_appearance",
            "printable_build_coverage_complete": not failed,
            "skipped_build_items": skipped, "failed_build_items": failed,
            "skipped_objects": skipped_objects,
            "failed_model_parts": [{"model_path": path, "reason": reason}
                                   for path, reason in self.model_errors.items()],
            "validation_scope": ("included_build_reachable_object_structures_and_geometry" if self.geometry_only
                                 else "all_indexed_object_structures_and_included_build_reachable_geometry"),
            "source_path": self.source_path, "metadata_ids": entry.metadata_ids.copy(),
            "source_metadata": self.source_metadata,
            "source_metadata_summary": {
                "entries_preserved": len(self.source_metadata), "entries_omitted": self.metadata_omitted,
                "entries_truncated": self.metadata_truncated,
                "characters_preserved": self.metadata_chars - self.slicer_metadata_chars,
                "shared_budget_characters_preserved": self.metadata_chars,
                "shared_budget_slicer_entries": self.slicer_metadata_entries,
                "complete": not self.metadata_omitted and not self.metadata_truncated,
            },
            "ignored_extension_content": list(self.ignored_content.values()),
            "ignored_content_summary_occurrences_omitted": self.ignored_summaries_omitted,
            "provenance": {
                "raw_coordinates": "Read original source model part and vertex order; not duplicated or inverse-transformed",
                "vertex_mapping": "component.source_vertex_to_instance_vertex_ids maps every raw source vertex to an output vertex",
                "metadata_ids": "Indices into this document's bounded source_metadata table; values are not interpreted",
            },
            "models": [{
                "model_path": m.path, "original_unit": m.unit,
                "unit_scale_mm": m.scale, "object_count": len(m.objects),
                "required_extensions": m.required_extensions,
                "metadata_ids": m.metadata_ids.copy(),
                "object_metadata": [{"object_id": o.object_id, "metadata_ids": o.metadata_ids.copy()}
                                    for o in m.objects.values() if o.metadata_ids],
                "extensions": [{
                    "namespace": uri, "prefixes": [p for p, u in m.namespaces.items() if u == uri],
                    "required": any(m.namespaces.get(p) == uri for p in m.required_extensions),
                    "handling": ("core_geometry" if uri == CORE_NS else "production_references" if uri == PRODUCTION_NS
                                 else "appearance_not_recovered" if uri == MATERIAL_NS else "optional_not_interpreted"),
                } for uri in dict.fromkeys(m.namespaces.values()) if uri != XML_NS],
            } for m in self.models.values()],
            "instances": [{k: item.metadata[k] for k in (
                "build_index", "object_id", "model_path", "name", "original_unit", "printable",
                "object_type", "effective_object_type", "reconstruction_eligible", "reconstruction_blockers",
                "reconstruction_veto", "role_mapping_status", "material_semantics_complete")}
                for item in instances],
            "statistics": {
                "models_parsed": len(self.models),
                "model_parts_attempted": len(self.models) + len(self.model_errors),
                "xml_parts_parsed": self.xml_parts,
                "xml_elements": self.xml_elements, "objects_indexed": self.objects_indexed,
                "objects_parsed": self.objects_parsed, "meshes_parsed": self.meshes_parsed,
                "source_vertices": self.source_vertices, "source_faces": self.source_faces,
                "component_occurrences": self.occurrences,
                "output_vertices_before_welding": self.output_vertices,
                "output_faces": self.output_faces,
                "emitted_output_vertices": sum(len(i.mesh.vertices) for i in instances),
                "emitted_output_faces": sum(len(i.mesh.faces) for i in instances),
                "expansion_counts_include_failed_attempts": bool(failed),
            },
            "limits": asdict(self.limits),
        }
        return ThreeMFDocument(instances, metadata)


def load_document(path: str | Path, *, limits: ThreeMFLimits = DEFAULT_LIMITS,
                  geometry_only: bool = False, isolate_build_errors: bool = False,
                  include_nonprintable: bool = False) -> ThreeMFDocument:
    """Read independent mm-coordinate entry-build meshes, with strict defaults.

    ``geometry_only`` permits known Materials required content without recovering
    appearance and allows core non-model types for analysis, not reconstruction.
    Unreachable structures are listed as skipped, not claimed to be validated.
    ``isolate_build_errors`` keeps valid items and records failed item reasons;
    unsafe paths, XML, unsupported required extensions and all budget failures
    remain fatal. Zero valid included instances always raises ThreeMFError.
    ``include_nonprintable`` includes those items with printable=False and a
    reconstruction blocker, rather than silently changing their purpose.

    Bambu model_settings.config role/plate/assembly evidence is bounded and uses
    the same safe XML parser/budgets. Only unique direct part mappings resolve;
    unsupported, incomplete or non-material roles veto whole-instance rebuilding.
    Private matrices/assembly poses are never applied or URI strings followed.
    See metadata['slicer_metadata']; role-only completeness is NOT manufacturing
    recovery, core-only packages without slicer evidence retain legacy behavior.

    Optional metadata is bounded/truncated with explicit source-table summaries.
    Raw coordinates remain in the source 3MF, referenced by model/object/vertex
    IDs. No CAD, normal repair, boolean union, material or color recovery occurs.
    Missing/unreadable files retain ordinary filesystem exceptions. Budgets are
    cumulative, including failed expansion attempts, never reset per build item.
    """
    source = Path(path)
    if source.stat().st_size > limits.max_archive_bytes:
        raise ThreeMFError("Compressed archive size limit exceeded")
    try:
        with zipfile.ZipFile(source) as archive:
            return _Package(archive, limits, source_path=str(source.resolve()),
                            geometry_only=geometry_only, isolate_build_errors=isolate_build_errors,
                            include_nonprintable=include_nonprintable).load()
    except _PackageError as exc:
        # Keep the public API/error_type stable; the private distinction only
        # controls whether a failure may be isolated inside the reader.
        raise ThreeMFError(str(exc)) from exc
    except ThreeMFError:
        raise
    except (zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError, NotImplementedError,
            EOFError, zlib.error, UnicodeError) as exc:
        raise ThreeMFError(f"Invalid/unsupported 3MF ZIP: {exc}") from exc


def load_3mf(path: str | Path, *, limits: ThreeMFLimits = DEFAULT_LIMITS) -> list[trimesh.Trimesh]:
    """Compatibility API: one mesh per printable build item, with mesh.metadata."""
    return [instance.mesh for instance in load_document(path, limits=limits).instances]
