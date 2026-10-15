"""Conservative, explanatory Bambu model_settings.config evidence, never CSG.

Only the conventional config member and unique, direct entry-object component
associations are understood. Private instance IDs, matrices, source files and
thumbnail URIs are opaque strings, not references or geometry instructions.
XML MUST come from _Package.xml: its DTD rejection and all cumulative XML budgets
also apply to configs. Optional evidence uses the package's shared metadata
entry/character budget; any loss or unsupported content vetoes reconstruction.

material_semantics_complete has the deliberately narrow ROLE_SCOPE below. It
never certifies printed material, appearance, manufacturing settings or a CAD
solid. With no slicer evidence it is False and mapping is not_applicable (the
legacy core-only gate is unchanged). normal_part never grants build approval.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any

CONFIG_MEMBER = "Metadata/model_settings.config"
GEOMETRY_FRAME = "3MF_core_build_world_mm_not_slicer_assembly_pose"
ROLE_SCOPE = "declared_part_roles_for_included_core_geometry_only_not_manufacturing_semantics"
_VERSION = 1
_ROLES = frozenset({"normal_part", "negative_part", "modifier_part", "parameter_modifier",
                    "modifier", "support_enforcer", "support_blocker", "support_part"})
# These are disclosed, not opened. Their contents/manufacturing effects are NOT
# part of the role-only completeness claim.
_KNOWN_UNREAD = re.compile(r"Metadata/(?:project_settings|slice_info|filament_settings_[0-9]+)\.config\Z")
_METADATA_KEYS = frozenset({
    "name", "extruder", "matrix", "source_file", "source_object_id", "source_volume_id",
    "source_offset_x", "source_offset_y", "source_offset_z", "plater_id", "plater_name",
    "locked", "filament_map_mode", "filament_maps", "filament_volume_maps",
    "thumbnail_file", "thumbnail_no_light_file", "top_file", "pick_file", "object_id",
    "instance_id", "identify_id",
})
_MESH_STATS = {"face_count", "edges_fixed", "degenerate_facets", "facets_removed",
               "facets_reversed", "backwards_edges"}


def is_role_config_member(name: str) -> bool:
    """Recognize suspicious aliases too, without accepting them as canonical."""
    return name.rsplit("/", 1)[-1].casefold() == "model_settings.config"


def _id(text):
    if not isinstance(text, str) or not re.fullmatch(r"[1-9][0-9]{0,99}", text):
        return None
    return int(text)


def _core_id(text):
    # Core IDs accept surrounding whitespace, a plus sign and leading zeros.
    # Count them by their already-validated numeric identity, without treating
    # the much narrower private config ID convention as a core XML rule.
    if not isinstance(text, str) or not re.fullmatch(r"[+]?[0-9]+", text.strip()):
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return value if value > 0 else None


def _unique(values):
    return list(dict.fromkeys(values))


def _children(node, locator):
    """Real XPath sibling positions, NOT positions among differently named tags."""
    counts = Counter()
    known = {"object", "part", "metadata", "mesh_stat", "plate", "model_instance",
             "assemble", "assemble_item"}
    for position, child in enumerate(node, 1):
        counts[child.tag] += 1
        step = f"{child.tag}[{counts[child.tag]}]" if child.tag in known else f"*[{position}]"
        yield child, f"{locator}/{step}"


class SlicerMetadata:
    """Internal reader adapter. No ZIP read or XML parser of its own."""
    def __init__(self, package):
        self.package = package
        self.issues: list[str] = []
        self.objects: list[dict[str, Any]] = []
        self.plates: list[dict[str, Any]] = []
        self.assembly: list[dict[str, Any]] = []
        self.uninterpreted: list[dict[str, Any]] = []
        self.uninterpreted_omitted = 0
        self.omitted = self.truncated = 0
        self.members = sorted(n for n in package.members if is_role_config_member(n))
        self.private_members = sorted(n for n in package.members if n.casefold().endswith(".config"))
        self.unknown_members = [n for n in self.private_members
                                if n != CONFIG_MEMBER and not _KNOWN_UNREAD.fullmatch(n)]
        self.status = "absent"
        self.assembly_present = False
        self.plate_count = 0
        # Even ambiguous configs go through the SAME fatal security parser. A
        # valid canonical member may not conceal an alias containing a DTD.
        roots = [(name, package.xml(name)[0]) for name in self.members]
        if roots:
            self.status = "parsed" if self.members == [CONFIG_MEMBER] else "ambiguous"
            if self.status == "ambiguous":
                self.issue("noncanonical_or_multiple_model_settings_configs")
            for name, root in roots:
                if name == CONFIG_MEMBER:
                    self.parse(root)
        if self.unknown_members:
            self.issue("unknown_private_configs_not_interpreted")

    def issue(self, reason):
        if reason not in self.issues:
            self.issues.append(reason)

    def snapshot(self, element, locator):
        """Keep bounded opaque strings; no URI resolution, transforms or repair."""
        p = self.package
        limits = p.limits
        if len(p.source_metadata) + p.slicer_metadata_entries >= limits.max_metadata_entries:
            self.omitted += 1
            self.issue("config_evidence_omitted")
            return None
        p.slicer_metadata_entries += 1
        truncated = False

        def bound(text):
            nonlocal truncated
            remaining = max(0, limits.max_metadata_total_chars - p.metadata_chars)
            result = text[:min(remaining, limits.max_metadata_text_chars)]
            p.metadata_chars += len(result)
            p.slicer_metadata_chars += len(result)
            truncated |= len(result) != len(text)
            return result

        attributes = {}
        for index, (key, value) in enumerate(element.attrib.items()):
            if index >= limits.max_metadata_attributes:
                truncated = True
                break
            attributes[bound(key)] = bound(value)
        text = bound(element.text or "")
        if truncated:
            self.truncated += 1
            self.issue("config_evidence_truncated")
        return {"source_config_member": CONFIG_MEMBER, "source_config_locator": locator,
                "attributes": attributes, "text": text, "truncated": truncated}

    def inspect(self, node, locator, attributes):
        if (set(node.attrib) - set(attributes) or (node.text or "").strip()
                or (node.tail or "").strip()):
            self.unhandled(node, locator, "unknown_attributes_or_text")

    def unhandled(self, node, locator, reason="unsupported_element"):
        self.issue("unsupported_config_content")
        if len(self.uninterpreted) < self.package.limits.max_ignored_content_summaries:
            self.uninterpreted.append({"source_config_locator": locator, "tag": node.tag[:256],
                                       "reason": reason, "includes_descendants": True,
                                       "handling": "not_interpreted_not_recovered"})
        else:
            self.uninterpreted_omitted += 1

    def metadata(self, node, locator):
        result = []
        keys = set()
        for child, loc in _children(node, locator):
            if child.tag != "metadata":
                continue
            self.inspect(child, loc, {"key", "value", "face_count"})
            key = child.get("key")
            if key is not None:
                if key not in _METADATA_KEYS:
                    self.unhandled(child, loc, "unknown_metadata_key")
                if key in keys:
                    self.issue("duplicate_config_metadata_key")
                keys.add(key)
                if "value" not in child.attrib or "face_count" in child.attrib:
                    self.issue("malformed_config_metadata")
            elif set(child.attrib) != {"face_count"}:
                self.issue("malformed_config_metadata")
            elif "@face_count" in keys:
                self.issue("duplicate_config_metadata_key")
            else:
                keys.add("@face_count")
            if len(child):
                self.unhandled(child, loc, "nested_metadata_not_interpreted")
            record = self.snapshot(child, loc)
            if record is not None:
                result.append(record)
        return result

    @staticmethod
    def name(metadata):
        names = [m["attributes"].get("value", "") for m in metadata
                 if m["attributes"].get("key") == "name"]
        return names[0] if len(names) == 1 else ""

    def parse(self, root):
        self.inspect(root, "config", set())
        if root.tag != "config":
            self.status = "unsupported" if self.status == "parsed" else self.status
            self.unhandled(root, "config", "unsupported_config_root")
            return
        object_ids, plate_ids, assemble_ids, plate_instance_ids = set(), set(), set(), set()
        self.assembly_present = root.find("assemble") is not None
        self.plate_count = len(root.findall("plate"))
        assemble_count = 0
        for node, loc in _children(root, "config"):
            record = self.snapshot(node, loc)
            if record is None:
                continue
            if node.tag == "object":
                self.inspect(node, loc, {"id"})
                object_id = _id(record["attributes"].get("id"))
                if object_id is None:
                    self.issue("invalid_config_object_id")
                if object_id in object_ids:
                    self.issue("duplicate_config_object_id")
                object_ids.add(object_id)
                metadata = self.metadata(node, loc)
                parts, part_ids = [], set()
                for part, ploc in _children(node, loc):
                    if part.tag == "metadata":
                        continue
                    if part.tag != "part":
                        self.unhandled(part, ploc)
                        continue
                    prec = self.snapshot(part, ploc)
                    if prec is None:
                        continue
                    self.inspect(part, ploc, {"id", "subtype", "uuid"})
                    part_id = _id(prec["attributes"].get("id"))
                    if part_id is None:
                        self.issue("invalid_config_part_id")
                    if part_id in part_ids:
                        self.issue("duplicate_config_part_id")
                    part_ids.add(part_id)
                    meta = self.metadata(part, ploc)
                    stats = []
                    for stat, sloc in _children(part, ploc):
                        if stat.tag == "metadata":
                            continue
                        if stat.tag != "mesh_stat":
                            self.unhandled(stat, sloc)
                            continue
                        self.inspect(stat, sloc, _MESH_STATS)
                        if len(stat):
                            self.unhandled(stat, sloc, "nested_mesh_stat")
                        srec = self.snapshot(stat, sloc)
                        if srec is not None:
                            stats.append(srec)
                    if len(stats) > 1:
                        self.issue("duplicate_mesh_stat")
                    parts.append({"part_id": part_id, "subtype": prec["attributes"].get("subtype"),
                                  "name": self.name(meta), "source": prec, "metadata": meta,
                                  "mesh_stat": stats})
                self.objects.append({"object_id": object_id, "name": self.name(metadata),
                                     "source": record, "metadata": metadata, "parts": parts})
            elif node.tag == "plate":
                self.inspect(node, loc, set())
                metadata = self.metadata(node, loc)
                ids = [m["attributes"].get("value") for m in metadata
                       if m["attributes"].get("key") == "plater_id"]
                if len(ids) != 1 or _id(ids[0]) is None or ids[0] in plate_ids:
                    self.issue("missing_or_duplicate_plate_id")
                plate_ids.update(ids)
                model_instances = []
                for child, cloc in _children(node, loc):
                    if child.tag == "metadata":
                        continue
                    if child.tag != "model_instance":
                        self.unhandled(child, cloc)
                        continue
                    self.inspect(child, cloc, set())
                    crec = self.snapshot(child, cloc)
                    if crec is None:
                        continue
                    cmeta = self.metadata(child, cloc)
                    values = {m["attributes"].get("key"): m["attributes"].get("value") for m in cmeta}
                    identity = (values.get("object_id"), values.get("instance_id"))
                    if (_id(identity[0]) is None or not isinstance(identity[1], str)
                            or not re.fullmatch(r"0|[1-9][0-9]{0,99}", identity[1])
                            or identity in plate_instance_ids):
                        self.issue("missing_or_duplicate_plate_instance_identity")
                    plate_instance_ids.add(identity)
                    for nested, nloc in _children(child, cloc):
                        if nested.tag != "metadata":
                            self.unhandled(nested, nloc)
                    model_instances.append({"source": crec, "source_config_locator": cloc, "metadata": cmeta})
                self.plates.append({"source": record, "metadata": metadata,
                                    "model_instances": model_instances,
                                    "core_build_instance_mapping": "not_inferred_from_private_instance_ids"})
            elif node.tag == "assemble":
                self.assembly_present = True
                assemble_count += 1
                self.inspect(node, loc, set())
                items = []
                for child, aloc in _children(node, loc):
                    if child.tag != "assemble_item":
                        self.unhandled(child, aloc)
                        continue
                    self.inspect(child, aloc, {"object_id", "instance_id", "volume_id", "transform", "offset"})
                    if len(child):
                        self.unhandled(child, aloc, "nested_assemble_item")
                    identity = tuple(child.get(k) for k in ("object_id", "instance_id", "volume_id"))
                    item_index = identity[1] if identity[1] is not None else identity[2]
                    if (identity in assemble_ids or _id(identity[0]) is None
                            or (identity[1] is None) == (identity[2] is None)
                            or not isinstance(item_index, str)
                            or not re.fullmatch(r"0|[1-9][0-9]{0,99}", item_index)):
                        self.issue("ambiguous_assemble_item_identity")
                    if not child.get("transform", "").strip():
                        self.issue("missing_assemble_transform")
                    assemble_ids.add(identity)
                    arec = self.snapshot(child, aloc)
                    if arec is not None:
                        items.append(arec)
                self.assembly.append({"source": record, "items": items,
                                      "transforms_validated": False, "transforms_applied": False})
            else:
                self.unhandled(node, loc)
        if assemble_count > 1:
            self.issue("duplicate_assemble_groups")

    def attach(self, entry, instances, *, coverage_complete=True):
        """Attach whole-instance vetoes, preserving every core mesh/transform."""
        required = bool(self.private_members or self.package.slicer_config_expected)
        if self.status == "absent" and required:
            self.status = "missing"
            self.issue("expected_model_settings_config_missing")
        by_id = defaultdict(list)
        for obj in self.objects:
            by_id[obj["object_id"]].append(obj)
        id_paths = defaultdict(set)
        for model in self.package.models.values():
            for object_id in model.objects:
                id_paths[object_id].add(model.path)
        # Count all entry build items, including skipped/failed ones. Never use
        # private instance_id as a build ordinal or to pick a duplicate occurrence.
        build_ids = Counter(_core_id(item.get("objectid")) for item in entry.build)
        if any(obj["object_id"] not in build_ids for obj in self.objects):
            self.issue("config_object_not_associated_with_entry_build")
        complete = not (self.omitted or self.truncated)
        for instance in instances:
            md = instance.metadata
            candidates = by_id.get(md["object_id"], [])
            parts = candidates[0]["parts"] if len(candidates) == 1 else []
            part_lookup = defaultdict(list)
            for pi, part in enumerate(parts):
                part_lookup[part["part_id"]].append(pi)
            leaf_ids = Counter(c["object_id"] for c in md["components"])
            conflicts = sorted({oid for oid in [md["object_id"], *leaf_ids]
                                if len(id_paths[oid]) > 1})
            leaves, used = [], set()
            for c in md["components"]:
                choices = part_lookup.get(c["object_id"], [])
                reasons = []
                applicable = self.status != "absent" or required
                if applicable:
                    if self.status != "parsed":
                        reasons.append("config_not_uniquely_supported")
                    if self.issues:
                        reasons.append("config_contains_unresolved_or_unsupported_content")
                    if len(candidates) != 1:
                        reasons.append("missing_or_duplicate_config_object")
                    if md["model_path"] != entry.path:
                        reasons.append("build_root_not_in_entry_model")
                    if build_ids[md["object_id"]] != 1:
                        reasons.append("repeated_build_object_occurrence")
                    chain = c["component_path"]
                    if (len(chain) != 1 or chain[0]["parent_model_path"] != entry.path
                            or chain[0]["parent_object_id"] != md["object_id"]):
                        reasons.append("not_a_direct_entry_object_component")
                    if leaf_ids[c["object_id"]] != 1:
                        reasons.append("repeated_leaf_id_or_occurrence")
                    if c["object_id"] in conflicts or md["object_id"] in conflicts:
                        reasons.append("cross_model_id_conflict")
                    if len(choices) != 1:
                        reasons.append("missing_or_duplicate_config_part")
                pi = choices[0] if applicable and not reasons else None
                part = parts[pi] if pi is not None else None
                if pi is not None:
                    used.add(pi)
                if part is not None and part["subtype"] not in _ROLES:
                    reasons.append("unknown_or_missing_slicer_subtype")
                leaf = {
                    "component_index": c["component_index"], "model_path": c["model_path"],
                    "object_id": c["object_id"],
                    "mapping_status": ("not_applicable" if not applicable else
                                       "unresolved" if reasons else "resolved"),
                    "mapping_issues": reasons, "config_part_index": pi,
                    "slicer_subtype": part["subtype"] if part else None,
                    "source_config_member": CONFIG_MEMBER if part else None,
                    "source_config_locator": part["source"]["source_config_locator"] if part else None,
                    "config_name": part["name"] if part else None,
                    "instance_face_range_half_open": [c["face_start"], c["face_end"]],
                    "boolean_difference_or_union_evaluated": False,
                }
                leaves.append(leaf)
                c.update(slicer_role=leaf["slicer_subtype"], slicer_role_mapping_status=leaf["mapping_status"],
                         slicer_role_evidence=leaf, slicer_config_member=leaf["source_config_member"],
                         slicer_config_locator=leaf["source_config_locator"], slicer_config_name=leaf["config_name"],
                         instance_face_range_half_open=leaf["instance_face_range_half_open"].copy())
            # Parts are independent declarations, not reconstructed from a copied
            # blocker list or from a caller's material-role summary.
            evidence_parts = [{key: part[key] for key in ("part_id", "subtype", "name", "source")}
                              for part in parts]
            evidence = {
                "schema_version": _VERSION, "config_status": self.status, "config_required": required,
                "cross_model_id_check_scope": "all_indexed_objects_in_loaded_core_model_parts",
                "config_members": self.members.copy(), "records_complete": complete,
                "config_issues": self.issues.copy(), "unknown_private_config_members": self.unknown_members.copy(),
                "config_object_count": len(candidates),
                "config_object_source": candidates[0]["source"] if len(candidates) == 1 else None,
                "root_build_occurrence_count": build_ids[md["object_id"]],
                "cross_model_id_conflicts": conflicts, "parts": evidence_parts,
                "unmatched_config_part_count": len(parts) - len(used),
                "component_count": len(leaves), "instance_face_count": len(instance.mesh.faces), "leaves": leaves,
            }
            not_applicable = self.status == "absent" and not required
            resolved = (self.status == "parsed" and complete and not self.issues and not conflicts
                        and len(candidates) == 1 and parts and len(used) == len(parts)
                        and all(l["mapping_status"] == "resolved" for l in leaves)
                        and all(p["subtype"] in _ROLES for p in parts))
            md.update(
                slicer_metadata_version=_VERSION, slicer_role_evidence=evidence,
                document_coverage_complete=coverage_complete,
                role_mapping_status="not_applicable" if not_applicable else "resolved" if resolved else "unresolved",
                material_semantics_complete=bool(resolved and all(p["subtype"] == "normal_part" for p in parts)),
                material_semantics_scope=ROLE_SCOPE, reconstruction_veto=False,
                geometry_frame=GEOMETRY_FRAME, slicer_assembly_transform_applied=False,
                slicer_private_transforms_applied=False, printed_material_geometry_recovered=False,
                slicer_plate_assembly_context={
                    "source_config_member": CONFIG_MEMBER if CONFIG_MEMBER in self.members else None,
                    "plate_count": self.plate_count, "assembly_pose_present_not_applied": self.assembly_present,
                    "mapping": "package_declarations_only_private_instance_ids_not_mapped_to_core_build_items",
                },
            )
            blockers = source_role_blockers(md)
            if not coverage_complete:
                blockers.append("document_coverage_complete=false")
            md["reconstruction_blockers"] = _unique(md["reconstruction_blockers"] + blockers)
            md["reconstruction_veto"] = bool(md["reconstruction_blockers"])
            md["reconstruction_eligible"] = not md["reconstruction_blockers"]
            md["reconstruction_eligibility_basis"] = "core_type_printability_and_slicer_source_vetoes_only_not_build_approval"
            # Trimesh may copy the supplied metadata dict in its constructor.
            instance.mesh.metadata.update(md)
        statuses = {i.metadata["role_mapping_status"] for i in instances}
        role_counts = Counter(c["slicer_role"] for i in instances for c in i.metadata["components"]
                              if c["slicer_role_mapping_status"] == "resolved")
        unresolved_count = sum(c["slicer_role_mapping_status"] == "unresolved"
                               for i in instances for c in i.metadata["components"])
        return {
            "schema_version": _VERSION, "config_status": self.status,
            "config_member": CONFIG_MEMBER if CONFIG_MEMBER in self.members else None,
            "config_members": self.members, "config_required": required,
            "private_config_discovery_scope": "archive_member_names_ending_dot_config_case_insensitive",
            "role_config_read_scope": "canonical_member_plus_basename_aliases_for_security_only",
            "private_config_members": [{"member": n, "handling": (
                "bounded_role_metadata_xml" if n == CONFIG_MEMBER else
                "known_filename_contents_not_interpreted" if _KNOWN_UNREAD.fullmatch(n) else
                "unknown_private_config_not_interpreted")} for n in self.private_members],
            "unknown_private_config_members": self.unknown_members,
            "config_issues": self.issues, "objects": self.objects, "plates": self.plates,
            "assembly": self.assembly, "assembly_pose_present_not_applied": self.assembly_present,
            "uninterpreted_content": self.uninterpreted,
            "uninterpreted_content_summaries_omitted": self.uninterpreted_omitted,
            "evidence_summary": {"records_preserved": self.package.slicer_metadata_entries,
                                 "records_omitted": self.omitted, "records_truncated": self.truncated,
                                 "characters_preserved": self.package.slicer_metadata_chars,
                                 "complete": complete, "budget_scope": "shared_with_core_source_metadata"},
            "role_mapping_status": ("not_applicable" if statuses == {"not_applicable"} else
                                    "resolved" if statuses == {"resolved"} else "unresolved"),
            "material_semantics_complete": coverage_complete and all(i.metadata["material_semantics_complete"] for i in instances),
            "material_semantics_scope": ROLE_SCOPE,
            "resolved_role_occurrence_counts": dict(role_counts),
            "negative_component_occurrence_count": role_counts["negative_part"],
            "unresolved_component_occurrence_count": unresolved_count,
            "reconstruction_veto": not coverage_complete or any(i.metadata["reconstruction_veto"] for i in instances),
            "reconstruction_blockers": _unique(
                [b for i in instances for b in i.metadata["reconstruction_blockers"]]
                + ([] if coverage_complete else ["document_coverage_complete=false"])),
            "reconstruction_veto_scope": "whole_build_instance_roles_and_package_entry_build_coverage_propagated_to_instances",
            "geometry_frame": GEOMETRY_FRAME, "slicer_assembly_transform_applied": False,
            "slicer_private_transforms_applied": False, "printed_material_geometry_recovered": False,
            "reconstruction_authorized_by_metadata": False,
            "known_scope": [
                "material_semantics_complete is role-only: every included leaf maps to normal_part; it is not manufacturing completeness.",
                "Unique direct entry-object part roles only; unknown, nested or repeated mappings are not guessed.",
                "ID conflicts are checked across indexed objects of loaded core model parts, not unread/unreferenced models.",
                "Package completeness is limited to entry-build geometry coverage and these declared roles, not unused resources.",
                "Plate and assembly declarations are opaque context, not core build item identities or placement.",
                "Private matrices, repair statistics, source files and thumbnail URIs are preserved only; never followed or applied.",
                "Other private config contents, appearance and manufacturing settings are not recovered.",
                "No union, difference, mesh repair, design intent or printed-material solid is recovered.",
                "Absent config without any slicer evidence is not_applicable, not a claim that material semantics were recovered.",
                "A missing expected config, unknown private config, conflicting records or evidence loss vetoes rebuilding.",
            ],
        }


def source_role_blockers(provenance) -> list[str]:
    """Strict, pure role-evidence gate for new provenance; legacy remains valid.

    Recompute vetoes from raw part declarations AND occurrence evidence, not the
    top-level blocker/eligibility flags. Deleting one new mandatory field is a
    veto. This is validation of supplied provenance, not cryptographic source
    authentication; callers must retain the original source/evidence as well.
    """
    if not isinstance(provenance, dict):
        return ["slicer_source_provenance_must_be_an_object"]
    md = provenance
    fields = {"slicer_metadata_version", "slicer_role_evidence", "role_mapping_status",
              "material_semantics_complete", "material_semantics_scope", "reconstruction_veto",
              "geometry_frame", "slicer_assembly_transform_applied", "slicer_private_transforms_applied",
              "printed_material_geometry_recovered", "slicer_plate_assembly_context"}
    components = md.get("components")
    leaf_marker = isinstance(components, list) and any(
        isinstance(c, dict) and any(isinstance(k, str) and k.startswith("slicer_") for k in c)
        for c in components)
    if not fields.intersection(md) and not leaf_marker:
        return []
    blockers = []

    def add(reason):
        if reason not in blockers:
            blockers.append(reason)

    def invalid(label):
        add("invalid_slicer_source_field:" + label)

    def boolean(obj, key, label):
        value = obj.get(key)
        if type(value) is not bool:
            invalid(label)
        return value

    def integer(obj, key, label, minimum=0):
        value = obj.get(key)
        if type(value) is not int or value < minimum:
            invalid(label)
            return None
        return value

    def strings(obj, key, label):
        value = obj.get(key)
        if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() or x != x.strip() for x in value):
            invalid(label)
            return []
        return value

    def raw_source(record, attributes, label):
        if not isinstance(record, dict):
            invalid(label)
            return
        attrs = record.get("attributes")
        text = record.get("text")
        if (not isinstance(attrs, dict) or set(attrs) - attributes
                or any(not isinstance(k, str) or not isinstance(v, str) for k, v in attrs.items())
                or not isinstance(text, str) or text.strip() or record.get("truncated") is not False):
            invalid(label)

    if type(md.get("slicer_metadata_version")) is not int or md["slicer_metadata_version"] != _VERSION:
        invalid("slicer_metadata_version")
    for key in ("slicer_assembly_transform_applied", "slicer_private_transforms_applied",
                "printed_material_geometry_recovered"):
        if boolean(md, key, key) is True:
            add("unsupported_slicer_geometry_or_manufacturing_recovery:" + key)
    if md.get("geometry_frame") != GEOMETRY_FRAME:
        invalid("geometry_frame")
    if md.get("material_semantics_scope") != ROLE_SCOPE:
        invalid("material_semantics_scope")
    context = md.get("slicer_plate_assembly_context")
    if not isinstance(context, dict):
        invalid("slicer_plate_assembly_context")
    else:
        integer(context, "plate_count", "slicer_plate_assembly_context.plate_count")
        boolean(context, "assembly_pose_present_not_applied", "slicer_plate_assembly_context.assembly_pose_present_not_applied")
        if ("source_config_member" not in context or context.get("source_config_member") not in (None, CONFIG_MEMBER)
                or context.get("mapping") != "package_declarations_only_private_instance_ids_not_mapped_to_core_build_items"):
            invalid("slicer_plate_assembly_context.scope")
    if boolean(md, "reconstruction_veto", "reconstruction_veto") is True:
        add("source_reconstruction_veto")
    material_complete = boolean(md, "material_semantics_complete", "material_semantics_complete")
    status = md.get("role_mapping_status")
    if status not in ("not_applicable", "resolved", "unresolved"):
        invalid("role_mapping_status")
    elif status == "unresolved":
        add("slicer_role_mapping_unresolved")
    evidence = md.get("slicer_role_evidence")
    if not isinstance(evidence, dict):
        invalid("slicer_role_evidence")
        return blockers
    ev = evidence
    if type(ev.get("schema_version")) is not int or ev["schema_version"] != _VERSION:
        invalid("evidence.schema_version")
    if ev.get("cross_model_id_check_scope") != "all_indexed_objects_in_loaded_core_model_parts":
        invalid("evidence.cross_model_id_check_scope")
    config_status = ev.get("config_status")
    if config_status not in ("absent", "missing", "parsed", "unsupported", "ambiguous"):
        invalid("evidence.config_status")
    required = boolean(ev, "config_required", "evidence.config_required")
    complete = boolean(ev, "records_complete", "evidence.records_complete")
    if complete is False:
        add("slicer_config_evidence_incomplete")
    members = strings(ev, "config_members", "evidence.config_members")
    issues = strings(ev, "config_issues", "evidence.config_issues")
    unknown = strings(ev, "unknown_private_config_members", "evidence.unknown_private_config_members")
    if isinstance(context, dict) and context.get("source_config_member") != (CONFIG_MEMBER if CONFIG_MEMBER in members else None):
        invalid("slicer_plate_assembly_context.config_consistency")
    if issues or unknown:
        add("slicer_config_unresolved_or_unsupported")
    count = integer(ev, "config_object_count", "evidence.config_object_count")
    occurrences = integer(ev, "root_build_occurrence_count", "evidence.root_build_occurrence_count", 1)
    unmatched = integer(ev, "unmatched_config_part_count", "evidence.unmatched_config_part_count")
    component_count = integer(ev, "component_count", "evidence.component_count", 1)
    face_count = integer(ev, "instance_face_count", "evidence.instance_face_count", 1)
    inactive = config_status == "absent" and required is False
    conflicts = ev.get("cross_model_id_conflicts")
    if not isinstance(conflicts, list) or any(type(x) is not int or x <= 0 for x in conflicts):
        invalid("evidence.cross_model_id_conflicts")
    elif conflicts and not inactive:
        add("slicer_cross_model_id_conflict")
    object_source = ev.get("config_object_source")
    if count == 1 and not inactive:
        raw_source(object_source, {"id"}, "evidence.config_object_raw_source")
        if not isinstance(object_source, dict):
            invalid("evidence.config_object_source")
        else:
            attrs = object_source.get("attributes")
            root_id = md.get("object_id")
            if (type(root_id) is not int or root_id <= 0 or not isinstance(attrs, dict)
                    or _id(attrs.get("id")) != root_id or object_source.get("truncated") is not False
                    or object_source.get("source_config_member") != CONFIG_MEMBER
                    or not isinstance(object_source.get("source_config_locator"), str)
                    or not re.fullmatch(r"config/object\[[1-9][0-9]*\]", object_source["source_config_locator"])):
                invalid("evidence.config_object_raw_declaration")
    elif "config_object_source" not in ev or object_source is not None:
        invalid("evidence.config_object_source")
    if inactive:
        if members or issues or unknown or count != 0 or unmatched != 0 or material_complete is not False or status != "not_applicable":
            invalid("not_applicable_evidence_consistency")
    else:
        if config_status != "parsed" or members != [CONFIG_MEMBER]:
            add("slicer_config_missing_ambiguous_or_unsupported")
        if count != 1 or occurrences != 1 or unmatched != 0:
            add("slicer_config_object_or_occurrence_mapping_unresolved")
        if status != "resolved":
            add("slicer_role_mapping_unresolved")
        if material_complete is False:
            add("slicer_material_semantics_incomplete")

    parts, leaves = ev.get("parts"), ev.get("leaves")
    if not isinstance(parts, list) or not isinstance(leaves, list):
        invalid("evidence.parts_or_leaves")
        return blockers
    if (inactive and parts) or (not inactive and not parts):
        invalid("evidence.parts")
    part_ids = set()
    for part in parts:
        if not isinstance(part, dict):
            invalid("evidence.part")
            continue
        part_id = integer(part, "part_id", "evidence.part.part_id", 1)
        if part_id in part_ids:
            add("slicer_duplicate_config_part_id")
        part_ids.add(part_id)
        role = part.get("subtype")
        if role != "normal_part":
            add("slicer_non_material_role:" + role if isinstance(role, str) and role in _ROLES else "slicer_unknown_role")
        source = part.get("source")
        raw_source(source, {"id", "subtype", "uuid"}, "evidence.part.raw_source")
        if not isinstance(source, dict):
            invalid("evidence.part.source")
        else:
            attrs = source.get("attributes")
            locator = source.get("source_config_locator")
            object_locator = object_source.get("source_config_locator") if isinstance(object_source, dict) else None
            if (source.get("source_config_member") != CONFIG_MEMBER
                    or not isinstance(locator, str)
                    or not re.fullmatch(r"config/object\[[1-9][0-9]*\]/part\[[1-9][0-9]*\]", locator)
                    or not isinstance(object_locator, str) or not locator.startswith(object_locator + "/part[")
                    or source.get("truncated") is not False or not isinstance(attrs, dict)
                    or _id(attrs.get("id")) != part_id or attrs.get("subtype") != role):
                invalid("evidence.part.raw_declaration")
        if not isinstance(part.get("name"), str):
            invalid("evidence.part.name")
    if len(leaves) != component_count or not leaves:
        invalid("evidence.leaves_coverage")
    if not isinstance(components, list) or len(components) != component_count:
        invalid("components_coverage")
        components = []
    used, offset = set(), 0
    for index, leaf in enumerate(leaves):
        if not isinstance(leaf, dict):
            invalid("evidence.leaf")
            continue
        if type(leaf.get("component_index")) is not int or leaf["component_index"] != index:
            invalid("evidence.leaf.component_index")
        lid = integer(leaf, "object_id", "evidence.leaf.object_id", 1)
        if not isinstance(leaf.get("model_path"), str) or not leaf["model_path"]:
            invalid("evidence.leaf.model_path")
        reasons = strings(leaf, "mapping_issues", "evidence.leaf.mapping_issues")
        if boolean(leaf, "boolean_difference_or_union_evaluated", "evidence.leaf.boolean_evaluated") is True:
            add("unsupported_slicer_boolean_evaluation")
        expected = "not_applicable" if inactive else "resolved"
        if leaf.get("mapping_status") != expected or reasons:
            add("slicer_leaf_mapping_unresolved")
        pi = leaf.get("config_part_index")
        if inactive:
            if pi is not None or any(leaf.get(k) is not None for k in (
                    "slicer_subtype", "source_config_member", "source_config_locator", "config_name")):
                invalid("evidence.leaf.not_applicable")
        elif type(pi) is not int or not 0 <= pi < len(parts) or not isinstance(parts[pi], dict):
            invalid("evidence.leaf.config_part_index")
        else:
            if pi in used:
                add("slicer_repeated_part_occurrence")
            used.add(pi)
            part = parts[pi]
            source = part.get("source", {})
            if not isinstance(source, dict):
                source = {}
            if (lid != part.get("part_id") or leaf.get("slicer_subtype") != part.get("subtype")
                    or leaf.get("source_config_member") != CONFIG_MEMBER
                    or leaf.get("source_config_locator") != source.get("source_config_locator")
                    or leaf.get("config_name") != part.get("name")):
                invalid("evidence.leaf.declaration_consistency")
        span = leaf.get("instance_face_range_half_open")
        if (not isinstance(span, list) or len(span) != 2 or any(type(x) is not int for x in span)
                or span[0] != offset or span[1] <= span[0]):
            invalid("evidence.leaf.face_range")
        else:
            offset = span[1]
        if index < len(components):
            c = components[index]
            if not isinstance(c, dict):
                invalid("component")
                continue
            if (type(c.get("object_id")) is not int or type(c.get("component_index")) is not int
                    or c["component_index"] != index
                    or type(c.get("face_start")) is not int or type(c.get("face_end")) is not int
                    or c.get("slicer_role_evidence") != leaf or c.get("slicer_role") != leaf.get("slicer_subtype")
                    or c.get("slicer_role_mapping_status") != leaf.get("mapping_status")
                    or c.get("slicer_config_locator") != leaf.get("source_config_locator")
                    or c.get("slicer_config_member") != leaf.get("source_config_member")
                    or c.get("slicer_config_name") != leaf.get("config_name")
                    or c.get("object_id") != lid or c.get("model_path") != leaf.get("model_path")
                    or c.get("instance_face_range_half_open") != span
                    or [c.get("face_start"), c.get("face_end")] != span):
                invalid("component_role_evidence_consistency")
            if not inactive and leaf.get("mapping_status") == "resolved":
                chain = c.get("component_path")
                if (not isinstance(chain, list) or len(chain) != 1 or not isinstance(chain[0], dict)
                        or chain[0].get("parent_model_path") != md.get("entry_model")
                        or md.get("model_path") != md.get("entry_model")
                        or chain[0].get("parent_object_id") != md.get("object_id")
                        or chain[0].get("object_id") != lid or chain[0].get("model_path") != leaf.get("model_path")):
                    invalid("component_direct_mapping")
    if offset != face_count or (not inactive and len(used) != len(parts)):
        invalid("evidence.geometry_or_declaration_coverage")
    return blockers
