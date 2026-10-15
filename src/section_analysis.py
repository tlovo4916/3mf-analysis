"""Discrete section evidence in an input mesh's local XYZ frame (millimeters).

Public API:
    section_at(mesh, z, tolerance=0.05) -> SectionEvidence
    polygons_to_section(points_list, z=0.0, tolerance=0.05) -> SectionEvidence
    analyze_sections(mesh, tolerance=0.05) -> SectionStack

SectionEvidence has ``geometry: Polygon | MultiPolygon`` and ``metadata: dict``.
Loops must explicitly repeat their first point: neither user tolerance, implicit
Polygon closure, buffer(0), snapping nor make_valid is used to repair them.
Loop IDs follow input order; role is material_outer at depth 0, void at odd
depths, and material_island at positive even depths. Smallest containing
polygons determine parents, independently of winding or input order.

SectionStack has ``samples: list[SectionEvidence]`` and ``metadata: dict``.
Samples are successful slices, sorted by Z. Failed attempts are recorded in
metadata.failed_samples, including their original path coordinates when
available. reference_sample_index indexes samples, or is None if the mandatory
0.25 reference failed. profile_varies is True for observed variation, False
for agreement at all attempted samples, and None when failures prevent that
comparison. None is also used for unavailable distance metrics; no JSON NaN or
Infinity is emitted. metadata.event_probes holds separate, bounded one-sided
finite-offset evidence for ambiguous_section_event failures only. It never
replaces an exact attempt or participates in reference/profile comparisons.
sampling_complete is *always* False: discrete agreement is not proof, and this
module never authorizes automatic full-thickness CAD.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import trimesh
from shapely import STRtree, linestrings, points
from shapely.geometry import LinearRing, MultiPolygon, Polygon
from shapely.validation import explain_validity

__all__ = [
    "SectionEvidence", "SectionStack", "SectionTopologyError",
    "section_at", "polygons_to_section", "analyze_sections",
]

DEFAULT_FRACTIONS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
MAX_SAMPLES = 40
REFERENCE_FRACTION = 0.25
EVENT_PROBE_MAX_ATTEMPTS_PER_SIDE = 4
EVENT_PROBE_BUDGET_PER_PRIMARY_ATTEMPT = 2


@dataclass
class SectionEvidence:
    geometry: Polygon | MultiPolygon
    metadata: dict[str, Any]


@dataclass
class SectionStack:
    samples: list[SectionEvidence]
    metadata: dict[str, Any]


def _finite_scalar(value, name: str, *, nonnegative: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result) or (nonnegative and result < 0):
        suffix = " and nonnegative" if nonnegative else ""
        raise ValueError(f"{name} must be finite{suffix}")
    return result


def _json_safe(value):
    """Keep failure evidence serializable, marking invalid numbers as null."""
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    return repr(value)


class SectionTopologyError(ValueError):
    """Invalid section; ``code`` and JSON ``metadata`` retain failure evidence."""

    def __init__(self, code: str, message: str, *, z_mm: float,
                 raw_paths=(), details=None):
        self.code = code
        self.metadata = {
            "z_mm": z_mm,
            "topology_status": "invalid",
            "topology_valid": False,
            "error_code": code,
            "reason": message,
            "raw_paths_mm": _json_safe(raw_paths),
            "details": _json_safe(details or {}),
        }
        super().__init__(f"{code}: {message}")


def polygons_to_section(points_list, z=0.0, tolerance=0.05) -> SectionEvidence:
    """Validate explicitly closed XY paths and construct material by parity.

    points_list is an iterable of (N, 2) paths, including closing endpoints.
    Empty input returns an empty Polygon with topology_status='empty'. Invalid,
    open, intersecting, overlapping, or touching loops raise SectionTopologyError
    rather than returning approximate or repaired material. tolerance is a
    nonnegative finite comparison tolerance, *not* a loop-closing tolerance.
    """
    tolerance = _finite_scalar(tolerance, "tolerance", nonnegative=True)
    z = _finite_scalar(z, "z")
    raw_paths = list(points_list)
    polygons = []
    points = []

    def fail(code, message, **details):
        raise SectionTopologyError(code, message, z_mm=z,
                                   raw_paths=raw_paths, details=details)

    for index, raw in enumerate(raw_paths):
        try:
            xy = np.asarray(raw, dtype=float)
        except (TypeError, ValueError, OverflowError):
            fail("invalid_coordinates", "Loop coordinates must be numeric XY pairs.",
                 loop_id=index)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
            fail("invalid_coordinates", "Each path must have shape (N, 2), N >= 2.",
                 loop_id=index)
        if not np.isfinite(xy).all():
            fail("nonfinite_coordinates", "Loop coordinates must all be finite.",
                 loop_id=index)
        # Polygon/LinearRing would silently close an open path. Check first,
        # using exact endpoint identity, not the user's manufacturing tolerance.
        if not np.array_equal(xy[0], xy[-1]):
            fail("open_path", "Path does not explicitly return to its first point.",
                 loop_id=index, closure_gap_mm=float(np.linalg.norm(xy[-1] - xy[0])))
        if len(xy) < 4:
            fail("degenerate_loop", "Closed loops need at least three vertices.",
                 loop_id=index)
        polygon = Polygon(xy)
        if not LinearRing(xy).is_simple:
            fail("self_intersecting_loop", "Loop is not simple (crossing or self-touch).",
                 loop_id=index, validity_reason=explain_validity(polygon))
        if not math.isfinite(polygon.area) or polygon.area <= 0:
            fail("zero_or_nonfinite_area", "Loop must have finite, nonzero area.",
                 loop_id=index)
        if not polygon.is_valid:
            fail("invalid_loop", "Loop polygon is invalid.", loop_id=index,
                 validity_reason=explain_validity(polygon))
        points.append(xy)
        polygons.append(polygon)

    containers = [[] for _ in polygons]
    for i, a in enumerate(polygons):
        for j in range(i + 1, len(polygons)):
            b = polygons[j]
            if a.boundary.intersects(b.boundary):
                if a.equals(b) or a.boundary.intersection(b.boundary).length > 0:
                    kind = "overlapping_loop_boundaries"
                elif a.overlaps(b):
                    kind = "crossing_loops"
                else:
                    kind = "touching_loops"
                fail(kind, "Separate loops must be disjoint or strictly nested; "
                     "boundary crossing, overlap, and touch are not supported.",
                     loop_ids=[i, j])
            if a.contains(b):
                containers[j].append(i)
            elif b.contains(a):
                containers[i].append(j)
            elif not a.disjoint(b):
                fail("overlapping_loops", "Loop interiors overlap without containment.",
                     loop_ids=[i, j])

    parents = [min(candidates, key=lambda i: polygons[i].area) if candidates else None
               for candidates in containers]
    depths = [0] * len(polygons)
    children = [[] for _ in polygons]
    # A strict parent has greater area, so no recursion or order assumption is needed.
    for i in sorted(range(len(polygons)), key=lambda i: polygons[i].area, reverse=True):
        parent = parents[i]
        if parent is not None:
            depths[i] = depths[parent] + 1
            children[parent].append(i)

    material = [Polygon(points[i], holes=[points[j] for j in children[i]])
                for i in range(len(polygons)) if depths[i] % 2 == 0]
    geometry = (Polygon() if not material else material[0] if len(material) == 1
                else MultiPolygon(material))
    if not geometry.is_valid or not math.isfinite(geometry.area):
        fail("invalid_material_geometry", "Parity material geometry is invalid; no repair applied.",
             validity_reason=explain_validity(geometry))

    # Canonical unordered containment forest: detects changes in nesting even
    # when component counts, hole counts and maximum depth all remain the same.
    signatures = [""] * len(polygons)
    for i in sorted(range(len(polygons)), key=lambda i: polygons[i].area):
        role = "M" if depths[i] % 2 == 0 else "V"
        signatures[i] = role + "[" + ",".join(sorted(signatures[j] for j in children[i])) + "]"
    metadata = {
        "z_mm": z,
        "tolerance_mm": tolerance,
        "source": "provided_closed_loops",
        "coordinate_frame": "input_local_xy_mm",
        "topology_status": "valid" if polygons else "empty",
        "topology_valid": True,
        "loops": [
            {"id": i, "parent_id": parents[i], "depth": depths[i],
             "role": ("material_outer" if depths[i] == 0 else
                      "void" if depths[i] % 2 else "material_island"),
             "is_closed": True, "points_mm": points[i].tolist(),
             "area_mm2": float(polygons[i].area)}
            for i in range(len(polygons))
        ],
        "loop_count": len(polygons),
        "outer_region_count": sum(parent is None for parent in parents),
        "material_region_count": len(material),
        "void_loop_count": sum(depth % 2 for depth in depths),
        "max_nesting_depth": max(depths, default=0),
        "net_area_mm2": float(geometry.area),
        "topology_signature": sorted(signatures[i] for i in range(len(polygons))
                                     if parents[i] is None),
    }
    return SectionEvidence(geometry, metadata)


def _source_mesh_section_at(mesh, z, tolerance):
    """Intersect source triangles once per shared edge, without Path3D merging.

    Endpoint identity is a source vertex ID or an unordered source edge ID,
    never rounded XY. Vertex/coplanar events are retained but rejected as
    ambiguous, including apparently traversable events. Ignoring an exactly
    zero-length intersection does not join its distinct source endpoints.
    """
    node_ids, node_keys, node_xyz, incident = {}, [], [], []
    segments, ignored, raw_xy, walked = [], [], [], []
    coplanar_edges = {}
    traversed = 0
    cycles_complete = False
    extraction = {
        "source": "source_mesh_edge_intersections",
        "extraction_method": "source_edge_id_plane_intersection",
        "backend": "numpy_source_triangle_edges",
        "projection": "local_xy_without_transform",
        "numeric_on_plane_epsilon": 128 * np.finfo(float).eps * max(1.0, abs(z)),
        "numeric_on_plane_epsilon_rule": "128 * float64_eps * max(1, abs(z), max(abs(vertices)))",
        "source_triangle_id_scope": "input_mesh_faces",
        "source_vertex_id_scope": "input_mesh_vertices",
        "closure_method": "exact_source_endpoint_identity",
        "user_tolerance_used_for_extraction": False,
        "coordinate_welding_performed": False,
        "mesh_repair_performed": False,
        "mesh_winding_modified": False,
        "self_intersection_status": "not_checked",
        "solid_validity_certified": False,
        "max_plane_deviation_mm": 0.0,
        "on_plane_vertex_ids": [],
        "event_triangle_ids": [],
        "coplanar_triangle_ids": [],
        "ambiguity_kinds": [],
    }

    def provenance():
        return {
            **extraction,
            "intersection_nodes": [
                {"id": i, "source_key": list(key), "point_xyz_mm": node_xyz[i].tolist(),
                 "incident_segment_ids": incident[i]}
                for i, key in enumerate(node_keys)
            ],
            "segments": segments,
            "segment_count": len(segments),
            "traversed_segment_count": traversed,
            "all_segments_traversed_once": cycles_complete and traversed == len(segments),
            "ignored_degenerate_segments": ignored,
            "ignored_degenerate_segment_count": len(ignored),
            "coplanar_edges": [
                {"source_vertex_ids": list(key), "source_triangle_ids": ids}
                for key, ids in sorted(coplanar_edges.items())
            ],
        }

    def fail(code, reason, **details):
        paths = raw_xy or [[node_xyz[n][:2] for n in s["endpoint_ids"]] for s in segments]
        raise SectionTopologyError(code, reason, z_mm=z, raw_paths=paths, details=details)

    try:
        vertices, faces = np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces)
        if (vertices.ndim != 2 or vertices.shape[1:] != (3,)
                or faces.ndim != 2 or faces.shape[1:] != (3,)
                or not np.issubdtype(faces.dtype, np.integer)):
            fail("invalid_source_mesh", "Source mesh must provide XYZ vertices and integer triangles.")
        if not np.isfinite(vertices).all():
            fail("nonfinite_coordinates", "Source mesh XYZ coordinates must be finite.")
        if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
            fail("invalid_source_mesh", "Source triangle index is outside the vertex array.")
        referenced_ids = np.unique(faces)
        referenced = vertices[referenced_ids]
        extraction['unused_vertex_count'] = len(vertices) - len(referenced_ids)
        scale = max(1.0, abs(z), float(np.max(np.abs(referenced))) if len(referenced) else 0.0)
        epsilon = 128 * np.finfo(float).eps * scale
        extraction["numeric_on_plane_epsilon"] = float(epsilon)
        with np.errstate(over="ignore", invalid="ignore"):
            distances = vertices[:, 2] - z
        if not np.isfinite(distances).all():
            fail("nonfinite_plane_distances", "Source vertex-to-plane distances must be finite.")
        signs = (distances > epsilon).astype(np.int8) - (distances < -epsilon).astype(np.int8)
        face_signs = signs[faces]
        extraction["on_plane_vertex_ids"] = np.unique(faces[face_signs == 0]).tolist()
        extraction["event_triangle_ids"] = np.flatnonzero(np.any(face_signs == 0, axis=1)).tolist()
        extraction["coplanar_triangle_ids"] = np.flatnonzero(np.all(face_signs == 0, axis=1)).tolist()

        def node(key):
            if key not in node_ids:
                if key[0] == "vertex":
                    # Keep even near-on-plane event coordinates as supplied;
                    # numerical classification is NOT an endpoint snap.
                    xyz = vertices[key[1]].copy()
                else:
                    a, b = key[1:]
                    # Canonical source-edge order: all adjacent faces reuse the
                    # very same computation/ID, even with inconsistent winding.
                    denominator = float(distances[a]) - float(distances[b])
                    if not math.isfinite(denominator):
                        fail("nonfinite_edge_interpolation", "Float64 edge interpolation overflow; "
                             "no endpoint substitution is applied.", source_key=key)
                    t = distances[a] / denominator
                    with np.errstate(over="ignore", invalid="ignore"):
                        xy = vertices[a, :2] + t * (vertices[b, :2] - vertices[a, :2])
                    xyz = np.array([xy[0], xy[1], z])
                if not np.isfinite(xyz).all():
                    fail("nonfinite_coordinates", "Source intersection must be finite.", source_key=key)
                extraction["max_plane_deviation_mm"] = max(
                    extraction["max_plane_deviation_mm"], float(abs(xyz[2] - z)))
                node_ids[key] = len(node_xyz)
                node_keys.append(key)
                node_xyz.append(xyz)
                incident.append([])
            return node_ids[key]

        crossing = np.flatnonzero((face_signs.min(axis=1) <= 0) & (face_signs.max(axis=1) >= 0))
        for face_id in crossing.tolist():
            face = faces[face_id].tolist()
            zero = [v for v in face if signs[v] == 0]
            endpoints = [node(("vertex", v)) for v in zero]
            if len(zero) >= 2:
                # A coincident edge/triangle has no unique 1D slice. Do not
                # choose an adjacent face, remove duplicates, or invent a ring.
                for a, b in zip(face, face[1:] + face[:1]):
                    if signs[a] == signs[b] == 0:
                        coplanar_edges.setdefault((min(a, b), max(a, b)), []).append(face_id)
                continue
            for a, b in zip(face, face[1:] + face[:1]):
                if signs[a] * signs[b] < 0:
                    endpoints.append(node(("edge", min(a, b), max(a, b))))
            if len(endpoints) == 1:  # A tangent vertex; the event gate below rejects it.
                continue
            if len(endpoints) != 2:
                fail("ambiguous_triangle_intersection", "Triangle did not yield exactly two endpoints.",
                     source_triangle_id=face_id, endpoint_ids=endpoints)
            a, b = endpoints
            if a == b or np.array_equal(node_xyz[a], node_xyz[b]):
                ignored.append({"source_triangle_id": face_id, "endpoint_ids": endpoints,
                                "reason": "exact_zero_length_intersection"})
                continue
            segment_id = len(segments)
            segments.append({"id": segment_id, "source_triangle_id": face_id, "endpoint_ids": endpoints})
            incident[a].append(segment_id)
            incident[b].append(segment_id)

        if extraction["on_plane_vertex_ids"]:
            kinds = ["vertex_on_plane_event"]
            if coplanar_edges:
                kinds.append("coplanar_edges")
            if extraction["coplanar_triangle_ids"]:
                kinds.append("coplanar_triangles")
            extraction["ambiguity_kinds"] = kinds
            fail("ambiguous_section_event", "Plane meets source vertex/coplanar events; "
                 "a unique section is not certified and no stitching is applied.")

        # Isolated nodes belonging only to ignored zero-length intersections
        # are not in the segment graph. Never join them to degree-one ends.
        bad = [{"node_id": i, "source_key": list(node_keys[i]), "degree": len(ids),
                "incident_segment_ids": ids} for i, ids in enumerate(incident) if ids and len(ids) != 2]
        if bad:
            fail("open_or_branched_section", "Source-edge section graph has degree other than two; "
                 "no open or branched segments may be discarded.", bad_degree_nodes=bad)

        visited = np.zeros(len(segments), dtype=bool)
        # Ordered seeds, not repeated min(remaining): traversal is linear in the
        # segment count, and every segment must belong to exactly one cycle.
        for seed in range(len(segments)):
            if visited[seed]:
                continue
            start = current = segments[seed]["endpoint_ids"][0]
            segment_id = seed
            loop_nodes, loop_segments = [start], []
            while True:
                if visited[segment_id]:
                    fail("incomplete_section_cycles", "Cycle reuses a segment before returning to its start.",
                         segment_id=segment_id)
                visited[segment_id] = True
                traversed += 1
                loop_segments.append(segment_id)
                a, b = segments[segment_id]["endpoint_ids"]
                following = b if current == a else a
                loop_nodes.append(following)
                if following == start:  # Exact source endpoint identity, never a closure tolerance.
                    break
                choices = incident[following]
                current = following
                segment_id = choices[0] if choices[0] != segment_id else choices[1]
            raw_xy.append(np.asarray([node_xyz[n][:2] for n in loop_nodes]))
            walked.append({"intersection_node_ids": loop_nodes, "segment_ids": loop_segments,
                           "source_triangle_ids": [segments[s]["source_triangle_id"] for s in loop_segments]})
        if traversed != len(segments) or not visited.all():
            fail("incomplete_section_cycles", "Not every source segment was traversed exactly once.")
        cycles_complete = True
        evidence = polygons_to_section(raw_xy, z=z, tolerance=tolerance)
        for loop, membership in zip(evidence.metadata["loops"], walked):
            loop.update(membership)
        evidence.metadata.update(provenance())
        return evidence
    except SectionTopologyError as exc:
        exc.metadata.update(provenance())
        raise
    except Exception as exc:
        error = SectionTopologyError(
            "section_extraction_failed", f"Source triangle/edge extraction failed: {exc}",
            z_mm=z, raw_paths=raw_xy, details={"exception_type": type(exc).__name__})
        error.metadata.update(provenance())
        raise error from exc


def section_at(mesh, z, tolerance=0.05) -> SectionEvidence:
    """Slice local Z without welding source edges, snapping or mesh repair.

    Real Trimesh inputs use source-edge/vertex identity, reject ambiguous event
    planes and require every segment to traverse a degree-two closed cycle.
    Source self-intersection is not certified. Duck-typed mesh.section fixtures
    retain the discrete-path fallback; its dangling/open entities are checked
    because path.discrete contains only closed cycles. XY is projected directly
    without a recentered path transform. Empty intersections are valid evidence.
    """
    tolerance = _finite_scalar(tolerance, "tolerance", nonnegative=True)
    z = _finite_scalar(z, "z")
    if isinstance(mesh, trimesh.Trimesh):
        return _source_mesh_section_at(mesh, z, tolerance)
    raw_xy = []
    try:
        path = mesh.section(plane_origin=[0.0, 0.0, z], plane_normal=[0.0, 0.0, 1.0])
        max_plane_deviation = 0.0
        plane_scale = max(1.0, abs(z))
        mesh_vertices = np.asarray(getattr(mesh, "vertices", []), dtype=float)
        if (mesh_vertices.ndim == 2 and mesh_vertices.shape[1] == 3 and len(mesh_vertices)
                and np.isfinite(mesh_vertices[:, 2]).all()):
            plane_scale = max(plane_scale, float(np.max(np.abs(mesh_vertices[:, 2]))))
        numerical_epsilon = 128 * np.finfo(float).eps * plane_scale
        if path is not None:
            raw_xyz = list(path.discrete)
            for i, raw in enumerate(raw_xyz):
                xyz = np.asarray(raw, dtype=float)
                if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
                    raise SectionTopologyError(
                        "invalid_section_coordinates", "mesh.section must provide nonempty XYZ paths.",
                        z_mm=z, raw_paths=raw_xyz, details={"path_id": i})
                raw_xy.append(xyz[:, :2])
                if not np.isfinite(xyz).all():
                    raise SectionTopologyError(
                        "nonfinite_coordinates", "Section XYZ coordinates must be finite.",
                        z_mm=z, raw_paths=raw_xyz, details={"path_id": i})
                deviation = float(np.max(np.abs(xyz[:, 2] - z)))
                numerical_epsilon = 128 * np.finfo(float).eps * max(
                    plane_scale, float(np.max(np.abs(xyz[:, 2]))))
                if deviation > numerical_epsilon:
                    raise SectionTopologyError(
                        "nonplanar_section_path", "Discrete path is not in the requested local Z plane.",
                        z_mm=z, raw_paths=raw_xyz,
                        details={"path_id": i, "max_plane_deviation_mm": deviation})
                max_plane_deviation = max(max_plane_deviation, deviation)

            dangling = [int(i) for i in getattr(path, "dangling", [])]
            if dangling or not bool(getattr(path, "is_closed", True)):
                open_entities = [
                    {"entity_id": i,
                     "points_xyz_mm": path.entities[i].discrete(path.vertices).tolist()}
                    for i in dangling
                ]
                raise SectionTopologyError(
                    "open_or_branched_section", "Section contains dangling/open or branched paths; "
                    "closed cycles alone are not a complete section.", z_mm=z, raw_paths=raw_xy,
                    details={"dangling_entity_ids": dangling, "unclosed_entities": open_entities})
        evidence = polygons_to_section(raw_xy, z=z, tolerance=tolerance)
        evidence.metadata.update({
            "source": "mesh.section.discrete",
            "extraction_method": "mesh.section.discrete_fallback",
            "backend": "mesh.section",
            "numeric_on_plane_epsilon": float(numerical_epsilon),
            "projection": "local_xy_without_transform",
            "max_plane_deviation_mm": max_plane_deviation,
        })
        return evidence
    except SectionTopologyError:
        raise
    except Exception as exc:
        raise SectionTopologyError(
            "section_extraction_failed", f"mesh.section/path extraction failed: {exc}",
            z_mm=z, raw_paths=raw_xy, details={"exception_type": type(exc).__name__}) from exc


def _sampling_schedule(vertices, max_samples=MAX_SAMPLES):
    zs = np.sort(vertices[:, 2])
    z_min, z_max = float(zs[0]), float(zs[-1])
    height = z_max - z_min
    if not math.isfinite(height) or height <= 0:
        raise ValueError("Mesh must have finite, positive local Z extent")
    # Numerical roundoff only, independent of the user's profile tolerance:
    # a 0.01 mm step must not be merged because tolerance happens to be 0.05 mm.
    # Rotation/inverse-rotation roundoff also scales with the XY extent, not
    # only the thin Z extent. Translation is included conservatively as well.
    event_epsilon = max(1e-9, 128 * np.finfo(float).eps * max(
        1.0, float(np.max(np.abs(vertices))), height))
    events = []
    start = 0
    for i in range(1, len(zs) + 1):
        if i == len(zs) or zs[i] - zs[start] > event_epsilon:
            lo, hi = float(zs[start]), float(zs[i - 1])
            events.append({"z_mm": lo + (hi - lo) / 2, "z_min_mm": lo,
                           "z_max_mm": hi, "vertex_count": i - start})
            start = i

    candidates = [
        {"z_mm": z_min + height * fraction, "fraction": fraction,
         "default_fraction": fraction, "sources": ["default_fraction"],
         "event_interval_indices": []}
        for fraction in DEFAULT_FRACTIONS
    ]
    for i in range(len(events) - 1):
        # Sample between cluster ranges, rather than through either event cluster.
        lo, hi = events[i]["z_max_mm"], events[i + 1]["z_min_mm"]
        z = lo + (hi - lo) / 2
        # Only mandatory samples can share an interval midpoint; midpoints
        # of different intervals are distinct. Avoid quadratic work for a mesh
        # with many genuine event layers. A near-equal mandatory Z at an event
        # endpoint must NOT replace the interior sample of a narrow step.
        match = min(candidates[:len(DEFAULT_FRACTIONS)], key=lambda c: abs(c["z_mm"] - z))
        if abs(match["z_mm"] - z) <= event_epsilon and lo < match["z_mm"] < hi:
            if "vertex_event_midpoint" not in match["sources"]:
                match["sources"].append("vertex_event_midpoint")
            match["event_interval_indices"].append(i)
        else:
            candidates.append({"z_mm": z, "fraction": (z - z_min) / height,
                               "default_fraction": None, "sources": ["vertex_event_midpoint"],
                               "event_interval_indices": [i]})

    additional = candidates[len(DEFAULT_FRACTIONS):]
    budget = max_samples - len(DEFAULT_FRACTIONS)
    if len(additional) > budget:
        selected_ids = set(np.rint(np.linspace(0, len(additional) - 1, budget)).astype(int).tolist())
    else:
        selected_ids = set(range(len(additional)))
    selected = candidates[:len(DEFAULT_FRACTIONS)] + [
        c for i, c in enumerate(additional) if i in selected_ids]
    omitted = [c for i, c in enumerate(additional) if i not in selected_ids]
    selected.sort(key=lambda c: c["z_mm"])
    policy = {
        "name": "mandatory_fractions_plus_vertex_z_event_midpoints",
        "default_fractions": list(DEFAULT_FRACTIONS),
        "reference_fraction": REFERENCE_FRACTION,
        "max_samples": max_samples,
        "event_merge_tolerance_mm": float(event_epsilon),
        "raw_unique_vertex_z_count": int(len(np.unique(zs))),
        "event_layer_count": len(events),
        "event_layers": events,
        "event_interval_count": max(0, len(events) - 1),
        "attempted_event_interval_count": sum(len(c["event_interval_indices"]) for c in selected),
        "candidate_sample_count": len(candidates),
        "attempted_sample_count": len(selected),
        "coverage_limited": bool(omitted),
        "omitted_sample_count": len(omitted),
        "omitted_event_midpoints_mm": [c["z_mm"] for c in omitted],
        "omitted_event_interval_indices": [i for c in omitted for i in c["event_interval_indices"]],
        "cap_selection": "retain_all_mandatory_fractions_then_evenly_spread_event_midpoints",
        "continuous_coverage_proven": False,
    }
    return selected, policy, [z_min, z_max]


def _probe_section_events(mesh, vertices, failures, tolerance, max_samples):
    """Supplement exact event failures, never repair or replace primary evidence.

    Neighbors are *raw* referenced vertex Z values, not the schedule's coalesced
    layers. Even a numerical near-event cannot be crossed to obtain a nicer
    slice. Thus tightly spaced/roundoff-split events can be unprobeable. Offsets
    are finite observations, not certified one-sided limits of the exact event.
    The extra extraction budget is local to this call/direction; round-robin
    allocation gives both sides a first chance before any exponential retry.
    """
    budget = EVENT_PROBE_BUDGET_PER_PRIMARY_ATTEMPT * max_samples
    policy = {
        "scope": "per_analyze_sections_call_in_one_local_z_direction",
        "trigger_error_code": "ambiguous_section_event",
        "additional_section_attempt_budget": budget,
        "max_attempts_per_side": EVENT_PROBE_MAX_ATTEMPTS_PER_SIDE,
        "maximum_total_section_attempts": max_samples + budget,
        "budget_allocation": "all_sides_first_then_round_robin_retries",
        "retry_growth_factor": 2,
        "retryable_error_codes": ["ambiguous_section_event"],
        "side_status_meanings": {
            "success": "valid_finite_offset_section_including_empty_intersection",
            "failed": "attempted_but_no_valid_probe_all_failures_retained",
            "unavailable": "no_probe_extraction_attempted_see_stop_reason",
        },
        "delta_rule": "actual_z_mm_minus_original_z_mm_signed",
        "neighbor_event_rule": "raw_referenced_vertex_z_without_coalescing",
        "max_offset_fraction_of_neighbor_gap": .25,
        "max_offset_fraction_of_local_z_extent": .001,
        "initial_offset_fraction_of_local_z_extent": 1e-6,
        "initial_offset_numeric_epsilon_factor": 4,
        "candidate_rounding_rule": "float64_then_nextafter_toward_original_if_over_offset_bound",
        "evidence_scope": "finite_offset_sections_not_exact_event_or_one_sided_limit_certification",
        "used_for_primary_samples_reference_or_profile": False,
        "exact_failures_preserved": True,
        "mesh_repair_performed": False,
    }
    if not any(failure["stage"] == "section" and failure["error_code"] == "ambiguous_section_event"
               for failure in failures):
        policy.update(event_failure_count=0, side_count=0, attempted_section_count=0,
                      remaining_section_attempt_budget=budget, retry_section_count=0,
                      successful_side_count=0, failed_side_count=0, unavailable_side_count=0,
                      budget_limited=False)
        return [], policy
    events, work = [], []
    used_budget, budget_limited = 0, False
    zs = np.unique(vertices[:, 2])
    height = float(zs[-1] - zs[0])
    # Every candidate is inside the referenced mesh bounds, so this is also
    # the source extractor's epsilon at every actual probe Z (including XY
    # translation/rotation scale). Do not use manufacturing tolerance here.
    epsilon = float(128 * np.finfo(float).eps * max(1., float(np.max(np.abs(vertices)))))
    for failure_index, failure in enumerate(failures):
        if failure["stage"] != "section" or failure["error_code"] != "ambiguous_section_event":
            continue
        z = failure["z_mm"]
        source = failure.get("evidence") or {}
        scopes = {key: source[key] for key in ("source_triangle_id_scope", "source_vertex_id_scope")
                  if key in source}
        identity = {"parent_attempt_index": failure["attempt_index"],
                    "parent_failed_sample_index": failure_index, "original_z_mm": z}
        event = {
            **identity, **scopes,
            "parent_error_code": failure["error_code"],
            "parent_evidence_json_pointer": f"/failed_samples/{failure_index}/evidence",
            "coordinate_frame": "input_local_xyz_mm",
            "sides": [],
        }
        events.append(event)
        supported = (isinstance(mesh, trimesh.Trimesh)
                     and source.get("extraction_method") == "source_edge_id_plane_intersection"
                     and bool(source.get("on_plane_vertex_ids")))
        for side, direction in (("below", -1), ("above", 1)):
            neighbor_index = (int(np.searchsorted(zs, z, side="left")) - 1 if direction < 0
                              else int(np.searchsorted(zs, z, side="right")))
            neighbor = float(zs[neighbor_index]) if 0 <= neighbor_index < len(zs) else None
            result = {
                **identity, **scopes, "side": side,
                "status": "unavailable", "stop_reason": None,
                "adjacent_event_z_mm": neighbor,
                "open_interval_z_mm": (sorted([z, neighbor]) if neighbor is not None else None),
                "numeric_on_plane_epsilon_mm": epsilon,
                "max_abs_delta_mm": 0., "initial_abs_delta_mm": None,
                "actual_z_mm": None, "delta_mm": None,
                "selected_probe_attempt_index": None, "attempts": [],
            }
            event["sides"].append(result)
            if not supported:
                result["stop_reason"] = "source_event_provenance_unavailable"
            elif neighbor is None or not zs[0] <= z <= zs[-1]:
                result["stop_reason"] = "outside_mesh_z_bounds"
            else:
                limit = min(abs(neighbor - z) * .25, height * .001)
                result["max_abs_delta_mm"] = limit
                if limit <= epsilon:
                    result["stop_reason"] = "no_numeric_clearance_before_adjacent_event"
                    continue
                initial = min(max(4 * epsilon, height * 1e-6), limit)
                result["initial_abs_delta_mm"] = initial
                work.append({"result": result, "direction": direction, "active": True,
                             "previous_requested": 0.})

    for retry_index in range(EVENT_PROBE_MAX_ATTEMPTS_PER_SIDE):
        for item in work:
            if not item["active"]:
                continue
            result = item["result"]
            if used_budget >= budget:
                result["stop_reason"] = "direction_probe_budget_exhausted"
                item["active"], budget_limited = False, True
                continue
            z, direction = result["original_z_mm"], item["direction"]
            requested = min(result["initial_abs_delta_mm"] * 2 ** retry_index,
                            result["max_abs_delta_mm"])
            if requested <= item["previous_requested"]:
                result["stop_reason"], item["active"] = "offset_bound_reached", False
                continue
            actual = z + direction * requested
            delta = actual - z
            if abs(delta) > result["max_abs_delta_mm"]:
                actual = math.nextafter(actual, z)
                delta = actual - z
            interval = result["open_interval_z_mm"]
            if (not math.isfinite(actual) or direction * delta <= 0
                    or abs(delta) > result["max_abs_delta_mm"]
                    or not interval[0] < actual < interval[1]
                    or actual == result["actual_z_mm"]):
                result["stop_reason"], item["active"] = "no_distinct_representable_bounded_offset", False
                continue
            index = int(np.searchsorted(zs, actual))
            clearance = min(abs(float(zs[i]) - actual) for i in (index - 1, index)
                            if 0 <= i < len(zs))
            if clearance <= epsilon:
                result["stop_reason"] = "no_numeric_clearance_before_adjacent_event"
                item["active"] = False
                continue
            item["previous_requested"] = requested
            sampling = {key: result[key] for key in (
                "parent_attempt_index", "parent_failed_sample_index", "original_z_mm", "side")}
            sampling.update(probe_attempt_index=used_budget, side_attempt_index=len(result["attempts"]),
                            actual_z_mm=actual, delta_mm=delta,
                            kind="one_sided_event_probe", included_in_primary_sampling=False)
            attempt = {
                **sampling,
                **{key: result[key] for key in ("source_triangle_id_scope", "source_vertex_id_scope")
                   if key in result},
                "requested_abs_delta_mm": requested,
                "nearest_vertex_z_clearance_mm": clearance,
            }
            used_budget += 1
            result["attempts"].append(attempt)
            result["actual_z_mm"], result["delta_mm"] = actual, delta
            try:
                evidence = section_at(mesh, actual, tolerance=tolerance)
            except Exception as exc:
                code = getattr(exc, "code", "section_extraction_failed")
                attempt.update(status="failed", error_code=code, error_type=type(exc).__name__,
                               reason=str(exc), evidence=getattr(exc, "metadata", None))
                result["status"] = "failed"
                if code != "ambiguous_section_event":
                    result["stop_reason"], item["active"] = "non_event_probe_failure_not_retried", False
            else:
                evidence.metadata["sampling"] = sampling
                evidence.metadata["comparison_to_reference"] = None
                attempt.update(status="success", evidence=evidence.metadata)
                result.update(status="success", stop_reason="probe_succeeded",
                              selected_probe_attempt_index=sampling["probe_attempt_index"])
                item["active"] = False
    for item in work:
        if item["active"]:
            item["result"]["stop_reason"] = "per_side_attempt_limit_reached"
    sides = [side for event in events for side in event["sides"]]
    policy.update(event_failure_count=len(events), side_count=len(sides),
                  attempted_section_count=used_budget, remaining_section_attempt_budget=budget - used_budget,
                  retry_section_count=sum(max(0, len(side["attempts"]) - 1) for side in sides),
                  successful_side_count=sum(side["status"] == "success" for side in sides),
                  failed_side_count=sum(side["status"] == "failed" for side in sides),
                  unavailable_side_count=sum(side["status"] == "unavailable" for side in sides),
                  budget_limited=budget_limited)
    return events, policy


_TOPOLOGY_FIELDS = (
    "loop_count", "outer_region_count", "material_region_count", "void_loop_count",
    "max_nesting_depth", "topology_signature",
)


def _boundary_data(geometry, tolerance):
    """Segment index and samples with a continuous-distance error bound.

    Distance to a closed boundary is 1-Lipschitz. Sampling every polyline segment
    with maximum gap g bounds its continuous maximum by sampled maximum + g/2.
    A spatial index avoids GEOS's quadratic densified Hausdorff implementation.
    """
    boundary = geometry.boundary
    rings = list(boundary.geoms) if hasattr(boundary, 'geoms') else [boundary]
    coordinates = [np.asarray(ring.coords, dtype=float) for ring in rings]
    segments = np.concatenate([np.stack((c[:-1], c[1:]), axis=1) for c in coordinates])
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    keep = lengths > 0
    segments, lengths = segments[keep], lengths[keep]
    if not len(segments):
        raise ValueError('Nonempty boundary has no nonzero segments')
    spacing = max(tolerance * .2, float(lengths.max()) * 1e-6, 1e-10)
    budget = max(20_000, len(segments))
    while True:
        counts = np.maximum(1, np.ceil(lengths / spacing)).astype(int)
        if counts.sum() <= budget:
            break
        spacing *= 2
    ids = np.repeat(np.arange(len(segments)), counts)
    starts = np.repeat(np.r_[0, np.cumsum(counts)[:-1]], counts)
    fractions = (np.arange(len(ids)) - starts) / counts[ids]
    samples = segments[ids, 0] + fractions[:, None] * (segments[ids, 1] - segments[ids, 0])
    return STRtree(linestrings(segments)), points(samples), float(np.max(lengths / counts)) / 2


def _bounded_boundary_hausdorff(a, b, tolerance, reference_boundary=None):
    if a.boundary.equals(b.boundary):
        return 0.0, 0.0, 0.0
    tree_a, points_a, error_a = reference_boundary or _boundary_data(a, tolerance)
    tree_b, points_b, error_b = _boundary_data(b, tolerance)
    _, distances_a = tree_b.query_nearest(points_a, return_distance=True, all_matches=False)
    _, distances_b = tree_a.query_nearest(points_b, return_distance=True, all_matches=False)
    lower = max(float(distances_a.max()), float(distances_b.max()))
    upper = max(float(distances_a.max()) + error_a, float(distances_b.max()) + error_b)
    return lower, upper, upper - lower


def _compare_sections(reference, sample, tolerance, reference_index, reference_boundary=None):
    changed = [key for key in _TOPOLOGY_FIELDS if reference.metadata[key] != sample.metadata[key]]
    a, b = reference.geometry, sample.geometry
    difference_area = float(a.symmetric_difference(b).area)
    # Dimensional area allowance: a boundary strip of the user-specified width.
    # This is a comparison heuristic, not a geometric equivalence certificate.
    area_threshold = tolerance * float(a.length)
    if a.is_empty and b.is_empty:
        distance, upper_bound, error_bound, distance_status = 0.0, 0.0, 0.0, "both_empty"
    elif a.is_empty or b.is_empty:
        distance, upper_bound, error_bound, distance_status = None, None, None, "undefined_one_empty"
    else:
        distance, upper_bound, error_bound = _bounded_boundary_hausdorff(
            a, b, tolerance, reference_boundary=reference_boundary)
        distance_status = "bounded_continuous_polyline_distance"
    if (not math.isfinite(difference_area) or not math.isfinite(area_threshold)
            or (distance is not None and not math.isfinite(distance))):
        raise ValueError("Nonfinite section comparison metric or area threshold")
    boundary_exceeds = distance is not None and distance > tolerance
    area_exceeds = difference_area > area_threshold
    return {
        "reference_sample_index": reference_index,
        "topology_changed": bool(changed),
        "changed_topology_fields": changed,
        "boundary_hausdorff_mm": distance,
        "boundary_hausdorff_status": distance_status,
        "boundary_hausdorff_upper_bound_mm": upper_bound,
        "boundary_hausdorff_error_bound_mm": error_bound,
        "boundary_hausdorff_value_is_lower_bound": True,
        "profile_comparison_inconclusive": bool(distance is not None and distance <= tolerance < upper_bound),
        "symmetric_difference_area_mm2": difference_area,
        "boundary_hausdorff_threshold_mm": tolerance,
        "symmetric_difference_area_threshold_mm2": area_threshold,
        "boundary_exceeds_tolerance": boundary_exceeds,
        "area_exceeds_threshold": area_exceeds,
        "profile_differs": bool(changed) or boundary_exceeds or area_exceeds,
    }


def analyze_sections(mesh, tolerance=0.05, *, max_samples=MAX_SAMPLES) -> SectionStack:
    """Compare up to 40 actual slices to the mandatory Z-extent fraction 0.25.

    Always attempt the seven default fractions, plus numerically distinct
    vertex-Z interval midpoints. If capped, retain the defaults and evenly
    spread the remaining midpoint budget; report omitted intervals explicitly.
    Failed slicing/comparison attempts are retained as evidence and never imply
    uniformity. Event failures alone receive separate finite-offset probes with
    at most 2 * max_samples additional extractions per call/direction and four
    attempts per side. These never enter samples, coverage, or comparisons and
    cannot remove an exact failure or CAD veto. 2D boundary metrics are not a
    bidirectional 3D surface test.
    """
    tolerance = _finite_scalar(tolerance, "tolerance", nonnegative=True)
    if type(max_samples) is not int or not len(DEFAULT_FRACTIONS) <= max_samples <= MAX_SAMPLES:
        raise ValueError(f'max_samples must be an integer in [{len(DEFAULT_FRACTIONS)}, {MAX_SAMPLES}]')
    try:
        vertices = np.asarray(mesh.vertices, dtype=float)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Mesh must provide finite XYZ vertices") from exc
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0
            or not np.isfinite(vertices).all()):
        raise ValueError("Mesh must provide a nonempty array of finite XYZ vertices")
    if isinstance(mesh, trimesh.Trimesh):
        faces = np.asarray(mesh.faces)
        if (faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces)
                or not np.issubdtype(faces.dtype, np.integer)):
            raise ValueError('Mesh must provide nonempty integer triangular faces')
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError('Source triangle index is outside the vertex array')
        vertices = vertices[np.unique(faces)]
    schedule, policy, bounds = _sampling_schedule(vertices, max_samples=max_samples)
    samples = []
    failures = []
    reference_index = None

    def record_failure(attempt_index, attempt, exc, stage):
        failures.append({
            "attempt_index": attempt_index,
            "sample_index": attempt.get("sample_index"),
            "z_mm": attempt["z_mm"],
            "fraction": attempt["fraction"],
            "default_fraction": attempt["default_fraction"],
            "sources": list(attempt["sources"]),
            "stage": stage,
            "error_type": type(exc).__name__,
            "error_code": getattr(exc, "code", "section_comparison_failed" if stage == "comparison"
                                  else "section_extraction_failed"),
            "reason": str(exc),
            "evidence": getattr(exc, "metadata", None),
        })

    for attempt_index, attempt in enumerate(schedule):
        attempt["sample_index"] = None
        try:
            evidence = section_at(mesh, attempt["z_mm"], tolerance=tolerance)
        except Exception as exc:
            attempt["status"] = "failed"
            record_failure(attempt_index, attempt, exc, "section")
            continue
        attempt["status"] = "success"
        attempt["sample_index"] = len(samples)
        evidence.metadata["sampling"] = {
            "attempt_index": attempt_index,
            "fraction": attempt["fraction"],
            "default_fraction": attempt["default_fraction"],
            "sources": list(attempt["sources"]),
            "event_interval_indices": list(attempt["event_interval_indices"]),
        }
        evidence.metadata["comparison_to_reference"] = None
        if attempt["default_fraction"] == REFERENCE_FRACTION:
            reference_index = len(samples)
        samples.append(evidence)

    event_probes, event_probe_policy = _probe_section_events(
        mesh, vertices, failures, tolerance, max_samples)
    policy["successful_sample_count"] = len(samples)
    policy["sampled_event_interval_count"] = sum(
        len(s.metadata["sampling"]["event_interval_indices"]) for s in samples)
    comparisons = []
    unavailable_hausdorff = []
    if reference_index is not None:
        reference = samples[reference_index]
        reference_boundary = (_boundary_data(reference.geometry, tolerance)
                              if not reference.geometry.is_empty else None)
        for sample_index, sample in enumerate(samples):
            try:
                comparison = _compare_sections(reference, sample, tolerance, reference_index,
                                               reference_boundary=reference_boundary)
            except Exception as exc:
                attempt_index = sample.metadata["sampling"]["attempt_index"]
                schedule[attempt_index]["status"] = "comparison_failed"
                record_failure(attempt_index, schedule[attempt_index], exc, "comparison")
                continue
            sample.metadata["comparison_to_reference"] = comparison
            comparisons.append(comparison)
            if comparison["boundary_hausdorff_mm"] is None:
                unavailable_hausdorff.append(sample_index)

    observed_variation = any(c["profile_differs"] for c in comparisons)
    inconclusive = any(c['profile_comparison_inconclusive'] for c in comparisons)
    profile_varies = True if observed_variation else None if failures or reference_index is None or inconclusive else False
    warnings = [
        "Discrete section sampling is not proof of a uniform profile; sampling_complete is always false.",
        "Boundary Hausdorff distances concern sampled 2D section boundaries only, "
        "not a bidirectional 3D surface certificate.",
    ]
    if policy["coverage_limited"]:
        warnings.append(f"Sampling capped at {max_samples} layers: {policy['omitted_sample_count']} "
                        "event-midpoint candidates were not sampled; coverage is limited.")
    if policy["raw_unique_vertex_z_count"] > policy["event_layer_count"]:
        warnings.append(f"Numerically coincident vertex Z events were coalesced at "
                        f"{policy['event_merge_tolerance_mm']:g} mm; sub-resolution features may be missed.")
    if failures:
        warnings.append(f"{len(failures)} failed sample/comparison attempt(s); automatic full-thickness "
                        "or uniform-extrusion suggestions are prohibited. See failed_samples.")
    if event_probes:
        warnings.append("Finite-offset one-sided event probes are supplemental only, not certified event "
                        "limits; exact failures, primary comparisons and CAD blockers remain unchanged.")
    if reference_index is None:
        warnings.append("The mandatory 0.25 reference section failed; reference comparisons are unavailable.")
    if unavailable_hausdorff:
        warnings.append("Some boundary distances are undefined because exactly one section is empty; "
                        "topology and symmetric-difference comparisons still apply.")
    if inconclusive:
        warnings.append('Some continuous boundary-distance bounds straddle the tolerance; '
                        'sample agreement is indeterminate, not certified.')
    if observed_variation:
        warnings.append("Sampled profiles vary; an unmodified full-thickness extrusion is not supported.")
    distances = [c["boundary_hausdorff_mm"] for c in comparisons if c["boundary_hausdorff_mm"] is not None]
    areas = [c["symmetric_difference_area_mm2"] for c in comparisons]
    area_thresholds = [c["symmetric_difference_area_threshold_mm2"] for c in comparisons]
    metadata = {
        "z_bounds_mm": bounds,
        "tolerance_mm": tolerance,
        "coordinate_frame": "input_local_xyz_mm",
        "sample_count": len(samples),
        "reference_fraction": REFERENCE_FRACTION,
        "reference_z_mm": bounds[0] + (bounds[1] - bounds[0]) * REFERENCE_FRACTION,
        "reference_sample_index": reference_index,
        "profile_varies": profile_varies,
        "profile_status": ("varies_at_samples" if profile_varies is True else
                           "consistent_at_samples" if profile_varies is False else "indeterminate"),
        "topology_changed": any(c["topology_changed"] for c in comparisons),
        "max_boundary_hausdorff_mm": max(distances, default=None),
        "max_boundary_hausdorff_upper_bound_mm": max(
            (c['boundary_hausdorff_upper_bound_mm'] for c in comparisons
             if c['boundary_hausdorff_upper_bound_mm'] is not None), default=None),
        "boundary_comparison_inconclusive": inconclusive,
        "max_symmetric_difference_area_mm2": max(areas, default=None),
        "boundary_hausdorff_comparison_complete": bool(comparisons) and not failures
                                                  and not unavailable_hausdorff,
        "hausdorff_unavailable_sample_indices": unavailable_hausdorff,
        "sampling_complete": False,
        "automatic_full_thickness_suggestion_allowed": False,
        "warnings": warnings,
        "sampling_policy": policy,
        "sampling_schedule": schedule,
        "failed_samples": failures,
        "event_probes": event_probes,
        "event_probe_policy": event_probe_policy,
        "comparison_policy": {
            "reference_fraction": REFERENCE_FRACTION,
            "boundary_hausdorff_threshold_mm": tolerance,
            "symmetric_difference_area_threshold_mm2": area_thresholds[0] if area_thresholds else None,
            "symmetric_difference_area_rule": "tolerance_mm * reference_material_boundary_length_mm",
            "topology_changes_always_vary": True,
            "boundary_hausdorff_method": "STRtree_segment_distance_with_1_Lipschitz_continuous_bounds",
            "reported_distance": "lower_bound_with_explicit_upper_bound",
            "nominal_sample_spacing_mm": tolerance * .2,
            "nominal_sample_budget_per_boundary": 20_000,
            "metric_scope": "2D_section_boundaries_not_a_3D_surface_certificate",
        },
    }
    return SectionStack(samples, metadata)
