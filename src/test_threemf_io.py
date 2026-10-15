"""Synthetic boxes, not a fixture tied to a particular exported part."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import warnings
import zipfile

import numpy as np
import trimesh

import threemf_io as io

IDENTITY = "1 0 0 0 1 0 0 0 1 0 0 0"


def matrix_text(matrix):
    matrix = np.asarray(matrix)
    return " ".join(str(value) for value in np.r_[matrix[:3, :3].T.ravel(), matrix[:3, 3]])


def translation(x, y, z):
    matrix = np.eye(4)
    matrix[:3, 3] = [x, y, z]
    return matrix


def mesh_xml(vertices=None, faces=None):
    box = trimesh.creation.box(extents=[2, 4, 6])
    vertices = box.vertices if vertices is None else vertices
    faces = box.faces if faces is None else faces
    v = "".join(f'<vertex x="{x}" y="{y}" z="{z}"/>' for x, y, z in vertices)
    f = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in faces)
    return f"<mesh><vertices>{v}</vertices><triangles>{f}</triangles></mesh>"


def box_object(object_id=1, name="box", **mesh_kwargs):
    return f'<object id="{object_id}" name="{name}">{mesh_xml(**mesh_kwargs)}</object>'


def assembly(object_id, children, name="assembly"):
    return f'<object id="{object_id}" name="{name}"><components>{children}</components></object>'


def model_xml(objects=None, items='<item objectid="1"/>', unit="millimeter", attributes="", namespace=io.CORE_NS):
    objects = box_object() if objects is None else objects
    ns = f'xmlns="{namespace}"' if namespace else ""
    build = "" if items is None else f"<build>{items}</build>"
    return (f'<model {ns} xmlns:p="{io.PRODUCTION_NS}" unit="{unit}" {attributes}>'
            f"<resources>{objects}</resources>{build}</model>")


def relationships(target, attributes="", extra=""):
    return (f'<Relationships xmlns="{io.REL_NS}">'
            f'<Relationship Id="main" Type="{io.MODEL_REL}" Target="{target}" {attributes}/>'
            f"{extra}</Relationships>")


class ThreeMFTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.counter = 0

    def package(self, parts):
        self.counter += 1
        path = self.directory / f"case-{self.counter}.3mf"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # duplicate ZIP tests
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in (parts.items() if isinstance(parts, dict) else parts):
                    archive.writestr(name, data)
        return path

    def document(self, xml=None, **kwargs):
        return io.load_document(self.package({"3D/3dmodel.model": xml or model_xml()}), **kwargs)

    def assert_invalid(self, xml, message):
        with self.assertRaisesRegex(io.ThreeMFError, message):
            self.document(xml)

    def test_transform_layout_and_validation(self):
        matrix = io.transform("0 1 0 -1 0 0 0 0 1 10 20 30")
        np.testing.assert_array_equal(matrix @ [1, 0, 0, 1], [10, 21, 30, 1])
        for text in (None, "", "  "):
            np.testing.assert_array_equal(io.transform(text), np.eye(4))
        bad = ["1 2 3", IDENTITY.replace("1", "nan", 1), IDENTITY.replace("1", "inf", 1),
               "0 0 0 0 1 0 0 0 1 0 0 0", "1 0 0 1 0 0 0 0 1 0 0 0", "wrong " * 12, 7]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(io.ThreeMFError):
                io.transform(text)
        tiny_scale = np.diag([1e-20, 1e-20, 1e-20, 1])
        np.testing.assert_array_equal(io.transform(matrix_text(tiny_scale)), tiny_scale)

    def test_group_components_by_real_build_item_and_keep_ranges(self):
        children = '<component objectid="1"/><component objectid="1" transform="1 0 0 0 1 0 0 0 1 10 0 0"/>'
        xml = model_xml(box_object() + assembly(2, children, "two boxes"), '<item objectid="2"/>')
        document = self.document(xml)
        self.assertEqual(len(document.instances), 1)
        instance = document.instances[0]
        self.assertEqual(len(instance.mesh.faces), 24)
        self.assertEqual(len(instance.mesh.split(only_watertight=False)), 2)
        self.assertEqual(instance.metadata["build_index"], 0)
        self.assertEqual(instance.metadata["object_id"], 2)
        self.assertEqual(instance.metadata["model_path"], "3D/3dmodel.model")
        self.assertEqual(instance.metadata["name"], "two boxes")
        self.assertEqual([c["face_start"] for c in instance.metadata["components"]], [0, 12])
        self.assertEqual([c["face_end"] for c in instance.metadata["components"]], [12, 24])
        for component in instance.metadata["components"]:
            self.assertEqual(component["source_face_start"], 0)
            self.assertEqual(component["source_face_end"], 12)
        json.dumps(document.metadata, allow_nan=False)
        json.dumps(instance.metadata, allow_nan=False)
        self.assertEqual(instance.mesh.metadata, instance.metadata)
        self.assertEqual(document.metadata["statistics"]["meshes_parsed"], 1)
        self.assertEqual(document.metadata["statistics"]["objects_parsed"], 2)

    def test_root_relationship_wins_over_conventional_name(self):
        package = self.package({
            "_rels/.rels": relationships("/Chosen/main.model"),
            "Chosen/main.model": model_xml(box_object(name="chosen")),
            "3D/3dmodel.model": model_xml(box_object(name="wrong")),
        })
        document = io.load_document(package)
        self.assertEqual(document.metadata["entry_model"], "Chosen/main.model")
        self.assertEqual(document.metadata["entry_model_selection"], "opc_root_relationship")
        self.assertEqual(document.instances[0].metadata["name"], "chosen")
        self.assertEqual(document.metadata["statistics"]["models_parsed"], 1)

    def test_relationship_validation_never_silently_falls_back(self):
        extra = f'<Relationship Id="second" Type="{io.MODEL_REL}" Target="3D/3dmodel.model"/>'
        cases = [relationships("http://invalid.example/part.model"),
                 relationships("/3D/3dmodel.model", 'TargetMode="External"'),
                 relationships("../outside.model"), relationships("missing.model"),
                 relationships("3D/3dmodel.model", extra=extra),
                 f'<Relationships xmlns="{io.REL_NS}"/>']
        for rel in cases:
            with self.subTest(rel=rel), self.assertRaises(io.ThreeMFError):
                io.load_document(self.package({"_rels/.rels": rel, "3D/3dmodel.model": model_xml()}))

    def test_nested_cross_model_units_and_noncommuting_transform_order(self):
        rotate = np.array([[0., -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        build = translation(2, 3, 4) @ rotate
        parent = translation(1, 2, 3)
        parent[:3, :3] = np.diag([2., 1, 0.5])
        child = translation(0.5, 1, -2) @ rotate
        package = self.package({
            "_rels/.rels": relationships("Assembly/root.model"),
            "Assembly/root.model": model_xml(assembly(10,
                f'<component objectid="20" p:path="nested/middle.model" transform="{matrix_text(parent)}"/>'),
                f'<item objectid="10" transform="{matrix_text(build)}"/>', unit="centimeter"),
            "Assembly/nested/middle.model": model_xml(assembly(20,
                f'<component objectid="30" p:path="/Parts/leaf.model" transform="{matrix_text(child)}"/>'),
                None, unit="inch"),
            "Parts/leaf.model": model_xml(box_object(30), None, unit="micron"),
        })
        document = io.load_document(package)
        instance = document.instances[0]
        expected_build = build.copy()
        expected_build[:3, 3] *= 10
        expected_parent = parent.copy()
        expected_parent[:3, 3] *= 10
        expected_child = child.copy()
        expected_child[:3, 3] *= 25.4
        expected = expected_build @ expected_parent @ expected_child @ np.diag([0.001] * 3 + [1])
        component = instance.metadata["components"][0]
        np.testing.assert_allclose(component["source_to_world_mm"], expected)
        box = trimesh.creation.box(extents=[2, 4, 6])
        np.testing.assert_allclose(instance.mesh.vertices, box.vertices @ expected[:3, :3].T + expected[:3, 3])
        self.assertEqual(component["original_unit"], "micron")
        self.assertEqual([step["transform_unit"] for step in component["component_path"]], ["centimeter", "inch"])
        np.testing.assert_array_equal(instance.metadata["build_transform"], build)
        np.testing.assert_array_equal(instance.metadata["build_transform_mm"], expected_build)
        self.assertEqual(instance.metadata["build_transform_unit"], "centimeter")
        self.assertEqual(document.metadata["statistics"]["models_parsed"], 3)

    def test_build_production_path_and_relative_parent_segments(self):
        package = self.package({
            "_rels/.rels": relationships("/Entry/root.model"),
            "Entry/root.model": model_xml("",
                '<item objectid="7" p:path="../Parts/box%20file.model" transform="1 0 0 0 1 0 0 0 1 1 0 0"/>', unit="inch"),
            "Parts/box file.model": model_xml(box_object(7), None, unit="centimeter"),
        })
        instance = io.load_document(package).instances[0]
        self.assertEqual(instance.metadata["model_path"], "Parts/box file.model")
        np.testing.assert_allclose(instance.mesh.extents, [20, 40, 60])
        np.testing.assert_allclose(instance.mesh.bounds.mean(axis=0), [25.4, 0, 0])
        np.testing.assert_allclose(instance.metadata["components"][0]["source_to_world_mm"],
                                   translation(25.4, 0, 0) @ np.diag([10, 10, 10, 1]))

    def test_multiple_instances_are_independent_and_source_objects_are_cached(self):
        children = '<component objectid="1"/>'
        xml = model_xml(box_object() + assembly(2, children),
                        '<item objectid="2"/><item objectid="2" transform="1 0 0 0 1 0 0 0 1 50 0 0"/>')
        document = self.document(xml)
        self.assertEqual(len(document.instances), 2)
        first, second = document.instances
        np.testing.assert_allclose(second.mesh.bounds.mean(axis=0), [50, 0, 0])
        self.assertFalse(np.shares_memory(first.mesh.vertices, second.mesh.vertices))
        self.assertFalse(np.shares_memory(first.mesh.faces, second.mesh.faces))
        original = second.mesh.vertices.copy()
        first.mesh.vertices += 100
        first.metadata["components"][0]["source_to_world_mm"][0][0] = 99
        np.testing.assert_array_equal(second.mesh.vertices, original)
        self.assertEqual(second.metadata["components"][0]["source_to_world_mm"][0][0], 1)
        stats = document.metadata["statistics"]
        self.assertEqual(stats["xml_parts_parsed"], 1)
        self.assertEqual(stats["objects_indexed"], 2)
        self.assertEqual(stats["objects_parsed"], 2)
        self.assertEqual(stats["meshes_parsed"], 1)
        self.assertEqual(stats["component_occurrences"], 4)

    def test_printable_flags_preserve_original_build_index(self):
        items = '<item objectid="1" printable="false"/><item objectid="1" printable="0"/><item objectid="1" printable="true"/><item objectid="1" printable="1"/>'
        document = self.document(model_xml(items=items))
        self.assertEqual([i.metadata["build_index"] for i in document.instances], [2, 3])
        self.assertEqual(document.metadata["build_item_count"], 4)
        self.assertEqual(len(document.metadata["skipped_build_items"]), 2)
        self.assertTrue(document.metadata["warnings"])
        self.assert_invalid(model_xml(items='<item objectid="1" printable="no"/>'), "printable")
        self.assert_invalid(model_xml(items='<item objectid="1" printable="false"/>'), "no printable")
        self.assert_invalid(model_xml(items='<item objectid="99" printable="false"/>'), "Missing object")

    def test_mirror_preserves_orientation_without_fixing_source_winding(self):
        mirror = translation(12, 3, -2) @ np.diag([-1., 1, 1, 1])
        box = trimesh.creation.box(extents=[2, 4, 6])
        for source_faces in (box.faces, box.faces[:, ::-1]):
            with self.subTest(inward=bool(np.array_equal(source_faces, box.faces[:, ::-1]))):
                xml = model_xml(box_object(faces=source_faces),
                                f'<item objectid="1" transform="{matrix_text(mirror)}"/>')
                instance = self.document(xml).instances[0]
                expected = trimesh.Trimesh(box.vertices.copy(), source_faces.copy(), process=False)
                expected.apply_transform(mirror)
                np.testing.assert_array_equal(instance.mesh.faces, expected.faces)
                np.testing.assert_allclose(instance.mesh.vertices, expected.vertices)
                self.assertAlmostEqual(instance.mesh.volume, expected.volume)
                self.assertTrue(instance.metadata["components"][0]["winding_reversed_for_reflection"])
        inward = self.document(model_xml(box_object(faces=box.faces[:, ::-1]))).instances[0]
        np.testing.assert_array_equal(inward.mesh.faces, box.faces[:, ::-1])
        self.assertLess(inward.mesh.volume, 0)

    def test_two_nested_reflections_cancel(self):
        reflection = "-1 0 0 0 1 0 0 0 1 0 0 0"
        objects = box_object() + assembly(2, f'<component objectid="1" transform="{reflection}"/>')
        instance = self.document(model_xml(objects, f'<item objectid="2" transform="{reflection}"/>')).instances[0]
        np.testing.assert_array_equal(instance.mesh.faces, trimesh.creation.box(extents=[2, 4, 6]).faces)
        self.assertFalse(instance.metadata["components"][0]["winding_reversed_for_reflection"])

    def test_tiny_real_transform_is_not_discarded(self):
        instance = self.document(model_xml(items='<item objectid="1" transform="1 0 0 0 1 0 0 0 1 0.00000000001 0 0"/>')).instances[0]
        difference = instance.mesh.vertices - trimesh.creation.box(extents=[2, 4, 6]).vertices
        np.testing.assert_allclose(difference[:, 0], 1e-11, rtol=1e-4, atol=0)

    def test_only_exact_vertices_weld_and_no_faces_are_removed(self):
        # Triangle soup box must become connected without Trimesh.process(), and
        # degenerate, duplicate and unused geometry must not disappear.
        box = trimesh.creation.box(extents=[2, 4, 6])
        soup_vertices = box.triangles.reshape(-1, 3)
        soup_faces = np.arange(len(soup_vertices)).reshape(-1, 3)
        clean = self.document(model_xml(box_object(vertices=soup_vertices, faces=soup_faces))).instances[0]
        self.assertEqual(len(clean.mesh.split(only_watertight=False)), 1)
        self.assertTrue(clean.mesh.is_watertight)
        vertices = np.vstack([soup_vertices, [[99, 99, 99], [99 + 1e-11, 99, 99]]])
        faces = np.vstack([soup_faces, soup_faces[0], [0, 0, 0]])
        instance = self.document(model_xml(box_object(vertices=vertices, faces=faces))).instances[0]
        cleanup = instance.metadata["mesh_cleanup"][0]
        self.assertEqual(cleanup["vertices_before"], 38)
        self.assertEqual(cleanup["vertices_after"], 10)
        self.assertEqual(cleanup["vertices_merged"], 28)
        self.assertEqual(cleanup["max_vertex_displacement_mm"], 0)
        self.assertEqual(cleanup["unreferenced_vertices_removed"], 0)
        self.assertEqual(cleanup["faces_removed"], 0)
        self.assertEqual(len(instance.mesh.faces), 14)
        np.testing.assert_array_equal(instance.mesh.triangles[:12], box.triangles)
        np.testing.assert_array_equal(instance.mesh.faces[0], instance.mesh.faces[-2])
        self.assertEqual(len(set(instance.mesh.faces[-1])), 1)
        # Very close vertices are NOT rounded into each other.
        self.assertNotEqual(instance.mesh.vertices[-1, 0], instance.mesh.vertices[-2, 0])

    def test_invalid_triangle_indices_and_vertices(self):
        for index in (-1, 8, 1.5, "nan", "Infinity", 10**100):
            with self.subTest(index=index):
                self.assert_invalid(model_xml(box_object(faces=[[index, 1, 2]])), "index|integer")
        for coordinate in ("nan", "inf", "-inf", "1e999", "invalid"):
            with self.subTest(coordinate=coordinate):
                self.assert_invalid(model_xml(box_object(vertices=[[coordinate, 0, 0]], faces=[[0, 0, 0]])), "vertex")
        self.assert_invalid(model_xml().replace('x="-1.0"', '', 1), "vertex")

    def test_empty_unknown_duplicate_and_unsupported_objects(self):
        cases = [
            (model_xml("", '<item objectid="1"/>'), "Missing object"),
            (model_xml(items=""), "no printable"),
            (model_xml(items=None), "no build"),
            (model_xml('<object id="1"/>'), "exactly one"),
            (model_xml('<object id="1"><components/></object>'), "Empty components"),
            (model_xml(box_object(vertices=[], faces=[])), "Empty mesh"),
            (model_xml(box_object(faces=[])), "Empty mesh"),
            (model_xml(box_object() + box_object()), "Duplicate"),
            (model_xml(box_object() + box_object("01")), "Duplicate"),
            (model_xml(box_object().replace('id="1"', 'id="0"', 1)), "integer"),
            (model_xml(box_object().replace('id="1"', 'id="bad"', 1)), "integer"),
            (model_xml(unit="parsec"), "unit"),
            (model_xml().replace('<object id="1"', '<object type="support" id="1"'), "non-model"),
            (model_xml(box_object() + '<object id="2" type="other"><components/></object>'), "non-model"),
            (model_xml(box_object().replace('<mesh>', '<mesh><beamlattice/>')), "mesh geometry"),
            (model_xml(box_object().replace('</object>', '<unknowngeometry/></object>')), "Unsupported object"),
        ]
        for xml, message in cases:
            with self.subTest(message=message, xml=xml[:100]):
                self.assert_invalid(xml, message)

    def test_cycles_within_and_across_model_parts(self):
        self.assert_invalid(model_xml(assembly(1, '<component objectid="1"/>')), "Cyclic")
        self.assert_invalid(model_xml(assembly(1, '<component objectid="2"/>') +
                                      assembly(2, '<component objectid="1"/>')), "Cyclic")
        package = self.package({
            "3D/3dmodel.model": model_xml(assembly(1, '<component objectid="1" p:path="other.model"/>')),
            "3D/other.model": model_xml(assembly(1, '<component objectid="1" p:path="3dmodel.model"/>'), None),
        })
        with self.assertRaisesRegex(io.ThreeMFError, "Cyclic"):
            io.load_document(package)

    def test_archive_escapes_duplicates_and_aliases_are_rejected(self):
        names = ["../escape.model", "/absolute.model", "a/../../escape.model", "a/./file.xml", "a//file.xml",
                 r"..\escape.model", "C:/file.xml", "%2e%2e/escape.model", "a/%2e%2e/file.xml", "a/%252e%252e/file.xml"]
        for name in names:
            with self.subTest(name=name), self.assertRaises(io.ThreeMFError):
                io.load_document(self.package({"3D/3dmodel.model": model_xml(), name: "unused"}))
        duplicates = [
            [("3D/3dmodel.model", model_xml()), ("3D/3dmodel.model", model_xml())],
            [("3D/3dmodel.model", model_xml()), ("3D/%33dmodel.model", model_xml())],
            [("3D/3dmodel.model", model_xml()), ("a/", ""), ("a", "")],
        ]
        for parts in duplicates:
            with self.subTest(parts=[p[0] for p in parts]), self.assertRaisesRegex(io.ThreeMFError, "Duplicate"):
                io.load_document(self.package(parts))

    def test_production_paths_cannot_escape_or_fetch_external_content(self):
        paths = ["../../outside.model", "/../outside.model", "../%2e%2e/outside.model", "http://example.org/a.model",
                 "//example.org/a.model", "file:///tmp/a.model", "C:/a.model", "bad.model#frag", "bad.model?q=1",
                 r"..\a.model", "%252e%252e/a.model", "%00a.model", "missing.model"]
        for path in paths:
            for where in ("component", "build"):
                if where == "component":
                    xml = model_xml(assembly(1, f'<component objectid="1" p:path="{path}"/>'))
                else:
                    xml = model_xml(items=f'<item objectid="1" p:path="{path}"/>')
                with self.subTest(path=path, where=where), self.assertRaises(io.ThreeMFError):
                    self.document(xml)
        self.assert_invalid(model_xml(items='<item objectid="1" path="other.model"/>'), "non-production")

    def test_required_extensions_are_resolved_by_namespace_not_prefix(self):
        xml = model_xml(attributes=f'xmlns:prod="{io.PRODUCTION_NS}" xmlns:c="{io.CORE_NS}" requiredextensions="prod c"')
        self.assertEqual(len(self.document(xml).instances), 1)
        for attributes in ('requiredextensions="missing"',
                           'xmlns:mat="http://schemas.microsoft.com/3dmanufacturing/material/2015/02" requiredextensions="mat"',
                           'xmlns:p="urn:not-production" requiredextensions="p"'):
            # Avoid duplicate xmlns:p in the last case.
            xml = model_xml(attributes=attributes)
            if attributes.startswith('xmlns:p='):
                xml = model_xml(attributes='requiredextensions="p"').replace(io.PRODUCTION_NS, "urn:not-production")
            with self.subTest(attributes=attributes):
                self.assert_invalid(xml, "requiredextensions")
        optional = self.document(model_xml(attributes='xmlns:ignored="urn:optional"'))
        self.assertTrue(any("urn:optional" in warning for warning in optional.metadata["warnings"]))

    def test_xml_dtd_entities_and_malformed_xml_are_rejected(self):
        declarations = [
            '<!DOCTYPE model [<!ENTITY x "expanded">]>',
            '<!DOCTYPE model SYSTEM "file:///etc/passwd">',
            '<!DOCTYPE model [<!ENTITY % x SYSTEM "http://invalid.example/x"> %x;]>',
        ]
        for declaration in declarations:
            for encoding in ("utf-8", "utf-16"):
                xml = ('<?xml version="1.0" encoding="' + encoding + '"?>' + declaration + model_xml()).encode(encoding)
                with self.subTest(encoding=encoding, declaration=declaration), self.assertRaisesRegex(io.ThreeMFError, "DTD/entity"):
                    io.load_document(self.package({"3D/3dmodel.model": xml}))
        self.assert_invalid(model_xml().replace('name="box"', 'name="&unknown;"'), "XML")
        self.assert_invalid("<model>", "XML")
        self.assert_invalid('<?xml version="1.0" encoding="invalid"?>' + model_xml(), "XML")
        self.assert_invalid('<?xml version="1.0" encoding="UTF-32"?>' + model_xml(), "XML")
        relation_dtd = '<!DOCTYPE Relationships [<!ENTITY x "bad">]>' + relationships("3D/3dmodel.model")
        with self.assertRaisesRegex(io.ThreeMFError, "DTD/entity"):
            io.load_document(self.package({"_rels/.rels": relation_dtd, "3D/3dmodel.model": model_xml()}))

    def test_xml_size_count_depth_and_source_geometry_limits(self):
        cases = [
            (model_xml(), "max_xml_bytes", 10, "XML part size"),
            (model_xml(), "max_total_xml_bytes", 10, "Total XML"),
            (model_xml(), "max_xml_elements", 10, "element count"),
            (model_xml(), "max_xml_depth", 4, "depth"),
            (model_xml(), "max_source_vertices", 7, "source vertex"),
            (model_xml(), "max_source_faces", 11, "source face"),
            (model_xml(box_object() + box_object(2)), "max_objects", 1, "object count"),
            (model_xml(items='<item objectid="1"/><item objectid="1"/>'), "max_build_items", 1, "Build item count"),
            (model_xml(items='<item objectid="1"/><item objectid="1"/>'), "max_instances", 1, "instance count"),
        ]
        for xml, field, value, message in cases:
            with self.subTest(field=field), self.assertRaisesRegex(io.ThreeMFError, message):
                self.document(xml, limits=replace(io.DEFAULT_LIMITS, **{field: value}))
        parts = {"3D/3dmodel.model": model_xml(), "extra.xml": "<ignored/>"}
        with self.assertRaisesRegex(io.ThreeMFError, "XML part count"):
            io.load_document(self.package(parts), limits=replace(io.DEFAULT_LIMITS, max_xml_parts=1))

    def test_zip_size_and_member_limits(self):
        package = self.package({"3D/3dmodel.model": model_xml(), "thumbnail.png": b"data"})
        cases = [("max_archive_bytes", 1, "archive size"), ("max_zip_members", 1, "member count"),
                 ("max_member_bytes", 10, "member size"), ("max_total_uncompressed_bytes", 10, "uncompressed size")]
        for field, value, message in cases:
            with self.subTest(field=field), self.assertRaisesRegex(io.ThreeMFError, message):
                io.load_document(package, limits=replace(io.DEFAULT_LIMITS, **{field: value}))
        path = self.directory / "not-a-zip.3mf"
        path.write_bytes(b"not a ZIP")
        with self.assertRaisesRegex(io.ThreeMFError, "ZIP"):
            io.load_document(path)

    def test_cumulative_part_limits_and_model_cache(self):
        xml = model_xml(assembly(1, '<component objectid="7" p:path="part.model"/><component objectid="7" p:path="part.model"/>'))
        parts = {"3D/3dmodel.model": xml, "3D/part.model": model_xml(box_object(7), None)}
        package = self.package(parts)
        doc = io.load_document(package)
        self.assertEqual(doc.metadata["statistics"]["models_parsed"], 2)
        self.assertEqual(doc.metadata["statistics"]["meshes_parsed"], 1)
        with self.assertRaisesRegex(io.ThreeMFError, "Model part count"):
            io.load_document(package, limits=replace(io.DEFAULT_LIMITS, max_models=1))
        # Neither file alone exceeds the cumulative budget.
        with self.assertRaisesRegex(io.ThreeMFError, "Total XML"):
            io.load_document(package, limits=replace(io.DEFAULT_LIMITS, max_total_xml_bytes=max(map(len, parts.values())) + 10))
        parts["3D/part.model"] = model_xml(box_object(7), None, unit="unknown")
        with self.assertRaisesRegex(io.ThreeMFError, "unit"):
            io.load_document(self.package(parts))

    def test_component_depth_occurrence_and_output_expansion_limits(self):
        doubled = box_object() + assembly(2, '<component objectid="1"/><component objectid="1"/>')
        for level in range(3, 8):
            doubled += assembly(level, f'<component objectid="{level - 1}"/><component objectid="{level - 1}"/>')
        xml = model_xml(doubled, '<item objectid="7"/>')
        cases = [("max_component_depth", 3, "Component depth"),
                 ("max_component_occurrences", 8, "occurrence"),
                 ("max_output_vertices", 31, "output vertex"), ("max_output_faces", 47, "output face")]
        for field, value, message in cases:
            with self.subTest(field=field), self.assertRaisesRegex(io.ThreeMFError, message):
                self.document(xml, limits=replace(io.DEFAULT_LIMITS, **{field: value}))
        # Expanded budget is global, and coincident welding cannot evade it.
        xml = model_xml(items='<item objectid="1"/><item objectid="1"/>')
        with self.assertRaisesRegex(io.ThreeMFError, "output vertex"):
            self.document(xml, limits=replace(io.DEFAULT_LIMITS, max_output_vertices=15))

    def test_required_extension_in_child_model_is_not_skipped(self):
        parts = {
            "3D/3dmodel.model": model_xml(assembly(1, '<component objectid="9" p:path="part.model"/>')),
            "3D/part.model": model_xml(box_object(9), None,
                attributes='xmlns:u="urn:unknown" requiredextensions="u"'),
        }
        with self.assertRaisesRegex(io.ThreeMFError, "requiredextensions"):
            io.load_document(self.package(parts))

    def test_archive_symlinks_and_unsupported_compression(self):
        path = self.directory / "symlink.3mf"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("3D/3dmodel.model", model_xml())
            link = zipfile.ZipInfo("unsafe-link")
            link.create_system = 3
            link.external_attr = 0o120777 << 16
            archive.writestr(link, "/etc/passwd")
        with self.assertRaisesRegex(io.ThreeMFError, "symlink"):
            io.load_document(path)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_BZIP2) as archive:
            archive.writestr("3D/3dmodel.model", model_xml())
        with self.assertRaisesRegex(io.ThreeMFError, "compression"):
            io.load_document(path)

    def test_closed_document_does_not_use_child_builds(self):
        # A production part's build is not another source of instances.
        parts = {
            "3D/3dmodel.model": model_xml("", '<item objectid="7" p:path="/parts/box.model"/>'),
            "parts/box.model": model_xml(box_object(7), '<item objectid="7"/><item objectid="7"/>'),
        }
        doc = io.load_document(self.package(parts))
        self.assertEqual(len(doc.instances), 1)
        self.assertEqual(len(doc.instances[0].mesh.faces), 12)

    def test_bad_nested_transforms_and_overflow_are_explicit(self):
        for text in (IDENTITY.replace("1", "nan", 1), "0 0 0 0 1 0 0 0 1 0 0 0"):
            self.assert_invalid(model_xml(assembly(1, f'<component objectid="2" transform="{text}"/>') + box_object(2)), "transform")
        huge = "1e308 0 0 0 1 0 0 0 1 0 0 0"
        xml = model_xml(items=f'<item objectid="1" transform="{huge}"/>')
        # One vertex at x=2 overflows despite the linear matrix being invertible.
        xml = xml.replace('x="1.0"', 'x="2.0"')
        self.assert_invalid(xml, "Non-finite transformed")
        self.assert_invalid(model_xml(unit="meter", items='<item objectid="1" transform="1 0 0 0 1 0 0 0 1 1e308 0 0"/>'), "Non-finite transform")

    def test_compatibility_api_and_legacy_namespace_warning(self):
        path = self.package({"3D/3dmodel.model": model_xml(namespace="")})
        meshes = io.load_3mf(path)
        self.assertEqual(len(meshes), 1)
        self.assertIsInstance(meshes[0], trimesh.Trimesh)
        self.assertEqual(meshes[0].metadata["object_id"], 1)
        document = io.load_document(path)
        self.assertTrue(any("Namespace-less" in text for text in document.metadata["warnings"]))
        for kwargs in ({"max_xml_bytes": 0}, {"max_models": -1}, {"max_instances": 1.5}):
            with self.assertRaises(ValueError):
                replace(io.DEFAULT_LIMITS, **kwargs)


if __name__ == "__main__":
    unittest.main()
