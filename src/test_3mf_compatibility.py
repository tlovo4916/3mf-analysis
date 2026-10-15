"""Exporter-shaped synthetic OPC packages; no fixture-specific geometry assumptions.

Namespace/type expectations come from 3MFConsortium/spec_core and spec_materials:
core Chapter 4 / XSD ST_ObjectType and Materials Appendix E.3. These are geometry
analysis compatibility tests, not claims of full extension or CAD support.
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import trimesh

import threemf_io as io
from test_threemf_io import (assembly, box_object, matrix_text, model_xml,
                             relationships, translation)


MATERIAL_URI = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
VENDOR_URI = "https://exporter.example/metadata/1"


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.counter = 0

    def package(self, xml=None, *, parts=None):
        self.counter += 1
        path = self.directory / f"export-{self.counter}.3mf"
        files = {
            "[Content_Types].xml": (
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
                '</Types>'),
            "_rels/.rels": relationships("/3D/3dmodel.model"),
            "3D/3dmodel.model": xml or model_xml(),
        }
        if parts:
            files.update(parts)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in files.items():
                archive.writestr(name, data)
        return path

    def document(self, xml=None, **kwargs):
        return io.load_document(self.package(xml), **kwargs)

    def color_model(self, *, required=True):
        colors = '<m:colorgroup id="20"><m:color color="#33AADDFF"/></m:colorgroup>'
        obj = box_object().replace('<object id="1"', '<object pid="20" pindex="0" id="1"')
        obj = obj.replace('<triangle ', '<triangle pid="20" p1="0" p2="0" p3="0" ')
        return model_xml(colors + obj, attributes=(
            f'xmlns:m="{MATERIAL_URI}" ' + ('requiredextensions="m"' if required else '')))

    def metadata_values(self, document, ids):
        return [document.metadata["source_metadata"][i]["value"] for i in ids]

    def test_required_color_extension_is_geometry_analysis_only_and_strict_stays_strict(self):
        self.assertEqual(io.MATERIAL_NS, MATERIAL_URI)
        path = self.package(self.color_model())
        for loader in (io.load_document, io.load_3mf):
            with self.subTest(loader=loader.__name__), self.assertRaisesRegex(io.ThreeMFError, "requiredextensions"):
                loader(path)
        doc = io.load_document(path, geometry_only=True)
        instance = doc.instances[0]
        self.assertTrue(doc.metadata["geometry_only"])
        self.assertTrue(doc.metadata["appearance_not_recovered"])
        self.assertTrue(instance.metadata["appearance_not_recovered"])
        self.assertTrue(instance.metadata["requires_review"])
        self.assertEqual(len(instance.mesh.faces), 12)
        np.testing.assert_array_equal(instance.mesh.vertices, trimesh.creation.box([2, 4, 6]).vertices)
        extension = next(e for e in doc.metadata["models"][0]["extensions"] if e["namespace"] == MATERIAL_URI)
        self.assertTrue(extension["required"])
        self.assertEqual(extension["handling"], "appearance_not_recovered")
        summaries = doc.metadata["ignored_extension_content"]
        self.assertTrue(any(s["name"] == f"{{{MATERIAL_URI}}}colorgroup" and s["scope"] == "resources" for s in summaries))
        self.assertTrue(any(s["name"] == "p1" and s["count"] == 12 for s in summaries))
        self.assertTrue(any("NOT recovered" in w for w in doc.metadata["warnings"]))
        json.dumps(doc.metadata, allow_nan=False)
        json.dumps(instance.metadata, allow_nan=False)

    def test_optional_appearance_is_declared_ignored_not_claimed_recovered(self):
        doc = self.document(self.color_model(required=False))
        extension = next(e for e in doc.metadata["models"][0]["extensions"] if e["namespace"] == MATERIAL_URI)
        self.assertFalse(extension["required"])
        self.assertEqual(extension["handling"], "appearance_not_recovered")
        self.assertTrue(doc.metadata["appearance_not_recovered"])

    def test_extension_alias_prefix_resolves_but_unverified_uris_never_pass(self):
        doc = self.document(model_xml(attributes=f'xmlns:color="{MATERIAL_URI}" requiredextensions="color"'), geometry_only=True)
        self.assertEqual(len(doc.instances), 1)
        uris = [
            "http://schemas.microsoft.com/3dmanufacturing/material/2015/01",
            "http://schemas.microsoft.com/3dmanufacturing/material/2015/02/",
            "http://schemas.microsoft.com/3dmanufacturing/beamlattice/2017/02",
            "http://schemas.microsoft.com/3dmanufacturing/volumetric/2017/02",
            "http://schemas.microsoft.com/3dmanufacturing/slice/2015/07",
            "http://schemas.3mf.io/3dmanufacturing/displacement/2022/07",
            "http://schemas.microsoft.com/3dmanufacturing/securecontent/2019/04",
            "urn:unknown-required",
        ]
        for uri in uris:
            xml = model_xml(attributes=f'xmlns:x="{uri}" requiredextensions="x"')
            with self.subTest(uri=uri), self.assertRaisesRegex(io.ThreeMFError, "requiredextensions"):
                self.document(xml, geometry_only=True, isolate_build_errors=True)
        with self.assertRaisesRegex(io.ThreeMFError, "requiredextensions"):
            self.document(model_xml(attributes='requiredextensions="notdeclared"'), geometry_only=True)

    def test_unknown_required_child_extension_is_fatal_even_after_valid_item(self):
        xml = model_xml(items='<item objectid="1"/><item objectid="7" p:path="part.model"/>')
        child = model_xml(box_object(7), None, attributes='xmlns:b="urn:unknown-geometry" requiredextensions="b"')
        with self.assertRaisesRegex(io.ThreeMFError, "requiredextensions"):
            io.load_document(self.package(xml, parts={"3D/part.model": child}),
                             geometry_only=True, isolate_build_errors=True)

    def test_unused_support_and_unknown_structures_are_summarized_not_parsed(self):
        resources = (box_object() + '<object id="2" type="support"><mesh><vendorShape/></mesh></object>'
                     '<object id="3" type="other"><unknowngeometry/></object>'
                     '<object id="4" type="future-type"><unknowngeometry/></object>')
        xml = model_xml(resources)
        with self.assertRaisesRegex(io.ThreeMFError, "non-model"):
            self.document(xml)
        doc = self.document(xml, geometry_only=True)
        skipped = doc.metadata["skipped_objects"]
        self.assertEqual([s["object_id"] for s in skipped], [2, 3, 4])
        self.assertTrue(all(not s["structure_validated"] for s in skipped))
        self.assertEqual(doc.metadata["statistics"]["objects_indexed"], 4)
        self.assertEqual(doc.metadata["statistics"]["objects_parsed"], 1)
        self.assertEqual(doc.metadata["statistics"]["meshes_parsed"], 1)

    def test_unused_model_with_unsupported_geometry_is_only_deferred_in_geometry_mode(self):
        xml = model_xml(box_object() + '<object id="2"><unknowngeometry/></object>')
        with self.assertRaisesRegex(io.ThreeMFError, "exactly one"):
            self.document(xml)
        self.assertEqual(self.document(xml, geometry_only=True).metadata["skipped_objects"][0]["object_id"], 2)
        with self.assertRaisesRegex(io.ThreeMFError, "exactly one"):
            self.document(xml, isolate_build_errors=True)

    def test_referenced_core_nonmodel_types_are_analysis_only(self):
        for kind in ("support", "solidsupport", "surface", "other"):
            xml = model_xml(box_object().replace('<object ', f'<object type="{kind}" ', 1))
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(io.ThreeMFError, "non-model"):
                    self.document(xml)
                item = self.document(xml, geometry_only=True).instances[0]
                self.assertEqual(item.metadata["object_type"], kind)
                self.assertEqual(item.metadata["effective_object_type"], kind)
                self.assertEqual(item.metadata["components"][0]["object_type"], kind)
                self.assertFalse(item.metadata["reconstruction_eligible"])
                self.assertIn(f"non_model_object_type:{kind}", item.metadata["reconstruction_blockers"])
                self.assertEqual(bool(item.metadata["core_spec_violations"]), kind == "other")

    def test_component_leaf_types_override_parent_and_mixed_types_are_explicit(self):
        support = box_object(2).replace('<object ', '<object type="support" ', 1)
        group = assembly(3, '<component objectid="1"/><component objectid="2"/>')
        doc = self.document(model_xml(box_object() + support + group, '<item objectid="3"/>'), geometry_only=True)
        item = doc.instances[0]
        self.assertEqual(item.metadata["object_type"], "model")
        self.assertEqual(item.metadata["effective_object_type"], "mixed")
        self.assertEqual(item.metadata["effective_object_types"], ["model", "support"])
        self.assertFalse(item.metadata["reconstruction_eligible"])
        group = assembly(3, '<component objectid="1"/>').replace('<object ', '<object type="surface" ', 1)
        item = self.document(model_xml(box_object() + group, '<item objectid="3"/>'), geometry_only=True).instances[0]
        self.assertEqual(item.metadata["object_type"], "surface")
        self.assertEqual(item.metadata["effective_object_type"], "model")
        self.assertTrue(item.metadata["reconstruction_eligible"])

    def test_unknown_type_and_reachable_unknown_geometry_still_fail_closed(self):
        objects = [box_object().replace('<object ', '<object type="custom" ', 1),
                   box_object().replace('</object>', '<unknowngeometry/></object>'),
                   box_object().replace('<mesh>', '<mesh><beamlattice/>'),
                   box_object().replace('</mesh>', '<u:unknownshape/></mesh>')]
        for obj in objects:
            xml = model_xml(obj, attributes='xmlns:u="urn:optional-geometry"')
            with self.subTest(obj=obj[:100]), self.assertRaises(io.ThreeMFError):
                self.document(xml, geometry_only=True)

    def test_metadata_groups_on_build_and_component_keep_scope_and_source_strings(self):
        def metadata(value):
            return f'<metadatagroup><metadata name="v:ref" preserve="true" type="xs:string">{value}</metadata></metadatagroup>'

        obj = box_object().replace('<mesh>', metadata('  object-value  ') + '<mesh>')
        group = assembly(2, '<component objectid="1">' + metadata('component-value') + '</component>')
        xml = model_xml(obj + group, '<item objectid="2" partnumber="tray-part">' + metadata('item-value') + '</item>',
                        attributes=f'xmlns:v="{VENDOR_URI}"')
        xml = xml.replace('<resources>', '<metadata name="Application">Exporter 1.2</metadata><resources>')
        for geometry_only in (False, True):
            with self.subTest(geometry_only=geometry_only):
                doc = self.document(xml, geometry_only=geometry_only)
                item = doc.instances[0]
                component = item.metadata["components"][0]
                self.assertEqual(self.metadata_values(doc, doc.metadata["metadata_ids"]), ["Exporter 1.2"])
                self.assertEqual(self.metadata_values(doc, component["metadata_ids"]), ["  object-value  "])
                self.assertEqual(self.metadata_values(doc, item.metadata["build_metadata_ids"]), ["item-value"])
                self.assertEqual(self.metadata_values(doc, component["component_path"][0]["metadata_ids"]), ["component-value"])
                self.assertEqual(item.metadata["build_part_number"], "tray-part")
                record = doc.metadata["source_metadata"][component["metadata_ids"][0]]
                self.assertEqual(record["attributes"]["preserve"], "true")
                self.assertEqual(record["name_namespace"], VENDOR_URI)
                self.assertEqual(len(doc.metadata["source_metadata"]), 4)
                self.assertTrue(doc.metadata["source_metadata_summary"]["complete"])
                json.dumps(doc.metadata, allow_nan=False)

    def test_unexpanded_reference_metadata_is_still_preserved_with_source_scope(self):
        group = assembly(2, '<component objectid="1"><metadatagroup>'
                         '<metadata name="v:ref">unreachable-component</metadata>'
                         '</metadatagroup></component>')
        items = ('<item objectid="1"/>'
                 '<item objectid="1" printable="invalid"><metadatagroup>'
                 '<metadata name="v:ref">failed-item</metadata></metadatagroup></item>')
        xml = model_xml(box_object() + group, items, attributes=f'xmlns:v="{VENDOR_URI}"')
        doc = self.document(xml, geometry_only=True, isolate_build_errors=True)
        records = doc.metadata["source_metadata"]
        self.assertTrue(any(m["scope"] == "object:2/component:0" and m["value"] == "unreachable-component" for m in records))
        failure = doc.metadata["failed_build_items"][0]
        self.assertEqual(self.metadata_values(doc, failure["metadata_ids"]), ["failed-item"])
        self.assertEqual(doc.metadata["skipped_objects"][0]["object_id"], 2)

    def test_unknown_nested_reference_content_is_not_metadata(self):
        for xml in (model_xml(items='<item objectid="1"><unknownshape/></item>'),
                    model_xml(box_object() + assembly(2, '<component objectid="1"><unknownshape/></component>'),
                              '<item objectid="2"/>')):
            with self.subTest(xml=xml[-150:]), self.assertRaisesRegex(io.ThreeMFError, "nested content"):
                self.document(xml, geometry_only=True)

    def test_optional_extension_content_has_bounded_declared_scope_summaries(self):
        obj = box_object().replace('</mesh>', '<t:trianglesets><t:triangleset name="face"/></t:trianglesets></mesh>')
        xml = model_xml(obj, attributes=f'xmlns:t="{io.TRIANGLESETS_NS}" xmlns:v="{VENDOR_URI}" v:producer="test"')
        xml = xml.replace('<resources>', '<v:settings><v:setting/></v:settings><resources>')
        doc = self.document(xml, geometry_only=True)
        summaries = doc.metadata["ignored_extension_content"]
        self.assertTrue(any(s["scope"] == "model" and s["name"].endswith('}settings') for s in summaries))
        self.assertTrue(any(s["scope"] == "mesh" and s["namespace"] == io.TRIANGLESETS_NS for s in summaries))
        self.assertTrue(any(s["kind"] == "extension_attribute" and s["name"].endswith('}producer') for s in summaries))
        self.assertTrue(all(s["handling"] == "not_interpreted_not_recovered" for s in summaries))
        doc = self.document(xml, geometry_only=True, limits=replace(io.DEFAULT_LIMITS, max_ignored_content_summaries=1))
        self.assertEqual(len(doc.metadata["ignored_extension_content"]), 1)
        self.assertGreater(doc.metadata["ignored_content_summary_occurrences_omitted"], 0)

    def test_metadata_limits_truncate_explicitly_and_do_not_duplicate_per_occurrence(self):
        meta = ''.join(f'<metadata name="v:m{i}">{"abcdef" * 30}</metadata>' for i in range(5))
        obj = box_object().replace('<mesh>', '<metadatagroup>' + meta + '</metadatagroup><mesh>')
        xml = model_xml(obj, '<item objectid="1"/><item objectid="1"/>', attributes=f'xmlns:v="{VENDOR_URI}"')
        limits = replace(io.DEFAULT_LIMITS, max_metadata_entries=2, max_metadata_text_chars=20,
                         max_metadata_total_chars=50)
        doc = self.document(xml, limits=limits)
        summary = doc.metadata["source_metadata_summary"]
        self.assertEqual(summary["entries_preserved"], 2)
        self.assertEqual(summary["entries_omitted"], 3)
        self.assertEqual(summary["entries_truncated"], 2)
        self.assertLessEqual(summary["characters_preserved"], 50)
        self.assertFalse(summary["complete"])
        self.assertEqual(doc.instances[0].metadata["metadata_ids"], doc.instances[1].metadata["metadata_ids"])
        self.assertTrue(all(len(m["value"]) <= 20 for m in doc.metadata["source_metadata"]))

    def test_isolated_bad_mesh_and_missing_reference_keep_valid_items_and_indices(self):
        bad = box_object(2, faces=[[0, 1, 999]])
        xml = model_xml(box_object() + bad, '<item objectid="2"/><item objectid="1"/><item objectid="99"/>')
        with self.assertRaisesRegex(io.ThreeMFError, "out of bounds"):
            self.document(xml, geometry_only=True)
        doc = self.document(xml, geometry_only=True, isolate_build_errors=True)
        self.assertEqual([i.metadata["build_index"] for i in doc.instances], [1])
        errors = doc.metadata["failed_build_items"]
        self.assertEqual([e["build_index"] for e in errors], [0, 2])
        self.assertIn("out of bounds", errors[0]["reason"])
        self.assertIn("Missing object", errors[1]["reason"])
        self.assertFalse(doc.metadata["document_coverage_complete"])
        self.assertFalse(doc.metadata["printable_build_coverage_complete"])
        json.dumps(doc.metadata, allow_nan=False)

    def test_strict_item_isolation_does_not_relax_types_or_skip_unreferenced_structures(self):
        bad = box_object(2).replace('<object ', '<object type="support" ', 1)
        xml = model_xml(box_object() + bad, '<item objectid="1"/><item objectid="2"/>')
        doc = self.document(xml, isolate_build_errors=True)
        self.assertFalse(doc.metadata["geometry_only"])
        self.assertEqual(len(doc.instances), 1)
        self.assertIn("non-model", doc.metadata["failed_build_items"][0]["reason"])
        with self.assertRaisesRegex(io.ThreeMFError, "non-model"):
            self.document(model_xml(box_object() + bad), isolate_build_errors=True)

    def test_failed_assembly_does_not_leak_partial_components(self):
        obj = box_object() + assembly(2, '<component objectid="1"/><component objectid="999"/>')
        doc = self.document(model_xml(obj, '<item objectid="2"/><item objectid="1"/>'),
                            geometry_only=True, isolate_build_errors=True)
        self.assertEqual(len(doc.instances), 1)
        self.assertEqual(doc.instances[0].metadata["object_id"], 1)
        self.assertEqual(len(doc.instances[0].mesh.faces), 12)
        stats = doc.metadata["statistics"]
        self.assertEqual(stats["meshes_parsed"], 1)
        self.assertEqual(stats["output_faces"], 24)  # budget includes failed attempt
        self.assertEqual(stats["emitted_output_faces"], 12)
        self.assertTrue(stats["expansion_counts_include_failed_attempts"])

    def test_failed_source_geometry_is_cached_without_charging_it_again(self):
        xml = model_xml(box_object() + box_object(2, faces=[[0, 1, 99]]),
                        '<item objectid="2"/><item objectid="2"/><item objectid="1"/>')
        doc = self.document(xml, geometry_only=True, isolate_build_errors=True,
                            limits=replace(io.DEFAULT_LIMITS, max_source_vertices=16))
        self.assertEqual(doc.metadata["statistics"]["objects_parsed"], 2)
        self.assertEqual(doc.metadata["statistics"]["source_vertices"], 16)
        self.assertEqual(len(doc.metadata["failed_build_items"]), 2)

    def test_invalid_transform_boolean_missing_part_and_cycles_are_isolatable(self):
        items = ('<item objectid="1" transform="0 0 0 0 1 0 0 0 1 0 0 0"/>'
                 '<item objectid="1" printable="invalid"/>'
                 '<item objectid="7" p:path="missing.model"/>'
                 '<item objectid="2"/><item objectid="1"/>')
        xml = model_xml(box_object() + assembly(2, '<component objectid="2"/>'), items)
        doc = self.document(xml, geometry_only=True, isolate_build_errors=True)
        self.assertEqual(doc.instances[0].metadata["build_index"], 4)
        self.assertEqual(len(doc.metadata["failed_build_items"]), 4)
        self.assertTrue(any("Cyclic" in e["reason"] for e in doc.metadata["failed_build_items"]))

    def test_zero_valid_instances_still_raises(self):
        xml = model_xml(box_object(faces=[[999, 0, 1]]))
        with self.assertRaisesRegex(io.ThreeMFError, "no printable.*failed build.*out of bounds"):
            self.document(xml, geometry_only=True, isolate_build_errors=True)

    def test_isolation_cannot_hide_dtd_or_malformed_xml_in_referenced_child(self):
        xml = model_xml(items='<item objectid="1"/><item objectid="7" p:path="bad.model"/>')
        malformed = ['<model>', '<!DOCTYPE model [<!ENTITY x "data">]>' + model_xml(box_object(7), None)]
        for child in malformed:
            with self.subTest(child=child[:100]), self.assertRaisesRegex(io.ThreeMFError, "XML|DTD/entity"):
                io.load_document(self.package(xml, parts={"3D/bad.model": child}),
                                 geometry_only=True, isolate_build_errors=True)

    def test_xml_preflight_cannot_be_short_circuited_by_item_or_component_errors(self):
        child = '<!DOCTYPE model [<!ENTITY x "data">]>' + model_xml(box_object(7), None)
        cases = [
            model_xml(items='<item objectid="1"/><item objectid="7" printable="bad" p:path="bad.model"/>'),
            model_xml(box_object() + assembly(2,
                '<component objectid="1" transform="bad"/><component objectid="7" p:path="bad.model"/>'),
                '<item objectid="1"/><item objectid="2"/>'),
            # A missing first component must not hide the second component's DTD.
            model_xml(box_object() + assembly(2,
                '<component objectid="999"/><component objectid="7" p:path="bad.model"/>'),
                '<item objectid="1"/><item objectid="2"/>'),
            model_xml(box_object() + assembly(2,
                '<component objectid="2"/><component objectid="7" p:path="bad.model"/>'),
                '<item objectid="1"/><item objectid="2"/>'),
        ]
        for xml in cases:
            with self.subTest(xml=xml[-160:]), self.assertRaisesRegex(io.ThreeMFError, "DTD/entity"):
                io.load_document(self.package(xml, parts={"3D/bad.model": child}),
                                 geometry_only=True, isolate_build_errors=True)

    def test_isolation_cannot_hide_unsafe_archive_or_reference_paths(self):
        path = self.package(parts={"../escape.xml": "unused"})
        with self.assertRaisesRegex(io.ThreeMFError, "Unsafe"):
            io.load_document(path, geometry_only=True, isolate_build_errors=True)
        for target in ("../../escape.model", "http://example.invalid/part.model", "%252e%252e/bad.model", "bad.model#fragment"):
            for component in (False, True):
                bad_ref = f'<component objectid="1" p:path="{target}"/>'
                resources = box_object() + (assembly(2, bad_ref) if component else '')
                bad_item = '<item objectid="2"/>' if component else f'<item objectid="1" p:path="{target}"/>'
                xml = model_xml(resources, '<item objectid="1"/>' + bad_item)
                with self.subTest(target=target, component=component), self.assertRaises(io.ThreeMFError):
                    self.document(xml, geometry_only=True, isolate_build_errors=True)

    def test_unsafe_paths_are_fatal_even_when_an_earlier_semantic_error_would_skip_them(self):
        cases = [
            model_xml(items='<item objectid="1"/><item objectid="1" printable="bad" p:path="../../escape.model"/>'),
            model_xml(box_object() + assembly(2,
                '<component objectid="1" transform="bad"/><component objectid="1" p:path="../../escape.model"/>'),
                '<item objectid="1"/><item objectid="2"/>'),
            # Deferred unreachable geometry is not exemption from package safety.
            model_xml(box_object() + assembly(2, '<component objectid="1" p:path="../../escape.model"/>')),
        ]
        for xml in cases:
            with self.subTest(xml=xml[-200:]), self.assertRaisesRegex(io.ThreeMFError, "escapes archive") as context:
                self.document(xml, geometry_only=True, isolate_build_errors=True)
            self.assertIs(type(context.exception), io.ThreeMFError)

    def test_bad_child_model_parse_is_cached_between_isolated_build_items(self):
        xml = model_xml(items='<item objectid="1"/><item objectid="7" p:path="part.model"/><item objectid="7" p:path="part.model"/>')
        child = model_xml(box_object(7), None, unit="unknown")
        doc = io.load_document(self.package(xml, parts={"3D/part.model": child}),
                               geometry_only=True, isolate_build_errors=True)
        self.assertEqual(len(doc.instances), 1)
        self.assertEqual(len(doc.metadata["failed_build_items"]), 2)
        self.assertEqual(doc.metadata["statistics"]["xml_parts_parsed"], 3)  # rels, root, bad child once
        self.assertEqual(doc.metadata["statistics"]["model_parts_attempted"], 2)
        self.assertTrue(all("unit" in e["reason"] for e in doc.metadata["failed_build_items"]))

    def test_isolation_does_not_reset_or_swallow_geometry_and_xml_budgets(self):
        xml = model_xml(items='<item objectid="1"/><item objectid="1"/>')
        for field, limit in (("max_output_vertices", 15), ("max_output_faces", 23),
                             ("max_source_vertices", 7), ("max_source_faces", 11),
                             ("max_instances", 1), ("max_component_occurrences", 1)):
            with self.subTest(field=field), self.assertRaisesRegex(io.ThreeMFError, "limit exceeded"):
                self.document(xml, geometry_only=True, isolate_build_errors=True,
                              limits=replace(io.DEFAULT_LIMITS, **{field: limit}))
        child = model_xml(box_object(7), None)
        xml = model_xml(items='<item objectid="1"/><item objectid="7" p:path="part.model"/>')
        with self.assertRaisesRegex(io.ThreeMFError, "Model part count"):
            io.load_document(self.package(xml, parts={"3D/part.model": model_xml(box_object(7), None, unit="bad")}),
                             geometry_only=True, isolate_build_errors=True,
                             limits=replace(io.DEFAULT_LIMITS, max_models=1))
        with self.assertRaisesRegex(io.ThreeMFError, "element count"):
            io.load_document(self.package(xml, parts={"3D/part.model": child}), geometry_only=True,
                             isolate_build_errors=True, limits=replace(io.DEFAULT_LIMITS, max_xml_elements=40))

    def test_coincident_and_touching_components_remain_distinct_including_strict_mode(self):
        for offset in (0, 2):
            children = f'<component objectid="1"/><component objectid="1" transform="{matrix_text(translation(offset, 0, 0))}"/>'
            xml = model_xml(box_object() + assembly(2, children), '<item objectid="2"/>')
            for geometry_only in (False, True):
                with self.subTest(offset=offset, geometry_only=geometry_only):
                    item = self.document(xml, geometry_only=geometry_only).instances[0]
                    self.assertEqual(len(item.mesh.vertices), 16)
                    self.assertEqual(len(item.mesh.faces), 24)
                    self.assertEqual(len(item.mesh.split(only_watertight=False)), 2)
                    c0, c1 = item.metadata["components"]
                    self.assertEqual((c0["vertex_start"], c0["vertex_end"]), (0, 8))
                    self.assertEqual((c1["vertex_start"], c1["vertex_end"]), (8, 16))
                    self.assertTrue(set(item.mesh.faces[:12].ravel()).isdisjoint(item.mesh.faces[12:].ravel()))
                    self.assertFalse(item.metadata["mesh_cleanup"][0]["cross_component_welding"])
                    self.assertEqual(item.metadata["mesh_cleanup"][0]["scope"], "leaf_occurrence_only")

    def test_soup_source_to_instance_vertex_map_is_complete_with_mirrors_and_unused_vertices(self):
        box = trimesh.creation.box([2, 4, 6])
        raw = np.vstack([box.triangles.reshape(-1, 3), [[99, 99, 99], [99, 99, 99]]])
        raw_faces = np.arange(36).reshape(-1, 3)
        mirror = translation(8, 2, 0) @ np.diag([-1, 1, 1, 1])
        objects = box_object(vertices=raw, faces=raw_faces) + assembly(2,
            f'<component objectid="1"/><component objectid="1" transform="{matrix_text(mirror)}"/>')
        doc = self.document(model_xml(objects, '<item objectid="2"/>'), geometry_only=True)
        item = doc.instances[0]
        self.assertEqual(len(item.mesh.vertices), 18)
        self.assertEqual(item.metadata["mesh_cleanup"][0]["vertices_before"], 76)
        self.assertEqual(item.metadata["mesh_cleanup"][0]["vertices_merged"], 58)
        for c in item.metadata["components"]:
            ids = np.array(c["source_vertex_to_instance_vertex_ids"])
            self.assertEqual(len(ids), len(raw))
            self.assertEqual(ids[-1], ids[-2])
            matrix = np.array(c["source_to_world_mm"])
            np.testing.assert_array_equal(item.mesh.vertices[ids], raw @ matrix[:3, :3].T + matrix[:3, 3])
            source_faces = raw_faces[:, ::-1] if c["winding_reversed_for_reflection"] else raw_faces
            np.testing.assert_array_equal(item.mesh.faces[c["face_start"]:c["face_end"]], ids[source_faces])
            self.assertGreaterEqual(int(ids.min()), c["vertex_start"])
            self.assertLess(int(ids.max()), c["vertex_end"])
            self.assertIn("source_3mf", c["source_coordinate_storage"])
        self.assertEqual(doc.metadata["statistics"]["meshes_parsed"], 1)

    def test_nonprintable_inclusion_preserves_flag_index_and_reconstruction_blocker(self):
        xml = model_xml(items='<item objectid="1" printable="false"/><item objectid="1"/>')
        doc = self.document(xml, geometry_only=True)
        self.assertEqual([i.metadata["build_index"] for i in doc.instances], [1])
        self.assertFalse(doc.metadata["document_coverage_complete"])
        self.assertTrue(doc.metadata["printable_build_coverage_complete"])
        doc = self.document(xml, geometry_only=True, include_nonprintable=True)
        self.assertEqual([i.metadata["build_index"] for i in doc.instances], [0, 1])
        self.assertEqual([i.metadata["printable"] for i in doc.instances], [False, True])
        self.assertFalse(doc.instances[0].metadata["reconstruction_eligible"])
        self.assertIn("non_printable_build_item", doc.instances[0].metadata["reconstruction_blockers"])
        self.assertEqual(doc.metadata["skipped_build_items"], [])
        self.assertTrue(doc.metadata["document_coverage_complete"])
        only = model_xml(items='<item objectid="1" printable="0"/>')
        self.assertEqual(len(self.document(only, include_nonprintable=True).instances), 1)

    def test_unexpanded_nonprintable_support_is_skipped_but_inclusion_validates_geometry(self):
        bad_support = '<object id="2" type="support"><unsupportedgeometry/></object>'
        xml = model_xml(box_object() + bad_support, '<item objectid="2" printable="0"/><item objectid="1"/>')
        doc = self.document(xml, geometry_only=True)
        self.assertEqual(doc.metadata["statistics"]["objects_parsed"], 1)
        self.assertEqual(doc.metadata["skipped_objects"][0]["object_id"], 2)
        self.assertEqual(doc.metadata["skipped_build_items"][0]["object_type"], "support")
        with self.assertRaisesRegex(io.ThreeMFError, "exactly one"):
            self.document(xml, geometry_only=True, include_nonprintable=True)
        doc = self.document(xml, geometry_only=True, include_nonprintable=True, isolate_build_errors=True)
        self.assertEqual(len(doc.instances), 1)
        self.assertFalse(doc.metadata["failed_build_items"][0]["printable"])

    def test_mixed_units_and_cached_inward_winding_remain_unrepaired(self):
        box = trimesh.creation.box([2, 4, 6])
        inward = box.faces[:, ::-1]
        mirror = translation(1, 2, 3) @ np.diag([-1, 1, 1, 1])
        root = model_xml(assembly(10, f'<component objectid="7" p:path="leaf.model" transform="{matrix_text(mirror)}"/>'),
                         '<item objectid="10"/><item objectid="10"/>', unit="inch")
        child = model_xml(box_object(7, faces=inward), None, unit="centimeter")
        doc = io.load_document(self.package(root, parts={"3D/leaf.model": child}), geometry_only=True)
        expected = mirror.copy()
        expected[:3, 3] *= 25.4
        expected = expected @ np.diag([10, 10, 10, 1])
        for item in doc.instances:
            component = item.metadata["components"][0]
            np.testing.assert_allclose(component["source_to_world_mm"], expected)
            np.testing.assert_array_equal(item.mesh.faces, inward[:, ::-1])
            self.assertLess(item.mesh.volume, 0)
            self.assertFalse(item.metadata["mesh_cleanup"][0]["normals_fixed"])
        self.assertEqual(doc.metadata["statistics"]["meshes_parsed"], 1)
        ids = doc.instances[1].metadata["components"][0]["source_vertex_to_instance_vertex_ids"].copy()
        doc.instances[0].metadata["components"][0]["source_vertex_to_instance_vertex_ids"][0] = 999
        self.assertEqual(doc.instances[1].metadata["components"][0]["source_vertex_to_instance_vertex_ids"], ids)


if __name__ == "__main__":
    unittest.main()
