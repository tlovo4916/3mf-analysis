"""Synthetic role evidence regressions; no booleans, repairs, or SW runtime claim.

Negative volumes use core type=model, even when they fully enclose normal parts.
Tests exercise package budgets and source vetoes independently of approval flags.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

import build_solidworks as builder
import threemf_io as io
from slicer_metadata import CONFIG_MEMBER, GEOMETRY_FRAME, ROLE_SCOPE
from test_threemf_io import assembly, box_object, matrix_text, model_xml, relationships, translation


class SlicerMetadataTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.counter = 0

    def package(self, xml=None, config=None, *, parts=None, duplicate=False):
        self.counter += 1
        path = self.directory / f"roles-{self.counter}.3mf"
        files = {"_rels/.rels": relationships("/3D/3dmodel.model"),
                 "3D/3dmodel.model": xml or self.model()}
        if config is not None:
            files[CONFIG_MEMBER] = config
        files.update(parts or {})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, value in files.items():
                    archive.writestr(name, value)
                if duplicate:
                    archive.writestr(CONFIG_MEMBER, config)
        return path

    @staticmethod
    def model(items='<item objectid="10"/>', extra="", children='<component objectid="1"/>', attributes=""):
        return model_xml(box_object() + extra + assembly(10, children), items, attributes=attributes)

    @staticmethod
    def config(role="normal_part", *, extra="", metadata="", object_id=10, part_id=1):
        return (f'<config><object id="{object_id}"><metadata key="name" value="Parent"/>'
                f'<part id="{part_id}" subtype="{role}"><metadata key="name" value="Role name"/>'
                f'{metadata}<mesh_stat edges_fixed="0"/></part></object>{extra}</config>')

    def load(self, config=None, xml=None, **kwargs):
        parts = kwargs.pop("parts", None)
        return io.load_document(self.package(xml, config, parts=parts), **kwargs)

    @staticmethod
    def plan(source, version=2):
        return {"schema_version": version, "strategy": "review_required_plate_profile",
                "parameters": {"thickness_mm": 1.0}, "outer_points_mm": [[0, 0], [5, 0], [5, 5], [0, 5]],
                "cuts": [], "approved_for_build": True, "build_blockers": [],
                "requires_approximation_acceptance": True, "accepted_approximation": True,
                "source_placement_applied": False, "source_provenance": source}

    def assert_veto(self, document, *, all_instances=True):
        self.assertTrue(document.metadata["reconstruction_veto"])
        self.assertTrue(document.metadata["reconstruction_blockers"])
        instances = document.instances if all_instances else [i for i in document.instances if i.metadata["reconstruction_veto"]]
        for instance in instances:
            md = instance.metadata
            self.assertTrue(md["reconstruction_veto"])
            self.assertFalse(md["reconstruction_eligible"])
            self.assertFalse(md["material_semantics_complete"])
            self.assertTrue(md["reconstruction_blockers"])
            source = deepcopy(md)
            # Removing copied decisions does not remove original role evidence.
            for key in ("reconstruction_blockers", "core_spec_violations", "reconstruction_eligible"):
                source.pop(key, None)
            source["reconstruction_veto"] = False
            blockers = builder.source_build_blockers(source)
            self.assertTrue(blockers)
            snapshot = deepcopy(source)
            for version in (1, 2):
                plan = self.plan(source, version)
                with self.assertRaisesRegex(ValueError, "source_provenance"):
                    builder.validate(plan)
                with self.assertRaisesRegex(ValueError, "source_provenance"):
                    builder.offline_validation_receipt(plan)
                self.assertEqual(builder.plan_volume_mm3(plan), 25)
            self.assertEqual(source, snapshot)

    def test_normal_role_explains_purpose_but_never_authorizes_reconstruction(self):
        doc = self.load(self.config())
        md = doc.instances[0].metadata
        self.assertEqual(md["role_mapping_status"], "resolved")
        self.assertTrue(md["material_semantics_complete"])
        self.assertEqual(md["material_semantics_scope"], ROLE_SCOPE)
        self.assertFalse(md["reconstruction_veto"])
        self.assertTrue(md["requires_review"])
        self.assertFalse(doc.metadata["reconstruction_authorized_by_metadata"])
        self.assertEqual(builder.source_build_blockers(md), [])
        c = md["components"][0]
        self.assertEqual(c["slicer_role"], "normal_part")
        self.assertEqual(c["slicer_config_name"], "Role name")
        self.assertEqual(c["slicer_config_member"], CONFIG_MEMBER)
        locator = c["slicer_config_locator"]
        self.assertIn("/part[", locator)
        self.assertEqual(ET.fromstring(self.config()).find(locator.removeprefix("config/")).get("id"), "1")
        self.assertEqual(c["instance_face_range_half_open"], [0, 12])
        self.assertEqual(c["slicer_role_evidence"], md["slicer_role_evidence"]["leaves"][0])
        self.assertEqual(md, doc.instances[0].mesh.metadata)
        plan = self.plan(md)
        plan["approved_for_build"] = False
        with self.assertRaisesRegex(ValueError, "approved_for_build"):
            builder.validate(plan)
        plan["approved_for_build"] = True
        plan["accepted_approximation"] = False
        with self.assertRaisesRegex(ValueError, "accepted_approximation"):
            builder.validate(plan)
        json.dumps(doc.metadata, allow_nan=False)
        json.dumps(md, allow_nan=False)

    def test_negative_core_model_enclosing_normal_part_is_not_material_boolean(self):
        negative = trimesh.creation.box(extents=[20, 30, 40])
        xml = self.model(extra=box_object(2, name="enclosing negative", vertices=negative.vertices, faces=negative.faces),
                         children='<component objectid="1"/><component objectid="2"/>')
        config = ('<config><object id="10"><part id="1" subtype="normal_part"/>'
                  '<part id="2" subtype="negative_part"/></object></config>')
        doc = self.load(config, xml)
        self.assert_veto(doc)
        md = doc.instances[0].metadata
        self.assertEqual(md["effective_object_types"], ["model"])
        self.assertEqual(md["core_spec_violations"], [])
        self.assertEqual(md["role_mapping_status"], "resolved")
        self.assertEqual(doc.metadata["slicer_metadata"]["resolved_role_occurrence_counts"],
                         {"normal_part": 1, "negative_part": 1})
        self.assertEqual(doc.metadata["slicer_metadata"]["negative_component_occurrence_count"], 1)
        self.assertIn("slicer_non_material_role:negative_part", builder.source_build_blockers(md))
        self.assertEqual([c["instance_face_range_half_open"] for c in md["components"]], [[0, 12], [12, 24]])
        self.assertEqual(len(doc.instances[0].mesh.faces), 24)
        self.assertEqual(len(doc.instances[0].mesh.split(only_watertight=False)), 2)
        self.assertFalse(md["printed_material_geometry_recovered"])
        self.assertTrue(all(not l["boolean_difference_or_union_evaluated"] for l in md["slicer_role_evidence"]["leaves"]))
        core = self.load(None, xml).instances[0]
        np.testing.assert_array_equal(doc.instances[0].mesh.vertices, core.mesh.vertices)
        np.testing.assert_array_equal(doc.instances[0].mesh.faces, core.mesh.faces)
        self.assertEqual(md["mesh_cleanup"], core.metadata["mesh_cleanup"])

    def test_non_material_unknown_missing_and_blank_roles_veto_even_with_core_model(self):
        for role in ("negative_part", "modifier_part", "parameter_modifier", "modifier", "support_enforcer",
                     "support_blocker", "support_part", "support", "future", "", " normal_part ", "NORMAL_PART"):
            with self.subTest(role=role):
                doc = self.load(self.config(role))
                self.assert_veto(doc)
                leaf = doc.instances[0].metadata["components"][0]
                if role in ("support", "future", "", " normal_part ", "NORMAL_PART"):
                    self.assertEqual(doc.instances[0].metadata["role_mapping_status"], "unresolved")
                    self.assertEqual(leaf["slicer_role_mapping_status"], "unresolved")
                    self.assertIn("unknown_or_missing_slicer_subtype", leaf["slicer_role_evidence"]["mapping_issues"])
        self.assert_veto(self.load(self.config().replace(' subtype="normal_part"', '')))

    def test_negative_role_is_recomputed_from_raw_evidence_when_decisions_are_falsified(self):
        md = self.load(self.config("negative_part")).instances[0].metadata
        source = deepcopy(md)
        for key in ("reconstruction_blockers", "core_spec_violations"):
            source[key] = []
        source.update(reconstruction_eligible=True, reconstruction_veto=False, material_semantics_complete=True,
                      role_mapping_status="resolved")
        self.assertIn("slicer_non_material_role:negative_part", builder.source_build_blockers(source))
        # Falsifying the derived role alone conflicts with the raw declaration.
        source["slicer_role_evidence"]["parts"][0]["subtype"] = "normal_part"
        self.assertTrue(any("raw_declaration" in b for b in builder.source_build_blockers(source)))
        with self.assertRaisesRegex(ValueError, "source_provenance"):
            builder.validate(self.plan(source))

    def test_whole_instance_veto_cannot_be_bypassed_by_selecting_only_the_normal_leaf(self):
        xml = self.model(extra=box_object(2), children='<component objectid="1"/><component objectid="2"/>')
        config = ('<config><object id="10"><part id="1" subtype="normal_part"/>'
                  '<part id="2" subtype="negative_part"/></object></config>')
        source = deepcopy(self.load(config, xml).instances[0].metadata)
        source.update(reconstruction_veto=False, reconstruction_eligible=True, reconstruction_blockers=[],
                      material_semantics_complete=True)
        source["components"] = source["components"][:1]
        self.assertTrue(builder.source_build_blockers(source))
        del source["slicer_role_evidence"]
        self.assertTrue(builder.source_build_blockers(source))

    def test_multiple_build_instances_are_not_private_instance_ids(self):
        items = '<item objectid="10"/><item objectid="10" transform="1 0 0 0 1 0 0 0 1 50 0 0"/>'
        context = ('<plate><metadata key="plater_id" value="1"/><model_instance>'
                   '<metadata key="object_id" value="10"/><metadata key="instance_id" value="0"/>'
                   '</model_instance><model_instance><metadata key="object_id" value="10"/>'
                   '<metadata key="instance_id" value="1"/></model_instance></plate>')
        for role in ("normal_part", "negative_part"):
            with self.subTest(role=role):
                doc = self.load(self.config(role, extra=context), self.model(items))
                self.assert_veto(doc)
                self.assertEqual([i.metadata["slicer_role_evidence"]["root_build_occurrence_count"] for i in doc.instances], [2, 2])
                self.assertTrue(all(i.metadata["role_mapping_status"] == "unresolved" for i in doc.instances))
                self.assertTrue(all("repeated_build_object_occurrence" in i.metadata["components"][0]["slicer_role_evidence"]["mapping_issues"] for i in doc.instances))
                np.testing.assert_allclose(doc.instances[1].mesh.bounds.mean(axis=0), [50, 0, 0])

    def test_distinct_build_instances_keep_role_veto_local_and_package_veto_aggregate(self):
        xml = self.model('<item objectid="10"/><item objectid="20"/>',
                         extra=box_object(2) + assembly(20, '<component objectid="2"/>'))
        config = ('<config><object id="10"><part id="1" subtype="normal_part"/></object>'
                  '<object id="20"><part id="2" subtype="negative_part"/></object></config>')
        doc = self.load(config, xml)
        self.assert_veto(doc, all_instances=False)
        self.assertEqual(builder.source_build_blockers(doc.instances[0].metadata), [])
        self.assertFalse(doc.instances[0].metadata["reconstruction_veto"])
        self.assertTrue(doc.instances[1].metadata["reconstruction_veto"])
        self.assertFalse(doc.metadata["material_semantics_complete"])
        self.assertEqual(doc.metadata["role_mapping_status"], "resolved")

    def test_duplicate_objects_parts_and_metadata_do_not_guess_mapping(self):
        configs = [
            '<config><object id="10"><part id="1" subtype="normal_part"/></object><object id="10"/></config>',
            '<config><object id="10"><part id="1" subtype="normal_part"/><part id="1" subtype="negative_part"/></object></config>',
            self.config(metadata='<metadata key="name" value="duplicate"/>'),
            self.config().replace('id="10"', 'id="010"'),
            self.config().replace('id="1"', 'id="+1"'),
        ]
        for config in configs:
            with self.subTest(config=config):
                doc = self.load(config)
                self.assert_veto(doc)
                self.assertEqual(doc.instances[0].metadata["role_mapping_status"], "unresolved")

    def test_missing_parts_objects_and_extra_negative_declarations_fail_closed(self):
        for config in ('<config/>', '<config><object id="10"/></config>', self.config(part_id=3),
                       self.config(object_id=30),
                       self.config().replace('</object>', '<part id="2" subtype="negative_part"/></object>'),
                       self.config().replace('</config>', '<object id="20"><part id="2" subtype="negative_part"/></object></config>')):
            with self.subTest(config=config):
                self.assert_veto(self.load(config))

    def test_nested_components_and_repeated_leaf_occurrences_are_unresolved(self):
        models = [self.model(children='<component objectid="1"/><component objectid="1"/>'),
                  self.model(extra=assembly(3, '<component objectid="1"/>'), children='<component objectid="3"/>')]
        for xml in models:
            with self.subTest(xml=xml[-250:]):
                doc = self.load(self.config(), xml)
                self.assert_veto(doc)
                for c in doc.instances[0].metadata["components"]:
                    self.assertEqual(c["slicer_role_mapping_status"], "unresolved")
                    self.assertIsNone(c["slicer_config_locator"])

    def test_direct_mesh_root_and_external_build_root_are_not_implicitly_mapped(self):
        direct = model_xml(box_object(), '<item objectid="1"/>')
        self.assert_veto(self.load(self.config(object_id=1), direct))
        xml = model_xml('', '<item objectid="10" p:path="child.model"/>')
        child = self.model(items=None)
        doc = self.load(self.config(), xml, parts={"3D/child.model": child})
        self.assert_veto(doc)

    def test_unique_cross_model_direct_mapping_but_id_conflicts_never_use_first_match(self):
        root = model_xml(assembly(10, '<component objectid="1" p:path="a.model"/>'), '<item objectid="10"/>')
        part = model_xml(box_object(), None)
        doc = self.load(self.config(), root, parts={"3D/a.model": part})
        self.assertEqual(builder.source_build_blockers(doc.instances[0].metadata), [])
        # Same numeric ID in another direct source model is ambiguous even if a
        # config has just one apparently matching part record.
        conflict = model_xml(assembly(10, '<component objectid="1" p:path="a.model"/><component objectid="1" p:path="b.model"/>'), '<item objectid="10"/>')
        doc = self.load(self.config(), conflict, parts={"3D/a.model": part, "3D/b.model": part})
        self.assert_veto(doc)
        self.assertEqual(doc.instances[0].metadata["slicer_role_evidence"]["cross_model_id_conflicts"], [1])
        self.assertTrue(all("cross_model_id_conflict" in c["slicer_role_evidence"]["mapping_issues"] for c in doc.instances[0].metadata["components"]))
        # A conflicting indexed unused resource also makes numeric-only private
        # IDs unsafe; no interpretation based on name, UUID or part order.
        conflict = root.replace('<resources>', '<resources>' + box_object(1))
        self.assert_veto(self.load(self.config(), conflict, parts={"3D/a.model": part}))

    def test_plate_assembly_private_matrices_source_and_thumbnail_are_opaque(self):
        extra = ('<plate><metadata key="plater_id" value="1"/><metadata key="plater_name" value="Tray A"/>'
                 '<metadata key="thumbnail_file" value="https://invalid.example/secret.png"/>'
                 '<model_instance><metadata key="object_id" value="10"/>'
                 '<metadata key="instance_id" value="7"/></model_instance></plate>'
                 '<assemble><assemble_item object_id="10" instance_id="7" transform="not a core transform" '
                 'offset="NaN inf invalid"/></assemble>')
        meta = ('<metadata key="matrix" value="0 0 0 singular"/>'
                '<metadata key="source_file" value="file:///outside/secret.step"/>')
        xml = self.model().replace('<item objectid="10"/>', f'<item objectid="10" transform="{matrix_text(translation(5, 6, 7))}"/>')
        path = self.package(xml, self.config(metadata=meta, extra=extra), parts={"Metadata/plate_1.png": b"not used"})
        opened = []
        original = zipfile.ZipFile.open

        def monitor(archive, name, *args, **kwargs):
            opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else name)
            return original(archive, name, *args, **kwargs)

        with patch.object(zipfile.ZipFile, "open", monitor):
            doc = io.load_document(path)
        self.assertEqual(set(opened), {"_rels/.rels", CONFIG_MEMBER, "3D/3dmodel.model"})
        md = doc.instances[0].metadata
        report = doc.metadata["slicer_metadata"]
        self.assertEqual(builder.source_build_blockers(md), [])
        self.assertTrue(report["assembly_pose_present_not_applied"])
        self.assertEqual(report["geometry_frame"], GEOMETRY_FRAME)
        self.assertEqual(md["geometry_frame"], GEOMETRY_FRAME)
        self.assertFalse(report["slicer_assembly_transform_applied"])
        self.assertFalse(md["slicer_private_transforms_applied"])
        np.testing.assert_allclose(doc.instances[0].mesh.bounds.mean(axis=0), [5, 6, 7])
        np.testing.assert_array_equal(md["components"][0]["source_to_world_mm"], translation(5, 6, 7))
        part = report["objects"][0]["parts"][0]
        self.assertIn("file:///outside/secret.step", [m["attributes"].get("value") for m in part["metadata"]])
        self.assertEqual(report["plates"][0]["core_build_instance_mapping"], "not_inferred_from_private_instance_ids")
        self.assertIn("assembly_pose_present_not_applied", md["slicer_plate_assembly_context"])

    def test_no_slicer_evidence_keeps_core_only_compatibility_but_not_semantic_recovery(self):
        doc = self.load(None, model_xml())
        self.assertEqual(doc.metadata["role_mapping_status"], "not_applicable")
        self.assertFalse(doc.metadata["material_semantics_complete"])
        self.assertFalse(doc.metadata["reconstruction_veto"])
        self.assertEqual(builder.source_build_blockers(doc.instances[0].metadata), [])
        builder.validate(self.plan(doc.instances[0].metadata))
        # Numeric IDs across model scopes are only a private-config ambiguity,
        # not an added restriction on legitimate core-only production models.
        xml = model_xml(box_object() + assembly(10, '<component objectid="1" p:path="child.model"/>'),
                        '<item objectid="10"/>')
        child = model_xml(box_object(), None)
        doc = self.load(None, xml, parts={"3D/child.model": child})
        self.assertEqual(builder.source_build_blockers(doc.instances[0].metadata), [])
        for legacy in ({}, {"reconstruction_eligible": True}, {"printable": True}):
            self.assertEqual(builder.source_build_blockers(legacy), [])

    def test_bambu_evidence_missing_config_is_veto_even_when_core_metadata_is_omitted(self):
        xml = self.model().replace('<resources>', '<metadata name="Application">BambuStudio-2.8</metadata><resources>')
        for limits in (io.DEFAULT_LIMITS, replace(io.DEFAULT_LIMITS, max_metadata_text_chars=1)):
            doc = self.load(None, xml, limits=limits)
            self.assert_veto(doc)
            self.assertEqual(doc.metadata["slicer_metadata"]["config_status"], "missing")
        xml = self.model(attributes='xmlns:b="http://schemas.bambulab.com/package/2021"')
        self.assert_veto(self.load(None, xml))
        # A known private config alone is also evidence requiring role config.
        self.assert_veto(self.load(None, parts={"Metadata/project_settings.config": b"not read"}))

    def test_unknown_private_configs_and_noncanonical_model_settings_are_disclosed_and_vetoed(self):
        for parts in ({"Metadata/future.config": b"{opaque json}"},
                      {"Metadata/MODEL_SETTINGS.CONFIG": "<config/>"},
                      {"elsewhere/model_settings.config": "<config/>"}):
            with self.subTest(parts=parts):
                doc = self.load(self.config(), parts=parts)
                self.assert_veto(doc)
                self.assertTrue(doc.metadata["slicer_metadata"]["unknown_private_config_members"])
                self.assertIsNone(doc.instances[0].metadata["components"][0]["slicer_role"])
                self.assertTrue(any(r["handling"] == "unknown_private_config_not_interpreted" for r in doc.metadata["slicer_metadata"]["private_config_members"]))
        doc = self.load(self.config(), parts={"Metadata/project_settings.config": b"not XML",
                                             "Metadata/filament_settings_1.config": b"uninterpreted"})
        self.assertEqual(builder.source_build_blockers(doc.instances[0].metadata), [])
        self.assertEqual(doc.metadata["statistics"]["xml_parts_parsed"], 3)

    def test_unknown_roots_elements_attributes_metadata_keys_and_nested_records_veto(self):
        configs = [self.config().replace('<config>', '<future>').replace('</config>', '</future>'),
                   self.config(extra='<unknown_shape/>'), self.config().replace('<config>', '<config version="future">'),
                   self.config().replace('<part id=', '<part path="https://invalid.example/source" id='),
                   self.config(metadata='<metadata key="future_boolean" value="union"/>'),
                   self.config(metadata='<metadata key="matrix" value="identity"><part id="2" subtype="negative_part"/></metadata>'),
                   self.config().replace('</part>', '<part id="2" subtype="negative_part"/></part>'),
                   self.config().replace('</object>', '<mesh_stat/></object>')]
        for config in configs:
            with self.subTest(config=config):
                self.assert_veto(self.load(config))

    def test_duplicate_plate_instances_and_assembly_identities_are_not_accepted(self):
        instance = ('<model_instance><metadata key="object_id" value="10"/>'
                    '<metadata key="instance_id" value="0"/></model_instance>')
        item = '<assemble_item object_id="10" instance_id="0" transform="opaque"/>'
        for extra in (f'<plate><metadata key="plater_id" value="1"/>{instance}{instance}</plate>',
                      f'<assemble>{item}{item}</assemble>', f'<assemble>{item}</assemble><assemble/>',
                      '<plate><metadata key="plater_id" value="1"/></plate><plate><metadata key="plater_id" value="1"/></plate>'):
            with self.subTest(extra=extra):
                self.assert_veto(self.load(self.config(extra=extra)))

    def test_config_security_is_fatal_and_uses_shared_package_xml_parser(self):
        attacks = ["<!DOCTYPE config [<!ENTITY x 'boom'>]><config>&x;</config>",
                   "<!DOCTYPE config SYSTEM 'file:///outside/secret'><config/>"]
        for xml in attacks:
            for encoding in ("utf-8", "utf-16"):
                with self.subTest(encoding=encoding, xml=xml), self.assertRaisesRegex(io.ThreeMFError, "DTD"):
                    self.load(xml.encode(encoding), geometry_only=True, isolate_build_errors=True)
        with self.assertRaisesRegex(io.ThreeMFError, "Invalid XML"):
            self.load(b"<config>", geometry_only=True, isolate_build_errors=True)
        # An alias cannot hide malicious XML behind the canonical member.
        with self.assertRaisesRegex(io.ThreeMFError, "DTD"):
            self.load(self.config(), parts={"Metadata/MODEL_SETTINGS.CONFIG": attacks[0]},
                      geometry_only=True, isolate_build_errors=True)
        with self.assertRaisesRegex(io.ThreeMFError, "Duplicate ZIP member"):
            io.load_document(self.package(config=self.config(), duplicate=True))

    def test_source_veto_stops_build_before_runtime_initialization(self):
        import test_build_validation as fixtures
        source = deepcopy(self.load(self.config("negative_part")).instances[0].metadata)
        source.update(reconstruction_blockers=[], reconstruction_eligible=True, reconstruction_veto=False,
                      material_semantics_complete=True, role_mapping_status="resolved")
        plan = self.plan(source)
        output = self.directory / "not-built.sldprt"
        with fixtures.MockBuildGuardsTests().fake_runtime(plan) as runtime:
            with self.assertRaisesRegex(ValueError, "source_provenance"):
                builder.build(plan, output)
            runtime.pythoncom.CoInitialize.assert_not_called()
            runtime.client.Dispatch.assert_not_called()
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".build.json").exists())

    def test_empty_private_instance_containers_are_charged_to_evidence_budget(self):
        extra = '<plate><metadata key="plater_id" value="1"/>' + '<model_instance/>' * 30 + '</plate>'
        limits = replace(io.DEFAULT_LIMITS, max_metadata_entries=12)
        doc = self.load(self.config(extra=extra), limits=limits)
        self.assert_veto(doc)
        report = doc.metadata["slicer_metadata"]
        self.assertEqual(report["evidence_summary"]["records_preserved"], 12)
        self.assertGreater(report["evidence_summary"]["records_omitted"], 0)
        self.assertLessEqual(len(report["plates"][0]["model_instances"]), 12)

    def test_unknown_content_summary_limits_are_explicit_not_silent(self):
        doc = self.load(self.config(extra='<future/><other/><nested/>'),
                        limits=replace(io.DEFAULT_LIMITS, max_ignored_content_summaries=1))
        self.assert_veto(doc)
        report = doc.metadata["slicer_metadata"]
        self.assertEqual(len(report["uninterpreted_content"]), 1)
        self.assertEqual(report["uninterpreted_content_summaries_omitted"], 2)

    def test_xml_byte_part_depth_and_element_budgets_include_private_config(self):
        config = self.config(extra=" " * 6000)
        path = self.package(config=config)
        with zipfile.ZipFile(path) as archive:
            sizes = {i.filename: i.file_size for i in archive.infolist()}
        xml_bytes = sum(sizes.values())
        limits = [
            replace(io.DEFAULT_LIMITS, max_xml_parts=2),
            replace(io.DEFAULT_LIMITS, max_xml_bytes=sizes[CONFIG_MEMBER] - 1),
            replace(io.DEFAULT_LIMITS, max_total_xml_bytes=xml_bytes - 1),
        ]
        for limit in limits:
            with self.subTest(limit=limit), self.assertRaisesRegex(io.ThreeMFError, "XML.*limit"):
                io.load_document(path, limits=limit, geometry_only=True, isolate_build_errors=True)
        core = self.load(None)
        count = core.metadata["statistics"]["xml_elements"]
        with self.assertRaisesRegex(io.ThreeMFError, "element count limit"):
            self.load(self.config(), limits=replace(io.DEFAULT_LIMITS, max_xml_elements=count), isolate_build_errors=True)
        deep = '<config>' + '<future>' * 64 + '</future>' * 64 + '</config>'
        with self.assertRaisesRegex(io.ThreeMFError, "depth limit"):
            self.load(deep, isolate_build_errors=True)
        parsed = []
        original = io._Package.xml

        def tracked(package, name):
            parsed.append(name)
            return original(package, name)

        with patch.object(io._Package, "xml", tracked):
            doc = self.load(self.config())
        self.assertIn(CONFIG_MEMBER, parsed)
        self.assertEqual(doc.metadata["statistics"]["xml_parts_parsed"], 3)

    def test_config_evidence_uses_shared_optional_metadata_budgets_and_fails_closed_on_loss(self):
        config = self.config(metadata='<metadata key="source_file" value="' + 'x' * 400 + '"/>')
        for limits in (replace(io.DEFAULT_LIMITS, max_metadata_entries=2),
                       replace(io.DEFAULT_LIMITS, max_metadata_text_chars=5),
                       replace(io.DEFAULT_LIMITS, max_metadata_total_chars=80),
                       replace(io.DEFAULT_LIMITS, max_metadata_attributes=1)):
            with self.subTest(limits=limits):
                doc = self.load(config, limits=limits)
                self.assert_veto(doc)
                summary = doc.metadata["slicer_metadata"]["evidence_summary"]
                self.assertFalse(summary["complete"])
                self.assertLessEqual(doc.metadata["source_metadata_summary"]["shared_budget_characters_preserved"], limits.max_metadata_total_chars)
                self.assertLessEqual(summary["records_preserved"] + doc.metadata["source_metadata_summary"]["entries_preserved"], limits.max_metadata_entries)
                json.dumps(doc.metadata, allow_nan=False)

    def test_config_never_relaxes_strict_core_extension_or_type_checks(self):
        for attributes in ('xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02" requiredextensions="m"',
                           'xmlns:x="urn:unknown" requiredextensions="x"'):
            xml = self.model(attributes=attributes)
            with self.subTest(attributes=attributes), self.assertRaisesRegex(io.ThreeMFError, "requiredextensions"):
                self.load(self.config(), xml)
        for kind in ("support", "other"):
            xml = self.model().replace('<object id="1"', f'<object type="{kind}" id="1"')
            with self.subTest(kind=kind), self.assertRaisesRegex(io.ThreeMFError, "non-model"):
                self.load(self.config(), xml)
            doc = self.load(self.config(), xml, geometry_only=True)
            self.assertTrue(doc.instances[0].metadata["reconstruction_veto"])
            self.assertIn(f"non_model_object_type:{kind}", builder.source_build_blockers(doc.instances[0].metadata))

    def test_skipped_or_failed_repeated_build_occurrences_cannot_hide_mapping_ambiguity(self):
        for extra in ('<item objectid="10" printable="false"/>',
                      '<item objectid="10" transform="singular"/>'):
            doc = self.load(self.config(), self.model('<item objectid="10"/>' + extra),
                            geometry_only=True, isolate_build_errors=True)
            self.assert_veto(doc)
            self.assertFalse(doc.metadata["document_coverage_complete"])
            self.assertEqual(doc.instances[0].metadata["slicer_role_evidence"]["root_build_occurrence_count"], 2)

    def test_new_source_bundle_is_strict_missing_or_malformed_fields_veto(self):
        good = self.load(self.config()).instances[0].metadata
        fields = ("slicer_metadata_version", "slicer_role_evidence", "role_mapping_status",
                  "material_semantics_complete", "material_semantics_scope", "reconstruction_veto",
                  "geometry_frame", "slicer_assembly_transform_applied", "slicer_private_transforms_applied",
                  "printed_material_geometry_recovered", "slicer_plate_assembly_context")
        for field in fields:
            for value in (None, [], {}, "", 0, 1):
                if field == "slicer_metadata_version" and type(value) is int and value == 1:
                    continue
                source = deepcopy(good)
                source[field] = value
                with self.subTest(field=field, value=value):
                    self.assertTrue(builder.source_build_blockers(source))
            source = deepcopy(good)
            del source[field]
            with self.subTest(deleted=field):
                self.assertTrue(builder.source_build_blockers(source))
        ev = good["slicer_role_evidence"]
        for field in ev:
            source = deepcopy(good)
            del source["slicer_role_evidence"][field]
            with self.subTest(evidence_deleted=field):
                self.assertTrue(builder.source_build_blockers(source))
        for value in (None, [], False, "normal_part"):
            source = deepcopy(good)
            source["slicer_role_evidence"]["parts"] = value
            self.assertTrue(builder.source_build_blockers(source))
        for value in (True, -1, 2, "0", None):
            source = deepcopy(good)
            source["slicer_role_evidence"]["leaves"][0]["config_part_index"] = value
            self.assertTrue(builder.source_build_blockers(source))
        for field in ("slicer_assembly_transform_applied", "slicer_private_transforms_applied", "printed_material_geometry_recovered"):
            source = deepcopy(good)
            source[field] = True
            self.assertTrue(builder.source_build_blockers(source))

    def test_locator_paths_and_core_numeric_ids_never_depend_on_metadata_interleaving(self):
        config = ('<config><plate><metadata key="plater_id" value="1"/></plate>'
                  '<object id="10"><metadata key="name" value="parent"/>'
                  '<part id="1" subtype="normal_part"><metadata key="name" value="first"/></part>'
                  '<metadata face_count="24"/><part id="2" subtype="normal_part"/></object></config>')
        xml = self.model(extra=box_object(2), children='<component objectid="1"/><component objectid="2"/>')
        for text in ("10", "010", "+10", " 10 "):
            with self.subTest(core_id=text):
                doc = self.load(config, xml.replace('<item objectid="10"/>', f'<item objectid="{text}"/>'))
                self.assertEqual(builder.source_build_blockers(doc.instances[0].metadata), [])
                root = ET.fromstring(config)
                for c in doc.instances[0].metadata["components"]:
                    loc = c["slicer_config_locator"]
                    self.assertEqual(root.find(loc.removeprefix("config/")).get("id"), str(c["object_id"]))
                for obj in doc.metadata["slicer_metadata"]["objects"]:
                    self.assertEqual(root.find(obj["source"]["source_config_locator"].removeprefix("config/")).get("id"), "10")

    def test_package_coverage_veto_is_separate_from_included_instance_role_mapping(self):
        xml = self.model('<item objectid="10"/><item objectid="20" printable="false"/>',
                         extra=box_object(2) + assembly(20, '<component objectid="2"/>'))
        config = ('<config><object id="10"><part id="1" subtype="normal_part"/></object>'
                  '<object id="20"><part id="2" subtype="negative_part"/></object></config>')
        doc = self.load(config, xml)
        self.assertTrue(doc.metadata["reconstruction_veto"])
        self.assertIn("document_coverage_complete=false", doc.metadata["reconstruction_blockers"])
        self.assertFalse(doc.metadata["material_semantics_complete"])
        self.assertTrue(doc.instances[0].metadata["material_semantics_complete"])
        self.assertTrue(doc.instances[0].metadata["reconstruction_veto"])
        self.assertFalse(doc.instances[0].metadata["document_coverage_complete"])
        self.assertEqual(doc.instances[0].metadata["role_mapping_status"], "resolved")
        source = deepcopy(doc.instances[0].metadata)
        source.update(reconstruction_blockers=[], reconstruction_veto=False, reconstruction_eligible=True)
        self.assertIn("document_coverage_complete=false", builder.source_build_blockers(source))

    def test_mutated_range_and_dropped_negative_evidence_cannot_masquerade_as_complete(self):
        good = self.load(self.config()).instances[0].metadata
        for mutation in (lambda s: s["slicer_role_evidence"]["leaves"][0].update(instance_face_range_half_open=[0, 11]),
                         lambda s: s["components"][0].update(face_end=11),
                         lambda s: s["components"][0].update(component_path=[]),
                         lambda s: s["slicer_role_evidence"].update(instance_face_count=13),
                         lambda s: s["slicer_role_evidence"].update(root_build_occurrence_count=2),
                         lambda s: s["slicer_role_evidence"].update(cross_model_id_conflicts=[1]),
                         lambda s: s["slicer_role_evidence"]["parts"][0]["source"]["attributes"].update(subtype="negative_part"),
                         lambda s: s["slicer_role_evidence"]["parts"][0]["source"].pop("text"),
                         lambda s: s["slicer_role_evidence"]["config_object_source"]["attributes"].update(future="boolean"),
                         lambda s: s["slicer_role_evidence"]["config_object_source"].update(text="unknown data")):
            source = deepcopy(good)
            mutation(source)
            self.assertTrue(builder.source_build_blockers(source))


if __name__ == "__main__":
    unittest.main()
