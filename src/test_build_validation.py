"""Offline plan/receipt tests. COM mocks test guard flow, not SolidWorks APIs."""
import builtins
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import build_solidworks as builder
from build_solidworks import offline_validation_receipt, plan_volume_mm3, validate


def plate(version=2):
    plan = {
        'schema_version': version,
        'strategy': 'review_required_plate_profile',
        'approved_for_build': True,
        'parameters': {'thickness_mm': 2.0},
        'outer_points_mm': [[0, 0], [20, 0], [20, 20], [0, 20]],
        'cuts': [
            {'id': 'hole_01', 'type': 'circle', 'center_mm': [5, 5], 'diameter_mm': 2},
            {'id': 'opening_02', 'type': 'polygon',
             'points_mm': [[12, 12], [14, 12], [14, 14], [12, 14]]},
        ],
    }
    if version == 2:
        plan.update(requires_approximation_acceptance=False, accepted_approximation=False,
                    build_blockers=[], source_placement_applied=False)
    return plan


def circle(cut_id, x, y, diameter=2):
    return {'id': cut_id, 'type': 'circle', 'center_mm': [x, y], 'diameter_mm': diameter}


def rectangle(cut_id, x1, y1, x2, y2):
    return {'id': cut_id, 'type': 'polygon',
            'points_mm': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]}


class ValidationTests(unittest.TestCase):
    def test_valid_v1_v2_return_none_without_mutation(self):
        for version in (1, 2):
            with self.subTest(version=version):
                plan = plate(version)
                snapshot = deepcopy(plan)
                self.assertIsNone(validate(plan))
                self.assertEqual(plan, snapshot)

    def test_no_cuts_is_valid(self):
        plan = plate()
        plan['cuts'] = []
        self.assertIsNone(validate(plan))
        self.assertEqual(plan_volume_mm3(plan), 800)

    def test_approval_gate(self):
        for version in (1, 2):
            for approval in (False, None, 0, 1, 'true', []):
                with self.subTest(version=version, approval=approval):
                    plan = plate(version)
                    plan['approved_for_build'] = approval
                    with self.assertRaises(ValueError):
                        validate(plan)
            plan = plate(version)
            del plan['approved_for_build']
            with self.assertRaises(ValueError):
                validate(plan)

    def test_approximation_acceptance_gate(self):
        plan = plate()
        plan['requires_approximation_acceptance'] = True
        with self.assertRaisesRegex(ValueError, 'accepted_approximation'):
            validate(plan)
        plan['accepted_approximation'] = True
        self.assertIsNone(validate(plan))
        plan['approved_for_build'] = False
        with self.assertRaises(ValueError):
            validate(plan)

    def test_blockers_never_overridden_by_approval_or_acceptance(self):
        for version in (1, 2):
            for approved in (False, True):
                for blocker in ('unsupported nesting', 'multiple solids', ''):
                    with self.subTest(version=version, approved=approved, blocker=blocker):
                        plan = plate(version)
                        plan.update(approved_for_build=approved, accepted_approximation=True,
                                    build_blockers=[blocker])
                        with self.assertRaisesRegex(ValueError, 'build_blockers'):
                            validate(plan)

    def test_v2_review_fields_required_and_strictly_typed(self):
        for field in ('requires_approximation_acceptance', 'accepted_approximation',
                      'source_placement_applied', 'build_blockers'):
            with self.subTest(field=field, value='missing'):
                plan = plate()
                del plan[field]
                with self.assertRaises(ValueError):
                    validate(plan)
            values = (None, False, {}, '', ()) if field == 'build_blockers' else (None, 0, 1, 'false', [])
            for value in values:
                with self.subTest(field=field, value=value):
                    plan = plate()
                    plan[field] = value
                    with self.assertRaises(ValueError):
                        validate(plan)
        plan = plate()
        plan['source_placement_applied'] = True
        with self.assertRaisesRegex(ValueError, 'placement'):
            validate(plan)

    def test_legacy_review_fields_are_optional_but_no_fake_placement(self):
        plan = plate(1)
        self.assertIsNone(validate(plan))
        plan['source_placement_applied'] = True
        with self.assertRaises(ValueError):
            validate(plan)

    def test_invalid_schema_and_strategy(self):
        for version in (None, 0, 3, -1, True, False, 1.0, 2.0, '2', [], {}):
            with self.subTest(version=version):
                plan = plate()
                plan['schema_version'] = version
                with self.assertRaises(ValueError):
                    validate(plan)
                with self.assertRaises(ValueError):
                    plan_volume_mm3(plan)
        plan = plate()
        del plan['schema_version']
        with self.assertRaises(ValueError):
            validate(plan)
        for strategy in (None, 'unknown', 1, [], {}):
            plan = plate()
            plan['strategy'] = strategy
            with self.assertRaises(ValueError):
                validate(plan)
        for plan in (None, [], 'plan', 42):
            with self.assertRaises(ValueError):
                validate(plan)

    def test_invalid_thickness_and_diameter(self):
        values = (-1, 0, True, False, '2', None, [], {}, math.nan,
                  math.inf, -math.inf, 10**1000)
        for field in ('thickness', 'diameter'):
            for value in values:
                with self.subTest(field=field, value=value):
                    plan = plate()
                    if field == 'thickness':
                        plan['parameters']['thickness_mm'] = value
                    else:
                        plan['cuts'][0]['diameter_mm'] = value
                    with self.assertRaises(ValueError):
                        validate(plan)
        for field in ('parameters', 'diameter_mm', 'thickness_mm'):
            plan = plate()
            if field == 'parameters':
                del plan['parameters']
            elif field == 'thickness_mm':
                del plan['parameters'][field]
            else:
                del plan['cuts'][0][field]
            with self.assertRaises(ValueError):
                validate(plan)
        plan = plate()
        plan['parameters']['thickness_mm'] = 1e308
        with self.assertRaises(ValueError):
            validate(plan)

    def test_invalid_coordinate_values_and_shapes(self):
        values = (True, False, None, '5', math.nan, math.inf, -math.inf, [], {}, 10**1000)
        for location in ('outer', 'polygon_cut', 'circle'):
            for value in values:
                with self.subTest(location=location, value=value):
                    plan = plate()
                    if location == 'outer':
                        plan['outer_points_mm'][0][0] = value
                    elif location == 'polygon_cut':
                        plan['cuts'][1]['points_mm'][0][0] = value
                    else:
                        plan['cuts'][0]['center_mm'][0] = value
                    with self.assertRaises(ValueError):
                        validate(plan)
        for value in (None, [], [1], [1, 2, 3], '12', {'x': 1, 'y': 2}):
            with self.subTest(point=value):
                plan = plate()
                plan['cuts'][0]['center_mm'] = value
                with self.assertRaises(ValueError):
                    validate(plan)
                plan = plate()
                plan['outer_points_mm'][0] = value
                with self.assertRaises(ValueError):
                    validate(plan)

    def test_missing_geometry_and_invalid_containers(self):
        for key in ('outer_points_mm', 'cuts', 'parameters'):
            for value in (None, 'bad', 42, {}):
                with self.subTest(key=key, value=value):
                    plan = plate()
                    plan[key] = value
                    with self.assertRaises(ValueError):
                        validate(plan)
            plan = plate()
            del plan[key]
            with self.assertRaises(ValueError):
                validate(plan)
        for cut in (None, 42, [], 'cut', {}):
            plan = plate()
            plan['cuts'] = [cut]
            with self.assertRaises(ValueError):
                validate(plan)
        plan = plate()
        del plan['cuts'][1]['points_mm']
        with self.assertRaises(ValueError):
            validate(plan)

    def test_unknown_cut_type(self):
        for cut_type in ('ellipse', 'Polygon', '', None, True, [], {}):
            with self.subTest(cut_type=cut_type):
                plan = plate()
                plan['cuts'][0]['type'] = cut_type
                with self.assertRaises(ValueError):
                    validate(plan)

    def test_empty_duplicate_or_nonstring_cut_id(self):
        for value in ('', ' ', '\t', None, 0, False, [], {}, ' hole ', 'hole_01'):
            with self.subTest(cut_id=value):
                plan = plate()
                plan['cuts'][1]['id'] = value
                with self.assertRaises(ValueError):
                    validate(plan)
        plan = plate()
        del plan['cuts'][0]['id']
        with self.assertRaises(ValueError):
            validate(plan)

    def test_self_intersecting_and_degenerate_outer_and_inner_polygons(self):
        rings = [
            [], [[2, 2]], [[2, 2], [4, 2]],
            [[2, 2], [4, 2], [6, 2]],  # collinear
            [[2, 2], [6, 6], [2, 6], [6, 2]],  # bow tie, zero signed area
            [[2, 2], [6, 6], [2, 6], [6, 2], [8, 2]],  # crossing, nonzero area
            [[2, 2], [6, 2], [6, 2], [2, 6]],  # zero-length segment
            [[2, 2], [6, 2], [6, 6], [2, 6], [2, 2]],  # repeated closing point
            [[2, 2], [6, 2], [4, 2], [4, 6], [2, 6]],  # overlapping edge
        ]
        for location in ('outer', 'cut'):
            for ring in rings:
                with self.subTest(location=location, ring=ring):
                    plan = plate()
                    if location == 'outer':
                        plan['outer_points_mm'] = ring
                        plan['cuts'] = []
                    else:
                        plan['cuts'] = [{'id': 'bad', 'type': 'polygon', 'points_mm': ring}]
                    with self.assertRaises(ValueError):
                        validate(plan)

    def test_clockwise_valid_ring_and_concave_outer(self):
        plan = plate()
        plan['outer_points_mm'].reverse()
        plan['cuts'][1]['points_mm'].reverse()
        self.assertIsNone(validate(plan))
        plan['outer_points_mm'] = [[0, 0], [20, 0], [20, 10], [10, 10], [10, 20], [0, 20]]
        plan['cuts'] = [circle('inside', 5, 5)]
        self.assertIsNone(validate(plan))
        plan['cuts'] = [circle('in_concavity', 14, 14)]
        with self.assertRaises(ValueError):
            validate(plan)

    def test_holes_outside_and_touching_outer(self):
        holes = [
            circle('outside', 25, 5), circle('partly_outside', .5, 5), circle('tangent', 1, 5),
            rectangle('outside', 22, 2, 24, 4), rectangle('partial', -1, 2, 2, 4),
            rectangle('edge', 0, 2, 2, 4),
            {'id': 'vertex_touch', 'type': 'polygon', 'points_mm': [[0, 5], [2, 4], [2, 6]]},
            rectangle('enclosing', -1, -1, 21, 21),
            rectangle('empty_cut', 0, 0, 20, 20),
            rectangle('split_plate', 9, 0, 11, 20),
        ]
        for hole in holes:
            with self.subTest(hole=hole):
                plan = plate()
                plan['cuts'] = [hole]
                with self.assertRaises(ValueError):
                    validate(plan)

    def test_intersection_overlap_containment_and_tangency_between_holes(self):
        pairs = [
            [circle('a', 5, 5), circle('b', 6, 5)],
            [circle('a', 5, 5), circle('b', 7, 5)],
            [circle('a', 5, 5, 6), circle('b', 5, 5)],
            [circle('a', 5, 5), circle('b', 5, 5)],
            [rectangle('a', 3, 3, 6, 6), rectangle('b', 5, 5, 8, 8)],
            [rectangle('a', 3, 3, 6, 6), rectangle('b', 6, 3, 8, 6)],
            [rectangle('a', 3, 3, 6, 6), rectangle('b', 6, 6, 8, 8)],
            [rectangle('a', 3, 3, 8, 8), rectangle('b', 4, 4, 5, 5)],
            [circle('a', 5, 5, 6), rectangle('b', 4, 4, 6, 6)],
            [rectangle('a', 3, 3, 8, 8), circle('b', 5, 5)],
            [circle('a', 5, 5), rectangle('b', 5.5, 4, 7, 6)],
            [circle('a', 5, 5), rectangle('b', 6, 4, 8, 6)],
        ]
        for pair in pairs:
            with self.subTest(pair=pair):
                plan = plate()
                plan['cuts'] = pair
                with self.assertRaises(ValueError):
                    validate(plan)

    def test_circle_chord_error_is_conservative_at_oblique_outer_edge(self):
        # Tangency lies halfway between buffer vertices, not at x/y extrema.
        theta = math.pi / (4 * builder._CIRCLE_QUAD_SEGS)
        n = (math.cos(theta), math.sin(theta))
        tangent = (-n[1], n[0])
        for signed_gap in (-1e-6, 0):
            plan = plate()
            lower = -1 - signed_gap
            plan['outer_points_mm'] = [
                [u*n[0] + v*tangent[0], u*n[1] + v*tangent[1]]
                for u, v in ((lower, -5), (5, -5), (5, 5), (lower, 5))
            ]
            plan['cuts'] = [circle('oblique', 0, 0)]
            with self.subTest(signed_gap=signed_gap), self.assertRaises(ValueError):
                validate(plan)
        plan['outer_points_mm'] = [
            [u*n[0] + v*tangent[0], u*n[1] + v*tangent[1]]
            for u, v in ((-1.001, -5), (5, -5), (5, 5), (-1.001, 5))
        ]
        self.assertIsNone(validate(plan))

    def test_circle_chord_error_is_conservative_for_oblique_circle_pairs(self):
        theta = math.pi / (4 * builder._CIRCLE_QUAD_SEGS)
        for gap in (-1e-6, 0):
            plan = plate()
            distance = 2 + gap
            plan['cuts'] = [circle('a', 5, 5), circle('b', 5 + distance*math.cos(theta),
                                                       5 + distance*math.sin(theta))]
            with self.subTest(gap=gap), self.assertRaises(ValueError):
                validate(plan)
        plan['cuts'][1]['center_mm'] = [5 + 2.001*math.cos(theta), 5 + 2.001*math.sin(theta)]
        self.assertIsNone(validate(plan))

    def test_subchord_positive_clearance_is_rejected_if_unresolved(self):
        # At buffer vertices, true clearance can be positive yet smaller than
        # the conservative error budget. Such near-touches must not pass.
        plan = plate()
        plan['cuts'] = [circle('near_outer', 1.000001, 5)]
        with self.assertRaises(ValueError):
            validate(plan)
        plan['cuts'] = [circle('a', 5, 5), circle('b', 7.000001, 5)]
        with self.assertRaises(ValueError):
            validate(plan)

    def test_circle_chord_error_against_oblique_polygon_cut(self):
        theta = math.pi / (4 * builder._CIRCLE_QUAD_SEGS)
        n = (math.cos(theta), math.sin(theta))
        tangent = (-n[1], n[0])
        points = [
            [5 + u*n[0] + v*tangent[0], 5 + u*n[1] + v*tangent[1]]
            for u, v in ((1, -1), (3, -1), (3, 1), (1, 1))
        ]
        plan = plate()
        plan['cuts'] = [circle('a', 5, 5), {'id': 'b', 'type': 'polygon', 'points_mm': points}]
        with self.assertRaises(ValueError):
            validate(plan)

    def test_final_difference_must_be_one_nonempty_polygon(self):
        # Defensive guard after Boolean subtraction, independently of the
        # containment and pairwise checks which normally prevent these cases.
        from shapely.geometry import MultiPolygon, Polygon
        empty = Polygon()
        one = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        two = Polygon([[2, 0], [3, 0], [3, 1], [2, 1]])
        for result in (empty, MultiPolygon([one, two])):
            with self.subTest(result=result.geom_type):
                with patch.object(Polygon, 'difference', return_value=result):
                    with self.assertRaises(ValueError):
                        validate(plate())

    def test_volume_is_analytic_current_parameters_not_source_estimate(self):
        for version in (1, 2):
            plan = plate(version)
            plan.update(approved_for_build=False, accepted_approximation=False,
                        requires_approximation_acceptance=True,
                        build_blockers=['unsupported nesting'],
                        source_volume_mm3=123, estimated_reconstruction_volume_mm3=456)
            expected = (400 - math.pi - 4) * 2
            self.assertAlmostEqual(plan_volume_mm3(plan), expected, places=10)
            with self.assertRaises(ValueError):
                validate(plan)
            plan['parameters']['thickness_mm'] = 3
            self.assertAlmostEqual(plan_volume_mm3(plan), expected * 1.5, places=10)
            plan['cuts'][0]['diameter_mm'] = 4
            self.assertAlmostEqual(plan_volume_mm3(plan), (400 - 4*math.pi - 4) * 3, places=10)

    def test_volume_still_rejects_invalid_geometry_without_review(self):
        plan = plate()
        plan['approved_for_build'] = False
        plan['cuts'][0]['center_mm'] = [-5, 5]
        with self.assertRaises(ValueError):
            plan_volume_mm3(plan)


class ReceiptAndCliTests(unittest.TestCase):
    def test_offline_receipt_claims_and_hash(self):
        plan = plate()
        receipt = offline_validation_receipt(plan)
        canonical = json.dumps(plan, sort_keys=True, separators=(',', ':'),
                               ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.assertEqual(receipt['plan_sha256'], hashlib.sha256(canonical).hexdigest())
        self.assertEqual(receipt['schema_version'], 2)
        self.assertEqual(receipt['receipt_type'], 'offline_validation')
        self.assertFalse(receipt['solidworks_runtime_verified'])
        self.assertIsNone(receipt['actual_solidworks_revision'])
        self.assertFalse(receipt['source_placement_applied'])
        self.assertIn('local', receipt['output_coordinates'])
        self.assertEqual(receipt['plan_volume_scope'], 'current_plan_parameters_only')
        self.assertFalse(receipt['source_geometry_equivalence_verified'])
        self.assertNotIn('output', receipt)
        self.assertEqual(receipt['warnings'], [])
        reordered = dict(reversed(list(plan.items())))
        self.assertEqual(offline_validation_receipt(reordered)['plan_sha256'], receipt['plan_sha256'])
        plan['parameters']['thickness_mm'] = 3
        self.assertNotEqual(offline_validation_receipt(plan)['plan_sha256'], receipt['plan_sha256'])

    def test_legacy_warning_is_explicit(self):
        receipt = offline_validation_receipt(plate(1))
        self.assertEqual(receipt['schema_version'], 1)
        self.assertIn('Legacy schema v1', receipt['warnings'][0])

    def test_receipt_rejects_nonfinite_metadata(self):
        plan = plate()
        plan['source_volume_mm3'] = math.nan
        with self.assertRaises(ValueError):
            offline_validation_receipt(plan)

    def test_validate_only_cli_no_output_artifact_or_pywin32_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plan.json'
            path.write_text(json.dumps(plate()), encoding='utf-8')
            original_import = builtins.__import__

            def no_pywin32(name, *args, **kwargs):
                if name == 'pythoncom' or name.startswith('win32com'):
                    raise AssertionError('Offline validation attempted pywin32 import')
                return original_import(name, *args, **kwargs)

            output = io.StringIO()
            with patch('builtins.__import__', side_effect=no_pywin32), redirect_stdout(output):
                builder.main([str(path), '--validate-only'])
            receipt = json.loads(output.getvalue())
            self.assertFalse(receipt['solidworks_runtime_verified'])
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ['plan.json'])
            completed = subprocess.run([sys.executable, str(Path(builder.__file__)), str(path), '--validate-only'],
                                       capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(completed.stdout)['plan_sha256'], receipt['plan_sha256'])

    def test_cli_requires_output_for_build_and_rejects_ignored_output(self):
        with redirect_stdout(io.StringIO()), patch('sys.stderr', new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                builder.main(['plan.json'])
            with self.assertRaises(SystemExit):
                builder.main(['plan.json', '--validate-only', '--output', 'ignored.sldprt'])

    def test_nonwindows_build_does_not_import_pywin32(self):
        with tempfile.TemporaryDirectory() as directory:
            original_import = builtins.__import__

            def no_pywin32(name, *args, **kwargs):
                if name == 'pythoncom' or name.startswith('win32com'):
                    raise AssertionError('Non-Windows build attempted pywin32 import')
                return original_import(name, *args, **kwargs)

            with patch.object(builder.sys, 'platform', 'linux'), patch('builtins.__import__', side_effect=no_pywin32):
                with self.assertRaisesRegex(RuntimeError, 'requires Windows'):
                    builder.build(plate(), Path(directory) / 'new.sldprt')
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_existing_part_and_receipt_and_wrong_suffix_are_rejected_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'new.sldprt'
            receipt = output.with_suffix('.build.json')
            for existing in (output, receipt):
                existing.write_bytes(b'keep this content')
                with self.assertRaises(FileExistsError):
                    builder.build(plate(), output)
                self.assertEqual(existing.read_bytes(), b'keep this content')
                existing.unlink()
            with self.assertRaises(ValueError):
                builder.build(plate(), Path(directory) / 'new.step')

    def test_broken_final_symlink_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'new.sldprt'
            output.symlink_to(Path(directory) / 'missing.sldprt')
            with self.assertRaises(FileExistsError):
                builder.build(plate(), output)
            self.assertTrue(output.is_symlink())
            output.unlink()
            receipt = output.with_suffix('.build.json')
            receipt.symlink_to(Path(directory) / 'missing.json')
            with self.assertRaises(FileExistsError):
                builder.build(plate(), output)
            self.assertTrue(receipt.is_symlink())


class MockBuildGuardsTests(unittest.TestCase):
    """Guard/lifecycle tests only. These mocks do not prove COM signatures work."""
    @contextmanager
    def fake_runtime(self, plan, revision='32.0.0', save_mode='ok', body_volume=None, phase_volumes=None):
        pythoncom = ModuleType('pythoncom')
        pythoncom.CoInitialize = Mock()
        pythoncom.CoUninitialize = Mock()
        pythoncom.VT_BYREF = 0x4000
        pythoncom.VT_I4 = 3
        client = ModuleType('win32com.client')
        client.VARIANT = lambda kind, value: SimpleNamespace(value=value)
        win32com = ModuleType('win32com')
        win32com.client = client
        plane = SimpleNamespace(GetTypeName2=Mock(return_value='RefPlane'), Select2=Mock(return_value=True))
        sketch_feature = SimpleNamespace(Name=None, Select2=Mock(return_value=True))
        sm = SimpleNamespace(
            ActiveSketch=SimpleNamespace(GetFeature=Mock(return_value=sketch_feature)),
            InsertSketch=Mock(), CreateLine=Mock(return_value=object()),
            CreateCircleByRadius=Mock(return_value=object()),
        )
        # Model feature side effects, not the SolidWorks geometry engine. Each
        # rebuild/mass read must see the volume of the CURRENT base/cut phase.
        from shapely.geometry import Polygon
        thickness = plan['parameters']['thickness_mm']
        base_volume = Polygon(plan['outer_points_mm']).area * thickness
        normal_volumes = [base_volume]
        for cut in plan['cuts']:
            area = (math.pi * (cut['diameter_mm'] / 2) ** 2 if cut['type'] == 'circle'
                    else Polygon(cut['points_mm']).area)
            normal_volumes.append(normal_volumes[-1] - area * thickness)
        if phase_volumes is not None and body_volume is not None:
            raise ValueError('Use phase_volumes or body_volume, not both')
        volumes = list(normal_volumes if phase_volumes is None else phase_volumes)
        if len(volumes) != len(normal_volumes):
            raise ValueError('phase_volumes needs one base entry and one per cut')
        if body_volume is not None:
            volumes[-1] = body_volume
        state = SimpleNamespace(phase=-1)

        def feature_side_effect():
            state.phase += 1
            return SimpleNamespace(Name=None)

        fm = SimpleNamespace(FeatureExtrusion2=Mock(side_effect=lambda *args: feature_side_effect()),
                             FeatureCut3=Mock(side_effect=lambda *args: feature_side_effect()))

        def mass_properties(density):
            if state.phase < 0:
                raise AssertionError('Read mass before extrusion')
            return [0, 0, 0, volumes[state.phase] / 1e9]

        body = SimpleNamespace(GetMassProperties=Mock(side_effect=mass_properties))

        def save_as(path, version, options, data, errors, warnings):
            if save_mode == 'raise':
                raise RuntimeError('simulated SaveAs exception')
            if save_mode not in ('missing', 'false'):
                Path(path).write_bytes(b'' if save_mode == 'empty' else b'mocked part, NOT a real SLDPRT')
            if save_mode == 'errors':
                errors.value = 1
            return save_mode != 'false'

        doc = SimpleNamespace(
            FirstFeature=Mock(return_value=plane), SketchManager=sm, FeatureManager=fm,
            ClearSelection2=Mock(), ForceRebuild3=Mock(return_value=True),
            GetBodies2=Mock(return_value=[body]), ViewZoomtofit2=Mock(),
            Extension=SimpleNamespace(SaveAs=Mock(side_effect=save_as)),
        )
        app = SimpleNamespace(
            RevisionNumber=Mock(return_value=revision), Visible=False,
            GetUserPreferenceStringValue=Mock(return_value='part.prtdot'),
            NewDocument=Mock(return_value=doc),
            CloseDoc=Mock(side_effect=AssertionError('Must not close any user document')),
            ActivateDoc3=Mock(side_effect=AssertionError('Must not activate an old document')),
        )
        client.Dispatch = Mock(return_value=app)
        with patch.dict(sys.modules, {'pythoncom': pythoncom, 'win32com': win32com, 'win32com.client': client}), \
                patch.object(builder.sys, 'platform', 'win32'), redirect_stdout(io.StringIO()):
            yield SimpleNamespace(pythoncom=pythoncom, client=client, app=app, doc=doc,
                                  fm=fm, sm=sm, body=body, state=state, phase_volumes=volumes)

    def test_reject_wrong_or_unreadable_revision_before_new_document(self):
        for revision in ('31.5.0', '33.0.0', '', None, 32, '32.bad', 'unknown'):
            with self.subTest(revision=revision), tempfile.TemporaryDirectory() as directory:
                with self.fake_runtime(plate(), revision=revision) as runtime:
                    with self.assertRaisesRegex(RuntimeError, 'major revision 32'):
                        builder.build(plate(), Path(directory) / 'new.sldprt')
                    runtime.app.NewDocument.assert_not_called()
                    runtime.pythoncom.CoUninitialize.assert_called_once_with()
                    self.assertEqual(list(Path(directory).iterdir()), [])

    def test_success_receipt_only_after_save_and_lifecycle_cleanup(self):
        for version in (1, 2):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                plan = plate(version)
                # Circle cuts must not accidentally use an irrelevant polygon field.
                plan['cuts'][0]['points_mm'] = [[1, 1], [2, 1], [1, 2]]
                output = Path(directory) / 'new.sldprt'
                with self.fake_runtime(plan, revision='32.3.1') as runtime:
                    builder.build(plan, output)
                    runtime.pythoncom.CoInitialize.assert_called_once_with()
                    runtime.pythoncom.CoUninitialize.assert_called_once_with()
                    runtime.app.CloseDoc.assert_not_called()
                    runtime.app.ActivateDoc3.assert_not_called()
                    runtime.sm.CreateCircleByRadius.assert_called_once()
                    self.assertEqual(runtime.sm.CreateLine.call_count, 8)
                    self.assertEqual(runtime.doc.ForceRebuild3.call_count, 3)
                    self.assertEqual(runtime.doc.GetBodies2.call_count, 3)
                    self.assertEqual(runtime.body.GetMassProperties.call_count, 3)
                    # Existing COM signature/direction is NOT certified by this
                    # mock, but must not change as part of the volume safeguards.
                    for call in runtime.fm.FeatureCut3.call_args_list:
                        self.assertEqual(call.args, (
                            True, False, True, 0, 0, .003, .003,
                            False, False, False, False, 0, 0, False, False, False, False,
                            False, True, True, False, False, False, 0, 0, False))
                receipt = json.loads(output.with_suffix('.build.json').read_text(encoding='utf-8'))
                self.assertEqual(receipt['actual_solidworks_revision'], '32.3.1')
                self.assertEqual(receipt['schema_version'], version)
                self.assertEqual(receipt['plan_volume_mm3'], plan_volume_mm3(plan))
                self.assertFalse(receipt['source_geometry_equivalence_verified'])
                self.assertFalse(receipt['source_placement_applied'])
                self.assertEqual(receipt['plan_sha256'], offline_validation_receipt(plan)['plan_sha256'])
                checks = receipt['feature_volume_checks']
                self.assertEqual(receipt['feature_volume_checks_scope'], 'current_plan_parameters_only')
                self.assertEqual([check['feature'] for check in checks],
                                 ['Plate_Thickness', 'hole_01', 'opening_02'])
                self.assertEqual([check['type'] for check in checks], ['base_extrusion', 'cut', 'cut'])
                for check, expected in zip(checks, (800, 2 * math.pi, 8)):
                    self.assertEqual(check['scope'], 'current_plan_parameters_only')
                    self.assertEqual(check['status'], 'passed')
                    self.assertEqual(check['solid_bodies'], 1)
                    self.assertAlmostEqual(check['actual_volume_mm3'], expected)
                    self.assertAlmostEqual(check['expected_volume_mm3'], expected)
                    self.assertEqual(check['relative_tolerance'], .002)
                    self.assertLessEqual(check['numerical_bound_mm3'], check['allowed_error_mm3'])
                self.assertEqual(receipt['final_volume_check']['status'], 'passed')
                if version == 1:
                    self.assertIn('Legacy schema v1', receipt['warnings'][0])

    def test_dispatch_and_revision_exceptions_always_uninitialize(self):
        for stage in ('dispatch', 'revision'):
            with tempfile.TemporaryDirectory() as directory, self.fake_runtime(plate()) as runtime:
                method = runtime.client.Dispatch if stage == 'dispatch' else runtime.app.RevisionNumber
                method.side_effect = RuntimeError('simulated failure')
                with self.assertRaisesRegex(RuntimeError, 'simulated failure'):
                    builder.build(plate(), Path(directory) / 'new.sldprt')
                runtime.pythoncom.CoUninitialize.assert_called_once_with()
                runtime.app.NewDocument.assert_not_called()

    def test_save_result_errors_existence_and_nonempty_checks(self):
        for mode in ('false', 'errors', 'missing', 'empty', 'raise'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'new.sldprt'
                with self.fake_runtime(plate(), save_mode=mode) as runtime:
                    with self.assertRaises(RuntimeError):
                        builder.build(plate(), output)
                    runtime.pythoncom.CoUninitialize.assert_called_once_with()
                    runtime.app.CloseDoc.assert_not_called()
                self.assertFalse(output.with_suffix('.build.json').exists())

    def test_nonfinite_or_mismatched_actual_volume_does_not_save(self):
        for volume in (0, -1, math.nan, math.inf, 800):
            with self.subTest(volume=volume), tempfile.TemporaryDirectory() as directory:
                with self.fake_runtime(plate(), body_volume=volume) as runtime:
                    with self.assertRaises((ValueError, RuntimeError)):
                        builder.build(plate(), Path(directory) / 'new.sldprt')
                    runtime.doc.Extension.SaveAs.assert_not_called()
                    runtime.pythoncom.CoUninitialize.assert_called_once_with()

    def test_late_receipt_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'new.sldprt'
            receipt_path = output.with_suffix('.build.json')
            with self.fake_runtime(plate()) as runtime:
                original_save = runtime.doc.Extension.SaveAs.side_effect

                def save_with_late_receipt(*args):
                    result = original_save(*args)
                    receipt_path.write_text('preserve existing receipt', encoding='utf-8')
                    return result

                runtime.doc.Extension.SaveAs.side_effect = save_with_late_receipt
                with self.assertRaises(FileExistsError):
                    builder.build(plate(), output)
                runtime.pythoncom.CoUninitialize.assert_called_once_with()
            self.assertEqual(receipt_path.read_text(encoding='utf-8'), 'preserve existing receipt')
            self.assertGreater(output.stat().st_size, 0)

    def test_late_part_or_receipt_before_save_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'new.sldprt'
            with self.fake_runtime(plate()) as runtime:
                runtime.doc.ViewZoomtofit2.side_effect = lambda: output.write_bytes(b'keep late user part')
                with self.assertRaises(FileExistsError):
                    builder.build(plate(), output)
                runtime.doc.Extension.SaveAs.assert_not_called()
                runtime.pythoncom.CoUninitialize.assert_called_once_with()
            self.assertEqual(output.read_bytes(), b'keep late user part')


if __name__ == '__main__':
    unittest.main()
