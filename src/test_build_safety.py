"""Source/volume fail-closed tests; COM mocks exercise flow, NOT runtime certification."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

import build_solidworks as builder
import test_build_validation as fixtures


class SourceBuildSafetyTests(unittest.TestCase):
    def test_source_vetoes_are_pure_strings_even_with_eligible_true(self):
        cases = (
            ({'reconstruction_eligible': False}, 'reconstruction_eligible=false'),
            ({'reconstruction_blockers': ['unsupported component']}, 'unsupported component'),
            ({'core_spec_violations': ['forbidden core reference']}, 'forbidden core reference'),
            ({'printable': False}, 'non_printable_build_item'),
            ({'effective_object_types': ['model', 'support']}, 'non_model_object_type:support'),
            ({'effective_object_type': 'other'}, 'non_model_object_type:other'),
            ({'document_coverage_complete': False}, 'document_coverage_complete=false'),
        )
        for fields, reason in cases:
            provenance = {'reconstruction_eligible': True, **fields}
            snapshot = deepcopy(provenance)
            with self.subTest(fields=fields):
                blockers = builder.source_build_blockers(provenance)
                self.assertIsInstance(blockers, list)
                self.assertTrue(all(isinstance(value, str) for value in blockers))
                self.assertIn(reason, blockers)
                self.assertEqual(provenance, snapshot)

    def test_all_evidence_survives_clearing_copied_top_level_blockers(self):
        source = {
            'reconstruction_eligible': False,
            'reconstruction_blockers': ['non_model_object_type:support'],
            'core_spec_violations': ['core reference is forbidden'],
            'printable': False, 'effective_object_type': 'support',
            'effective_object_types': ['model', 'support'],
            'document_coverage_complete': False,
        }
        # Delete the explicit source list, and then even the eligibility flag.
        # The remaining original evidence must still veto the reviewed plan.
        for deleted in ((), ('reconstruction_blockers',),
                        ('reconstruction_blockers', 'reconstruction_eligible', 'core_spec_violations')):
            provenance = {key: value for key, value in source.items() if key not in deleted}
            for version in (1, 2):
                with self.subTest(version=version, deleted=deleted):
                    plan = fixtures.plate(version)
                    plan.update(build_blockers=[], approved_for_build=True, accepted_approximation=True,
                                reconstruction_eligible=True, reconstruction_blockers=[],
                                core_spec_violations=[], source_provenance=provenance)
                    snapshot = deepcopy(plan)
                    with self.assertRaisesRegex(ValueError, 'source_provenance'):
                        builder.validate(plan)
                    with self.assertRaisesRegex(ValueError, 'source_provenance'):
                        builder.offline_validation_receipt(plan)
                    # The geometric estimate deliberately remains independent
                    # of review authorization and source equivalence.
                    self.assertGreater(builder.plan_volume_mm3(plan), 0)
                    self.assertEqual(plan, snapshot)

    def test_each_independent_source_veto_reaches_the_review_gate(self):
        sources = (
            {'reconstruction_eligible': False},
            {'reconstruction_blockers': ['do not build']},
            {'core_spec_violations': ['do not build']},
            {'printable': False},
            {'effective_object_types': ['model', 'solidsupport']},
            {'effective_object_type': 'surface'},
            {'document_coverage_complete': False},
        )
        for source in sources:
            for version in (1, 2):
                with self.subTest(source=source, version=version):
                    plan = fixtures.plate(version)
                    plan.update(build_blockers=[], source_provenance=source,
                                approved_for_build=True, accepted_approximation=True)
                    with self.assertRaisesRegex(ValueError, 'source_provenance'):
                        builder.validate(plan)

    def test_missing_fields_keep_legacy_and_synthetic_plans_compatible(self):
        good_sources = (
            {}, {'build_index': 0, 'unrelated_metadata': None},
            {'reconstruction_eligible': True, 'reconstruction_blockers': [],
             'core_spec_violations': [], 'printable': True,
             'effective_object_type': 'model', 'effective_object_types': ['model'],
             'document_coverage_complete': True},
        )
        for source in good_sources:
            self.assertEqual(builder.source_build_blockers(source), [])
            for version in (1, 2):
                with self.subTest(source=source, version=version):
                    plan = fixtures.plate(version)
                    builder.validate(plan)  # Completely absent source_provenance.
                    plan['source_provenance'] = source
                    builder.validate(plan)

    def test_present_source_object_and_each_present_field_are_strictly_typed(self):
        bad_sources = [None, [], (), '', False, 0]
        for field in ('reconstruction_eligible', 'printable', 'document_coverage_complete'):
            bad_sources.extend({field: value} for value in (None, 0, 1, 'false', [], {}))
        for field in ('reconstruction_blockers', 'core_spec_violations', 'effective_object_types'):
            bad_sources.extend({field: value} for value in (None, False, 0, 'model', {}, ()))
            bad_sources.extend({field: [value]} for value in (None, False, 0, '', ' ', ' model ', [], {}))
        bad_sources.extend({'effective_object_type': value}
                           for value in (None, False, 0, '', ' ', ' model ', [], {}))
        for source in bad_sources:
            with self.subTest(source=source):
                self.assertTrue(builder.source_build_blockers(source))
                for version in (1, 2):
                    plan = fixtures.plate(version)
                    plan['source_provenance'] = source
                    with self.assertRaisesRegex(ValueError, 'source_provenance'):
                        builder.validate(plan)

    def test_source_blocked_build_stops_before_com_and_preserves_existing_files(self):
        plan = fixtures.plate()
        plan['source_provenance'] = {'printable': False}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'existing.sldprt'
            receipt = output.with_suffix('.build.json')
            output.write_bytes(b'preserve part')
            receipt.write_bytes(b'preserve receipt')
            with fixtures.MockBuildGuardsTests().fake_runtime(plan) as runtime:
                with self.assertRaisesRegex(ValueError, 'source_provenance'):
                    builder.build(plan, output)
                runtime.pythoncom.CoInitialize.assert_not_called()
                runtime.client.Dispatch.assert_not_called()
            self.assertEqual(output.read_bytes(), b'preserve part')
            self.assertEqual(receipt.read_bytes(), b'preserve receipt')


class BuiltVolumeSafetyTests(unittest.TestCase):
    def test_relative_tolerance_has_no_small_part_absolute_floor(self):
        for expected in (1e-300, 1e-20, .01, 1, 1e20, 1e300):
            for fraction in (0, .0019, -.0019):
                with self.subTest(expected=expected, fraction=fraction):
                    check = builder.validate_built_volume(expected * (1 + fraction), expected)
                    self.assertEqual(check['scope'], 'current_plan_parameters_only')
                    self.assertEqual(check['allowed_error_mm3'], expected * .002)
            for fraction in (.0021, -.0021, 1):
                with self.subTest(expected=expected, fraction=fraction):
                    with self.assertRaisesRegex(RuntimeError, 'Volume validation failed'):
                        builder.validate_built_volume(expected * (1 + fraction), expected)

    def test_small_plate_entire_hole_omission_fails_public_check(self):
        expected = (1 - math.pi * .4 ** 2) * .1
        self.assertLess(abs(.1 - expected), .1)  # The old absolute floor would pass.
        with self.assertRaisesRegex(RuntimeError, 'micro plate'):
            builder.validate_built_volume(.1, expected, label='micro plate')

    def test_invalid_or_unresolvable_quantities_never_pass(self):
        for value in (None, True, False, '', [], {}, math.nan, math.inf, -math.inf, 0, -1):
            for field in ('actual', 'expected'):
                with self.subTest(field=field, value=value):
                    with self.assertRaises((ValueError, RuntimeError)):
                        builder.validate_built_volume(value if field == 'actual' else 1,
                                                      value if field == 'expected' else 1)
        for expected in (math.ulp(0.0), 1e-322, 1e-320):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(RuntimeError, 'unable to resolve'):
                    builder.validate_built_volume(expected, expected)


class FeatureVolumeSafetyTests(unittest.TestCase):
    """No mocked output from this class constitutes a real SolidWorks receipt."""
    fake_runtime = fixtures.MockBuildGuardsTests.fake_runtime

    def assert_rejected_flow(self, plan, phases, message='Volume validation'):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'new.sldprt'
            with self.fake_runtime(plan, phase_volumes=phases) as runtime:
                with self.assertRaisesRegex((ValueError, RuntimeError), message):
                    builder.build(plan, output)
                runtime.doc.Extension.SaveAs.assert_not_called()
                runtime.pythoncom.CoUninitialize.assert_called_once_with()
                runtime.app.CloseDoc.assert_not_called()
                runtime.app.ActivateDoc3.assert_not_called()
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix('.build.json').exists())

    def test_correct_small_plate_base_and_cut_checks_are_recorded(self):
        plan = fixtures.plate()
        plan['outer_points_mm'] = [[0, 0], [1, 0], [1, 1], [0, 1]]
        plan['parameters']['thickness_mm'] = .1
        plan['cuts'] = [fixtures.circle('micro_hole', .5, .5, .8)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'mock_only.sldprt'
            with self.fake_runtime(plan) as runtime:
                builder.build(plan, output)
                self.assertEqual(runtime.doc.ForceRebuild3.call_count, 2)
                self.assertEqual(runtime.body.GetMassProperties.call_count, 2)
                runtime.pythoncom.CoUninitialize.assert_called_once_with()
                runtime.app.CloseDoc.assert_not_called()
            receipt = json.loads(output.with_suffix('.build.json').read_text())
            base, cut = receipt['feature_volume_checks']
            self.assertAlmostEqual(base['actual_volume_mm3'], .1)
            self.assertAlmostEqual(cut['actual_removed_volume_mm3'], math.pi * .4 ** 2 * .1)
            self.assertAlmostEqual(cut['expected_removed_volume_mm3'], math.pi * .4 ** 2 * .1)
            self.assertFalse(receipt['source_geometry_equivalence_verified'])
            self.assertEqual(receipt['feature_volume_checks_scope'], 'current_plan_parameters_only')
            self.assertIn('not source-geometry equivalence', receipt['feature_volume_checks_note'])

    def test_small_plate_whole_hole_missed_or_partial_is_rejected(self):
        plan = fixtures.plate()
        plan['outer_points_mm'] = [[0, 0], [1, 0], [1, 1], [0, 1]]
        plan['parameters']['thickness_mm'] = .1
        plan['cuts'] = [fixtures.circle('micro_hole', .5, .5, .8)]
        removed = math.pi * .4 ** 2 * .1
        for fraction in (0, .5, .9, 1.1, -1):
            with self.subTest(fraction=fraction):
                self.assert_rejected_flow(plan, [.1, .1 - removed * fraction], 'micro_hole removed volume')

    def test_small_missed_or_partial_cut_fails_even_when_net_volume_would_pass(self):
        plan = fixtures.plate()
        plan['cuts'][1] = fixtures.circle('tiny_hole', 12, 12, .01)
        before = 800 - 2 * math.pi
        tiny_cut = math.pi * .005 ** 2 * 2
        for fraction in (0, .5, .99, 1.01):
            final_volume = before - tiny_cut * fraction
            # Demonstrate the danger of checking only the final body volume.
            builder.validate_built_volume(final_volume, builder.plan_volume_mm3(plan))
            with self.subTest(fraction=fraction):
                self.assert_rejected_flow(plan, [800, before, final_volume], 'tiny_hole removed volume')

    def test_compensating_later_cut_cannot_hide_first_missed_cut(self):
        plan = fixtures.plate()
        plan['cuts'] = [fixtures.circle('tiny_a', 5, 5, .01), fixtures.circle('tiny_b', 12, 12, .01)]
        correct_final = builder.plan_volume_mm3(plan)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'new.sldprt'
            with self.fake_runtime(plan, phase_volumes=[800, 800, correct_final]) as runtime:
                with self.assertRaisesRegex(RuntimeError, 'tiny_a removed volume'):
                    builder.build(plan, output)
                self.assertEqual(runtime.fm.FeatureCut3.call_count, 1)
                runtime.doc.Extension.SaveAs.assert_not_called()
                runtime.pythoncom.CoUninitialize.assert_called_once_with()
                runtime.app.CloseDoc.assert_not_called()

    def test_correct_tiny_cut_can_pass_when_delta_is_resolvable(self):
        plan = fixtures.plate()
        plan['cuts'] = [fixtures.circle('tiny', 5, 5, .01)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'mock_only.sldprt'
            with self.fake_runtime(plan):
                builder.build(plan, output)
            receipt = json.loads(output.with_suffix('.build.json').read_text())
            check = receipt['feature_volume_checks'][1]
            self.assertGreater(check['actual_removed_volume_mm3'], 0)
            self.assertLessEqual(check['numerical_bound_mm3'], check['allowed_error_mm3'])

    def test_insufficient_before_volume_precision_fails_even_with_correct_net_volume(self):
        plan = fixtures.plate()
        plan['outer_points_mm'] = [[0, 0], [1e8, 0], [1e8, 1e8], [0, 1e8]]
        plan['parameters']['thickness_mm'] = 1
        plan['cuts'] = [fixtures.circle('unresolvable', 5, 5)]
        expected = builder.plan_volume_mm3(plan)
        builder.validate_built_volume(expected, expected)  # Net volume looks perfect.
        self.assert_rejected_flow(plan, [1e16, expected], 'unable to resolve.*unresolvable')

    def test_native_cubic_metre_underflow_is_not_certified_after_mm3_conversion(self):
        plan = fixtures.plate()
        plan['cuts'] = []
        plan['parameters']['thickness_mm'] = 1e-314
        expected = builder.plan_volume_mm3(plan)
        builder.validate_built_volume(expected, expected)  # mm3 itself has adequate precision.
        self.assert_rejected_flow(plan, [expected], 'unable to resolve')

    def test_small_base_error_inside_relative_tolerance_is_recorded(self):
        plan = fixtures.plate()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'new.sldprt'
            with self.fake_runtime(plan, phase_volumes=[799, 799 - 2 * math.pi, 791 - 2 * math.pi]):
                # 0.125% base error and the final net error fit their individual
                # tolerances; receipt preserves the nonzero error, not equivalence.
                builder.build(plan, output)
            receipt = json.loads(output.with_suffix('.build.json').read_text())
            self.assertAlmostEqual(receipt['feature_volume_checks'][0]['absolute_error_mm3'], 1)
            self.assertFalse(receipt['source_geometry_equivalence_verified'])

    def test_final_net_volume_is_checked_even_when_all_feature_checks_pass(self):
        plan = fixtures.plate()
        base_volume = 801.59  # Base error < 800*.002, but > net-plan-volume*.002.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'new.sldprt'
            with self.fake_runtime(plan, phase_volumes=[base_volume, base_volume - 2 * math.pi,
                                                       base_volume - 2 * math.pi - 8]) as runtime:
                with self.assertRaisesRegex(RuntimeError, 'Final solid body volume'):
                    builder.build(plan, output)
                self.assertEqual(runtime.fm.FeatureCut3.call_count, 2)
                runtime.doc.Extension.SaveAs.assert_not_called()
                runtime.pythoncom.CoUninitialize.assert_called_once_with()
                runtime.app.CloseDoc.assert_not_called()
            self.assertFalse(output.with_suffix('.build.json').exists())

    def test_base_extrusion_wrong_volume_is_rejected_before_any_cut(self):
        plan = fixtures.plate()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'new.sldprt'
            with self.fake_runtime(plan, phase_volumes=[400, 400 - 2 * math.pi, 392 - 2 * math.pi]) as runtime:
                with self.assertRaisesRegex(RuntimeError, 'Plate_Thickness'):
                    builder.build(plan, output)
                runtime.fm.FeatureCut3.assert_not_called()
                runtime.doc.Extension.SaveAs.assert_not_called()
                runtime.pythoncom.CoUninitialize.assert_called_once_with()
                runtime.app.CloseDoc.assert_not_called()

    def test_exactly_one_body_and_successful_rebuild_required_at_every_phase(self):
        plan = fixtures.plate()
        for phase in range(3):
            for failure in ('body_none', 'body_zero', 'body_two', 'rebuild', 'mass_none', 'mass_short'):
                with self.subTest(phase=phase, failure=failure), tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / 'new.sldprt'
                    with self.fake_runtime(plan) as runtime:
                        if failure.startswith('body_'):
                            invalid = {'body_none': None, 'body_zero': [],
                                       'body_two': [runtime.body, runtime.body]}[failure]
                            runtime.doc.GetBodies2.side_effect = [[runtime.body]] * phase + [invalid]
                        elif failure == 'rebuild':
                            runtime.doc.ForceRebuild3.side_effect = [True] * phase + [False]
                        else:
                            original = runtime.body.GetMassProperties.side_effect

                            def wrong_mass(density):
                                if runtime.state.phase == phase:
                                    return None if failure == 'mass_none' else []
                                return original(density)

                            runtime.body.GetMassProperties.side_effect = wrong_mass
                        with self.assertRaises(RuntimeError):
                            builder.build(plan, output)
                        runtime.doc.Extension.SaveAs.assert_not_called()
                        runtime.pythoncom.CoUninitialize.assert_called_once_with()
                        runtime.app.CloseDoc.assert_not_called()
                        runtime.app.ActivateDoc3.assert_not_called()
                    self.assertFalse(output.with_suffix('.build.json').exists())

    def test_com_feature_and_volume_exceptions_always_uninitialize_without_closing(self):
        for stage in ('extrusion', 'cut', 'rebuild', 'bodies', 'mass'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                with self.fake_runtime(fixtures.plate()) as runtime:
                    method = {
                        'extrusion': runtime.fm.FeatureExtrusion2, 'cut': runtime.fm.FeatureCut3,
                        'rebuild': runtime.doc.ForceRebuild3, 'bodies': runtime.doc.GetBodies2,
                        'mass': runtime.body.GetMassProperties,
                    }[stage]
                    method.side_effect = RuntimeError('simulated COM failure')
                    with self.assertRaisesRegex(RuntimeError, 'simulated COM failure'):
                        builder.build(fixtures.plate(), Path(directory) / 'new.sldprt')
                    runtime.doc.Extension.SaveAs.assert_not_called()
                    runtime.pythoncom.CoUninitialize.assert_called_once_with()
                    runtime.app.CloseDoc.assert_not_called()
                    runtime.app.ActivateDoc3.assert_not_called()


if __name__ == '__main__':
    unittest.main()
