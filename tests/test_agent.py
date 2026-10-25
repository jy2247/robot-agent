from dataclasses import replace
import json
from pathlib import Path
import unittest

from robot_agent.agent import ObserverAgent
from robot_agent.backends.mock import MockBackend
from robot_agent.config import load_config
from robot_agent.context import messages

ROOT = Path(__file__).resolve().parents[1]


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(str(ROOT / 'configs/default.toml'))
        self.agent = ObserverAgent(self.config, MockBackend())
        self.clock = 0

    def observe(self, phase, status='completed', episode='test', error=None,
                end_of_episode=False):
        self.clock += 1
        note = f'phase={phase} status={status}' + (f' error={error}' if error else '')
        return self.agent.observe({'episode_id': episode, 'frame_id': str(self.clock),
                                   'timestamp_s': self.clock, 'note': note,
                                   'end_of_episode': end_of_episode})

    def test_all_phases_need_confirmation(self):
        for phase in self.config.phases:
            self.assertFalse(self.observe(phase)['task_completed'])
            result = self.observe(phase)
            self.assertTrue(result['phase_completion_confirmed'])
            self.assertEqual(result['task_completed'], phase == self.config.phases[-1])
        self.assertEqual(result['recommendation'], 'task_completed')

    def test_failure_clears_phase_confirmation_count(self):
        self.observe('approach')
        result = self.observe('approach', 'failed')
        self.assertEqual(result['recommendation'], 'retry_phase')
        self.assertEqual(result['decision'], 'retry')
        self.assertFalse(self.observe('approach')['phase_completion_confirmed'])

    def test_failure_recoverability_drives_decision(self):
        for error, decision in (('misaligned', 'retry'), ('object_dropped', 'replan'),
                                ('unsafe_motion', 'fail')):
            with self.subTest(error=error):
                self.agent.reset()
                result = self.observe('approach', 'failed', error=error)
                self.assertEqual(result['decision'], decision)

    def test_final_phase_alone_does_not_finish_task(self):
        self.observe('release_retract')
        result = self.observe('release_retract')
        self.assertFalse(result['task_completed'])
        self.assertEqual(result['completed_phases'], ['release_retract'])

    def test_episode_isolation_and_bounded_history(self):
        for _ in range(10):
            self.observe('approach', 'in_progress')
        self.assertEqual(len(self.agent.history), self.config.history_size)
        with self.assertRaises(ValueError):
            self.observe('approach', episode='another')
        self.agent.reset()
        self.assertEqual(len(self.agent.history), 0)
        self.observe('approach', episode='another')

    def test_out_of_order_frames_rejected_without_mutation(self):
        self.observe('approach')
        with self.assertRaises(ValueError):
            self.agent.observe({'episode_id': 'test', 'frame_id': 'old', 'timestamp_s': 0})
        self.assertEqual(len(self.agent.history), 1)

    def test_bad_backend_outputs_are_uncertain(self):
        class Broken:
            def __init__(self, raw):
                self.raw = raw
            def generate(self, messages):
                if isinstance(self.raw, Exception):
                    raise self.raw
                return self.raw
        for bad in ('not json', '{"phase":"invented"}', TimeoutError(), json.dumps({
                'phase': 'approach', 'status': 'completed', 'confidence': float('nan'),
                'errors': [], 'evidence': ['f1']})):
            with self.subTest(bad=repr(bad)):
                self.agent = ObserverAgent(self.config, Broken(bad))
                result = self.observe('approach')
                self.assertEqual(result['status'], 'uncertain')
                self.assertFalse(result['task_completed'])

    def test_low_confidence_cannot_confirm(self):
        self.agent = ObserverAgent(replace(self.config, confidence_threshold=0.99), MockBackend())
        first = self.observe('approach', 'in_progress')
        second = self.observe('approach', 'in_progress')
        third = self.observe('approach', 'in_progress')
        self.assertEqual((first['decision'], second['decision']), ('continue', 'continue'))
        self.assertTrue(first['confidence_grace_active'])
        self.assertEqual(third['decision'], 'review')
        self.assertFalse(self.observe('approach')['phase_completion_confirmed'])

    def test_next_phase_cannot_confirm_one_completed_observation(self):
        first = self.observe('approach')
        self.assertIsNone(first['confirmed_phase'])
        transition = self.observe('grasp', 'in_progress')
        self.assertFalse(transition['transition_completion_confirmed'])
        self.assertIsNone(transition['confirmed_phase'])
        self.assertEqual(transition['completed_phases'], [])
        self.assertEqual(transition['decision'], 'continue')

    def test_confirmations_survive_phase_switches_motion_and_occlusion(self):
        self.observe('approach')
        self.observe('grasp')
        self.observe('approach', 'in_progress')
        self.observe('unknown', 'uncertain')
        result = self.observe('approach')
        self.assertEqual(result['confirmed_phase'], 'approach')
        self.assertEqual(result['phase_confirmation_counts']['grasp'], 1)
        self.assertEqual(self.observe('grasp')['confirmed_phase'], 'grasp')

    def test_failure_only_clears_affected_phase_and_reset_clears_all(self):
        self.observe('approach')
        self.observe('grasp')
        result = self.observe('approach', 'failed')
        self.assertEqual(result['phase_confirmation_counts']['approach'], 0)
        self.assertEqual(result['phase_confirmation_counts']['grasp'], 1)
        self.assertTrue(self.observe('grasp')['phase_completion_confirmed'])
        self.agent.reset()
        result = self.observe('grasp')
        self.assertFalse(result['phase_completion_confirmed'])

    def test_backend_error_does_not_erase_prior_completion_evidence(self):
        self.observe('approach')
        class Broken:
            def generate(self, messages):
                raise TimeoutError()
        self.agent.backend = Broken()
        self.assertEqual(self.observe('unknown')['phase_confirmation_counts']['approach'], 1)
        self.agent.backend = MockBackend()
        self.assertTrue(self.observe('approach')['phase_completion_confirmed'])

    def test_end_of_episode_can_confirm_final_phase_once(self):
        for phase in self.config.phases[:-1]:
            self.observe(phase)
            self.observe(phase)
        result = self.observe(self.config.phases[-1], end_of_episode=True)
        self.assertTrue(result['end_of_episode_confirmation'])
        self.assertTrue(result['task_completed'])
        self.assertEqual(result['decision'], 'success')

    def test_end_of_episode_does_not_confirm_intermediate_phase(self):
        result = self.observe(self.config.phases[0], end_of_episode=True)
        self.assertFalse(result['end_of_episode_confirmation'])
        self.assertFalse(result['phase_completion_confirmed'])

    def test_failure_bypasses_confidence_grace(self):
        class LowConfidenceFailure:
            def generate(self, messages):
                return json.dumps({'phase': 'approach', 'status': 'failed',
                                   'confidence': .5, 'errors': ['misaligned'],
                                   'evidence': ['f1: visible miss']})
        self.agent = ObserverAgent(self.config, LowConfidenceFailure())
        result = self.observe('approach')
        self.assertEqual(result['decision'], 'review')
        self.assertFalse(result['confidence_grace_active'])

    def test_later_phase_is_recorded_despite_missing_earlier_phase(self):
        phase = self.config.phases[-2]
        self.observe(phase)
        result = self.observe(phase)
        self.assertEqual(result['completed_phases'], [phase])
        result = self.observe(self.config.phases[-1], 'uncertain', end_of_episode=True)
        self.assertEqual(result['completed_phases'], [phase])
        self.assertEqual(result['overall_reason'], 'INCOMPLETE_AT_END')
        self.assertEqual(result['decision'], 'fail')
        self.assertEqual(result['status'], 'uncertain')
        self.assertEqual(result['errors'], [])
        with self.assertRaises(ValueError):
            self.observe(phase)

    def test_confirmed_task_survives_occlusion_but_not_contradiction(self):
        for phase in self.config.phases:
            self.observe(phase)
            self.observe(phase)
        self.assertTrue(self.observe('unknown', 'uncertain')['task_completed'])
        result = self.observe(self.config.phases[-1], 'failed', error='object_dropped')
        self.assertFalse(result['task_completed'])
        result = self.observe('unknown', 'uncertain', end_of_episode=True)
        self.assertEqual(result['overall_reason'], 'PHASE_FAILED')
        self.assertEqual(result['decision'], 'fail')

    def test_terminal_success_uses_memory(self):
        for phase in self.config.phases:
            self.observe(phase)
            self.observe(phase)
        result = self.observe('unknown', 'uncertain', end_of_episode=True)
        self.assertEqual(result['overall_status'], 'success')
        self.assertEqual(result['decision'], 'success')

    def test_terminal_alias_and_validation(self):
        payload = {'episode_id': 'test', 'frame_id': '1', 'timestamp_s': 1,
                   'is_terminal': True}
        for bad in ({**payload, 'is_terminal': 'true'},
                    {**payload, 'end_of_episode': False}):
            with self.assertRaises(ValueError):
                self.agent.observe(bad)
        result = self.agent.observe(payload)
        self.assertTrue(result['is_terminal'])
        self.assertEqual(result['overall_reason'], 'INCOMPLETE_AT_END')

    def test_backend_error_at_end_still_terminates(self):
        class Broken:
            def generate(self, messages):
                raise TimeoutError()
        self.agent = ObserverAgent(self.config, Broken())
        result = self.observe('unknown', end_of_episode=True)
        self.assertEqual(result['status'], 'uncertain')
        self.assertEqual(result['overall_reason'], 'INCOMPLETE_AT_END')
        self.assertEqual(result['decision'], 'fail')

    def test_failure_can_be_reconfirmed_and_reset_clears_memory(self):
        phase = self.config.phases[0]
        self.observe(phase, 'failed', error='object_dropped')
        self.observe(phase)
        result = self.observe(phase)
        self.assertEqual(result['failed_phases'], [])
        self.assertEqual(result['completed_phases'], [phase])
        self.agent.reset()
        result = self.observe('unknown', 'uncertain', end_of_episode=True)
        self.assertEqual(result['completed_phases'], [])
        self.assertEqual(result['overall_reason'], 'INCOMPLETE_AT_END')

    def test_controller_memory_is_separate_from_visual_observations(self):
        phase = self.config.phases[0]
        current = {'episode_id': 'test', 'frame_id': 'f1', 'timestamp_s': 1}
        content = messages(self.config, [], current, None, {phase})[-1]['content']
        self.assertEqual(json.loads(content[0]['text'])['frame_id'], 'f1')
        self.assertIn('Controller memory (not new visual evidence)', content[-1]['text'])
        self.assertIn(phase, content[-1]['text'])

    def test_pipette_configs_share_criteria_and_keep_phase_ids(self):
        configs = [load_config(str(ROOT / f'configs/{name}.toml')) for name in
                   ('g1_pipette', 'g1_pipette_fast', 'vega_pipette')]
        for config in configs[1:]:
            self.assertEqual(config.phases, configs[0].phases)
            self.assertEqual(config.criteria, configs[0].criteria)
            self.assertEqual(config.failure_criteria, configs[0].failure_criteria)

    def test_initial_scene_reference_survives_history_and_resets(self):
        first = {'episode_id': 'test', 'frame_id': 'initial', 'timestamp_s': 0,
                 'images': ['data:image/png;base64,aA==', 'data:image/png;base64,aQ==']}
        self.agent.observe(first)
        for _ in range(self.config.history_size + 1):
            self.observe('approach', 'in_progress')
        content = messages(self.config, self.agent.history,
                           {'frame_id': 'next'}, None, reference=self.agent.reference)[-1]['content']
        self.assertEqual(json.loads(content[0]['text'])['temporal_role'], 'initial_reference')
        self.assertEqual(content[1]['image_url']['url'], first['images'][0])
        self.assertEqual(len(self.agent.reference['images']), 1)
        self.agent.reset()
        self.assertIsNone(self.agent.reference)


if __name__ == '__main__':
    unittest.main()
