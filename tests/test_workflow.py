import unittest

from robot_agent.workflow import WorkflowTracker


class WorkflowTests(unittest.TestCase):
    def test_ordered_episode_transitions(self):
        tracker = WorkflowTracker(('pick', 'place'))
        first = tracker.apply('e1', {'phase': 'pick', 'decision': 'success',
                                     'phase_completion_confirmed': True})
        self.assertTrue(first['accepted'])
        self.assertEqual(first['next_phase'], 'place')
        second = tracker.apply('e2', {'phase': 'place', 'decision': 'success',
                                      'phase_completion_confirmed': True})
        self.assertTrue(second['workflow_completed'])

    def test_retry_budget_and_out_of_order_phase(self):
        tracker = WorkflowTracker(('pick', 'place'), max_retries=1)
        self.assertEqual(tracker.apply('e1', {'phase': 'pick', 'decision': 'retry'})['action'], 'retry')
        self.assertEqual(tracker.apply('e2', {'phase': 'pick', 'decision': 'retry'})['action'], 'fail')
        tracker = WorkflowTracker(('pick', 'place'))
        result = tracker.apply('e3', {'phase': 'place', 'decision': 'success',
                                      'phase_completion_confirmed': True})
        self.assertEqual(result['action'], 'replan')
        self.assertFalse(result['accepted'])


if __name__ == '__main__':
    unittest.main()
