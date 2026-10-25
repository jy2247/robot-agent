from copy import deepcopy
import unittest
from robot_agent.evaluate import evaluate


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.labels = [{'episode_id': 'e', 'frame_id': '1', 'phase': 'a', 'status': 'completed'},
                       {'episode_id': 'e', 'frame_id': '2', 'phase': 'b', 'status': 'failed'}]
        self.rows = [{**r, 'backend': 'mock', 'backend_error_type': None,
                      'metrics': {'agent_latency_ms': i * 100 + 10, 'over_budget': bool(i)}}
                     for i, r in enumerate(self.labels)]

    def test_metrics_and_wrong_phase_completion(self):
        result = evaluate(self.rows, self.labels, ('a', 'b'))
        self.assertEqual(result['phase_macro_f1'], 1)
        self.assertEqual(result['latency_ms'], {'p50': 10, 'p95': 110})
        self.assertEqual(result['deadline_miss_rate'], 0.5)
        self.rows[0]['phase'] = 'b'
        result = evaluate(self.rows, self.labels, ('a', 'b'))
        self.assertEqual(result['completion']['fp'], 1)
        self.assertEqual(result['completion']['fn'], 1)
        self.assertEqual(result['completion']['tp'], 0)

    def test_bad_joins_and_latency(self):
        for rows in ([], self.rows[:1], self.rows + self.rows[:1]):
            with self.assertRaises(ValueError):
                evaluate(rows, self.labels, ('a', 'b'))
        rows = deepcopy(self.rows)
        rows[0]['metrics']['agent_latency_ms'] = float('nan')
        with self.assertRaises(ValueError):
            evaluate(rows, self.labels, ('a', 'b'))
