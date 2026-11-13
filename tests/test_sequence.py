import unittest

from robot_agent.sequence import MonotonicEvidenceDecoder


def result(statuses, confidence=0.9):
    return {
        "votes": [{"verdict": {"status": status, "confidence": confidence}}
                  for status in statuses],
        "backend_error_type": None,
        "status": "in_progress",
        "confidence": 0.5,
    }


class SequenceDecoderTests(unittest.TestCase):
    def test_soft_prior_blocks_early_transition_and_order_is_monotonic(self):
        decoder = MonotonicEvidenceDecoder(["p1", "p2"], 0.4)
        transition, _, _ = decoder.update(
            1.6, result(["success"] * 3 + ["in_progress"] * 2))
        self.assertIsNone(transition)
        self.assertEqual(decoder.phase, "p1")

        transition, _, _ = decoder.update(
            4.4, result(["success"] * 3 + ["in_progress"] * 2))
        self.assertEqual(transition["to_phase"], "p2")
        self.assertEqual(decoder.phase, "p2")

    def test_final_phase_is_not_blocked_by_duration_prior(self):
        decoder = MonotonicEvidenceDecoder(["p1"], 0.4)
        transition, _, _ = decoder.update(
            0.4, result(["success"] * 3 + ["in_progress"] * 2))
        self.assertIsNotNone(transition)
        self.assertTrue(decoder.finished)
        self.assertEqual(decoder.completed, ["p1"])

        transition, _, _ = decoder.update(0.8, result(["success"] * 5))
        self.assertIsNone(transition)
        self.assertEqual(decoder.completed, ["p1"])
