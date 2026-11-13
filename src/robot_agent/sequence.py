import math


class MonotonicEvidenceDecoder:
    """Decode ordered phase transitions from leaky historical vote evidence."""

    def __init__(self, phases, cadence_s, soft_duration_center_s=4.0,
                 soft_duration_scale_s=0.8, decay=0.65, threshold=0.45,
                 use_duration_prior=True):
        self.phases = phases
        self.cadence_s = cadence_s
        self.soft_duration_center_s = soft_duration_center_s
        self.soft_duration_scale_s = soft_duration_scale_s
        self.decay = decay
        self.threshold = threshold
        self.use_duration_prior = use_duration_prior
        self.phase_index = 0
        self.phase_started_s = 0.0
        self.score = 0.0
        self.last_low_evidence_s = None
        self.first_positive_s = None
        self.completed = []
        self.finished = False

    @property
    def phase(self):
        return self.phases[self.phase_index]

    def _masses(self, result):
        masses = {"success": 0.0, "fail": 0.0, "in_progress": 0.0}
        votes = result.get("votes") or []
        for vote in votes:
            verdict = vote.get("verdict")
            if verdict:
                masses[verdict["status"]] += verdict["confidence"] / 5
        if not votes and result["backend_error_type"] is None:
            masses[result["status"]] = result["confidence"]
        return masses

    def update(self, timestamp_s, result):
        if self.finished:
            return None, self._masses(result), self.score
        masses = self._masses(result)
        evidence = (masses["success"] - 0.05 * masses["in_progress"]
                    - 0.20 * masses["fail"])
        self.score = max(0.0, self.decay * self.score + evidence)
        elapsed_s = timestamp_s - self.phase_started_s
        # Earlier transitions need a soft anti-cascade duration prior. The final
        # phase has no downstream phase to corrupt, so terminal success is exempt.
        duration_prior = (1.0 if (not self.use_duration_prior
                                  or self.phase_index == len(self.phases) - 1) else
                          1 / (1 + math.exp(-(
                              elapsed_s - self.soft_duration_center_s)
                              / self.soft_duration_scale_s)))
        posterior_score = self.score * duration_prior

        positive = masses["success"] >= 0.45
        if masses["success"] < 0.25:
            self.last_low_evidence_s = timestamp_s
            if posterior_score < self.threshold * 0.35:
                self.first_positive_s = None
        if positive and self.first_positive_s is None:
            self.first_positive_s = timestamp_s

        transition = None
        if positive and posterior_score >= self.threshold:
            evidence_time_s = self.first_positive_s or timestamp_s
            lower_s = (evidence_time_s - self.cadence_s
                       if self.last_low_evidence_s is None
                       else self.last_low_evidence_s)
            old_phase = self.phase
            self.completed.append(old_phase)
            transition = {
                "from_phase": old_phase,
                "to_phase": (self.phases[self.phase_index + 1]
                             if self.phase_index < len(self.phases) - 1 else None),
                "accepted_time_s": timestamp_s,
                "first_evidence_time_s": evidence_time_s,
                "last_nonterminal_time_s": self.last_low_evidence_s,
                "bracket_midpoint_time_s": (lower_s + evidence_time_s) / 2,
                "success_vote_mass": masses["success"],
                "fail_vote_mass": masses["fail"],
                "historical_evidence_score": self.score,
                "soft_duration_prior": duration_prior,
                "posterior_score": posterior_score,
            }
            if self.phase_index < len(self.phases) - 1:
                self.phase_index += 1
                self.phase_started_s = timestamp_s
                self.score = 0.0
                self.last_low_evidence_s = timestamp_s
                self.first_positive_s = None
            else:
                self.finished = True
        return transition, masses, posterior_score
