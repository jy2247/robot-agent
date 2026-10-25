class WorkflowTracker:
    """Tracks accepted phase outcomes across isolated episodes; stores no observations."""

    def __init__(self, phases, max_retries=2, start_phase=None):
        self.phases = tuple(phases)
        if not self.phases or len(set(self.phases)) != len(self.phases):
            raise ValueError('phases must be unique and nonempty')
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError('max_retries must be a nonnegative integer')
        if start_phase is not None and start_phase not in self.phases:
            raise ValueError('start_phase must be configured')
        self.index = self.phases.index(start_phase) if start_phase else 0
        self.max_retries = max_retries
        self.retry_count = 0
        self.episodes = set()

    def apply(self, episode_id, result):
        if not isinstance(episode_id, str) or not episode_id or episode_id in self.episodes:
            raise ValueError('episode_id must be new and nonempty')
        self.episodes.add(episode_id)
        expected = self.phases[self.index] if self.index < len(self.phases) else None
        observed, action = result.get('phase'), result.get('decision', 'review')
        accepted, reason = False, action

        if expected is None:
            action, reason = 'fail', 'workflow_already_completed'
        elif observed != expected:
            action, reason = 'replan', 'unexpected_phase'
        elif action == 'success':
            if result.get('phase_completion_confirmed'):
                accepted = True
                self.index += 1
                self.retry_count = 0
                reason = 'phase_completion_accepted'
            else:
                action, reason = 'continue', 'awaiting_completion_confirmation'
        elif action == 'retry':
            self.retry_count += 1
            if self.retry_count > self.max_retries:
                action, reason = 'fail', 'retry_budget_exhausted'

        next_phase = self.phases[self.index] if self.index < len(self.phases) else None
        return {'episode_id': episode_id, 'expected_phase': expected, 'observed_phase': observed,
                'accepted': accepted, 'action': action, 'reason': reason,
                'retry_count': self.retry_count, 'next_phase': next_phase,
                'workflow_completed': next_phase is None}
