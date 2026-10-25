"""Offline frame-level metrics; labels and predictions joined by episode/frame ID."""
import math


def evaluate(predictions, labels, phases):
    def index(rows):
        out = {}
        for row in rows:
            key = (row['episode_id'], row['frame_id'])
            if key in out:
                raise ValueError(f'duplicate frame: {key}')
            out[key] = row
        if not out:
            raise ValueError('evaluation inputs must not be empty')
        return out

    pred, truth = index(predictions), index(labels)
    if pred.keys() != truth.keys():
        raise ValueError('prediction and label frame IDs must match exactly')
    names = (*phases, 'unknown')
    matrix = {a: {b: 0 for b in names} for a in names}
    binary = {name: {'tp': 0, 'fp': 0, 'fn': 0} for name in ('completion', 'failure')}
    latencies, misses, failures = [], 0, 0
    for key, target in truth.items():
        row = pred[key]
        if target['phase'] not in names or row['phase'] not in names:
            raise ValueError('unknown phase label')
        if target['status'] not in ('in_progress', 'completed', 'failed', 'uncertain'):
            raise ValueError('invalid status label')
        matrix[target['phase']][row['phase']] += 1
        for name, status in (('completion', 'completed'), ('failure', 'failed')):
            # Completion is correct only for the labeled phase, not any completed verdict.
            actual = row['status'] == status
            expected = target['status'] == status
            correct = expected and (name == 'failure' or row['phase'] == target['phase'])
            binary[name]['tp'] += int(actual and correct)
            binary[name]['fp'] += int(actual and not correct)
            binary[name]['fn'] += int(expected and not (actual and correct))
        latency = row['metrics'].get('end_to_end_latency_ms', row['metrics']['agent_latency_ms'])
        if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
            raise ValueError('latency must be finite and nonnegative')
        latencies.append(latency)
        misses += bool(row['metrics']['over_budget'])
        failures += bool(row.get('backend_error_type'))
    f1s = []
    for phase in phases:
        tp = matrix[phase][phase]
        fp = sum(matrix[x][phase] for x in names if x != phase)
        fn = sum(matrix[phase][x] for x in names if x != phase)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    for counts in binary.values():
        tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
        counts['precision'] = tp / (tp + fp) if tp + fp else None
        counts['recall'] = tp / (tp + fn) if tp + fn else None
    latencies.sort()
    return {'frames': len(pred), 'backends': sorted({r['backend'] for r in pred.values()}),
            'phase_accuracy': sum(matrix[x][x] for x in names) / len(pred),
            'phase_macro_f1': sum(f1s) / len(f1s), 'confusion_matrix': matrix,
            **binary, 'latency_ms': {f'p{p}': latencies[math.ceil(p / 100 * len(latencies)) - 1]
                                      for p in (50, 95)},
            'mean_latency_ms': sum(latencies) / len(latencies),
            'effective_hz': 1000 / (sum(latencies) / len(latencies)),
            'deadline_miss_rate': misses / len(pred), 'backend_error_rate': failures / len(pred)}
