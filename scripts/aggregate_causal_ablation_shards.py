"""Aggregate independently collected five-chain causal-ablation shards."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def mean(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return statistics.mean(values) if values else None


def aggregate_variant(name, parts):
    chains = [chain for part in parts for chain in part["chains"]]
    for chain in chains:
        chain["chain_index"] = (chain["episode_range"][0] - 237) // 5 + 1
    chains.sort(key=lambda chain: chain["chain_index"])
    boundaries = [boundary for chain in chains for boundary in chain["boundaries"]]
    total = 4 * len(chains)
    result = {
        "name": name,
        "transition_count_detected": len(boundaries),
        "transition_count_total": total,
        "transition_recall": len(boundaries) / total,
        "boundary_mae_s_conditional_on_detection": mean(
            boundaries, "boundary_absolute_error_s"),
        "boundary_mean_signed_error_s": mean(
            boundaries, "boundary_signed_error_s"),
        "online_detection_latency_mean_s": mean(
            boundaries, "online_detection_latency_s"),
        "perception_delay_mean_s": mean(boundaries, "perception_delay_s"),
        "decoder_delay_mean_s": mean(boundaries, "decoder_delay_s"),
        "false_transition_count": sum(
            boundary["false_transition"] for boundary in boundaries),
        "full_chain_accuracy": sum(
            chain["full_chain_correct"] for chain in chains) / len(chains),
        "chains": chains,
    }
    if "parameters" in parts[0]:
        result["parameters"] = parts[0]["parameters"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("summaries", type=Path, nargs="+")
    args = parser.parse_args()
    shards = [json.loads(path.read_text()) for path in args.summaries]
    names = [result["name"] for result in shards[0]["results"]]
    if any([result["name"] for result in shard["results"]] != names
           for shard in shards):
        raise ValueError("shards do not contain the same decoder variants")
    by_name = {name: [] for name in names}
    for shard in shards:
        for result in shard["results"]:
            by_name[result["name"]].append(result)
    health = Counter()
    vote_errors = Counter()
    backend_errors = Counter()
    for shard in shards:
        item = shard["evidence_health"]
        for key in ("causal_windows", "phase_assessments", "individual_votes",
                    "backend_error_count", "vote_error_count"):
            health[key] += item[key]
        vote_errors.update(item["vote_errors_by_type"])
        backend_errors.update(item["backend_errors_by_type"])
    health["vote_errors_by_type"] = dict(vote_errors)
    health["backend_errors_by_type"] = dict(backend_errors)
    metadata = [item for shard in shards for item in shard["evidence_metadata"]]
    metadata.sort(key=lambda item: item["first_episode"])
    for item in metadata:
        item["chain_index"] = (item["first_episode"] - 237) // 5 + 1
    output = {
        "protocol": {
            **shards[0]["protocol"],
            "chains": len(metadata),
            "episode_range": [metadata[0]["first_episode"],
                              metadata[-1]["first_episode"] + 4],
            "shard_count": len(shards),
        },
        "evidence_health": dict(health),
        "evidence_metadata": metadata,
        "results": [aggregate_variant(name, by_name[name]) for name in names],
        "source_summaries": [str(path) for path in args.summaries],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "chains": len(metadata),
        "episode_range": output["protocol"]["episode_range"],
        "causal_windows": health["causal_windows"],
    }, indent=2))


if __name__ == "__main__":
    main()
