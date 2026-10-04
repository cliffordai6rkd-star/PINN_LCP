"""Read physical logs and measure replan revisions and issued-command jitter.

The .pt input must follow docs/trajectory_jitter_evaluation.md. q must be in
radians, timestamps int64 nanoseconds, unique and increasing within each trace.
selected_sample is provided by the log; this tool never chooses a sample/mode.
Replans are matched at equal physical future timestamps, separately per episode.
Missing overlap, takeover metadata, or derivatives produce null, never zero.
This offline tool loads no checkpoint, filters nothing and sends no commands.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from train.trajectory_jitter_metrics import evaluate_jitter_log


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, required=True, help="trajectory_jitter_v1 .pt physical forecast/execution log")
    parser.add_argument("--output", type=Path, help="optional JSON report; always also printed")
    args = parser.parse_args()
    payload = torch.load(args.log, map_location="cpu", weights_only=True)
    result = evaluate_jitter_log(payload)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
