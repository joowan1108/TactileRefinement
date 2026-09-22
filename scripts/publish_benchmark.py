#!/usr/bin/env python
"""Validate and publish a completed UniVTAC baseline benchmark."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import cv2
from huggingface_hub import HfApi


TASKS = (
    "grasp_classify", "pull_out_key", "insert_HDMI", "lift_bottle",
    "lift_can", "insert_tube", "put_bottle_in_shelf", "insert_hole",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--videos-per-task", type=int, default=3)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "summary.json"
    episodes_path = run_dir / "episodes.jsonl"
    if not summary_path.exists() or not episodes_path.exists():
        raise FileNotFoundError("Benchmark is incomplete: summary.json or episodes.jsonl is missing")
    summary = json.loads(summary_path.read_text())
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    expected = len(TASKS) * args.episodes_per_task
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} episode records, found {len(rows)}")

    result_rows = []
    for task in TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        if len(task_rows) != args.episodes_per_task:
            raise ValueError(f"{task}: expected {args.episodes_per_task} episodes, found {len(task_rows)}")
        successes = sum(bool(row["success"]) for row in task_rows)
        result_rows.append({
            "task": task,
            "successes": successes,
            "episodes": len(task_rows),
            "success_rate": successes / len(task_rows),
        })
        video_rows = [row for row in task_rows if row.get("video")]
        if len(video_rows) != args.videos_per_task:
            raise ValueError(f"{task}: expected {args.videos_per_task} videos, found {len(video_rows)}")
        for row in video_rows:
            path = run_dir / row["video"]
            capture = cv2.VideoCapture(str(path))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            opened = capture.isOpened()
            capture.release()
            if not opened or frames < 2:
                raise ValueError(f"Invalid rollout video: {path} ({frames} frames)")

    with (run_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("task", "successes", "episodes", "success_rate"))
        writer.writeheader()
        writer.writerows(result_rows)

    provenance = run_dir / "provenance"
    provenance.mkdir(exist_ok=True)
    for source in (
        Path("runs/smolvla_full_dense_b16_ga4_72h/recipe.json"),
        Path("runs/smolvla_full_dense_b16_ga4_72h/parameter_training.json"),
        Path("runs/smolvla_full_dense_b16_ga4_72h/dataset_sha256.txt"),
        Path("configs/benchmark_environment.json"),
    ):
        shutil.copy2(source, provenance / source.name)

    checkpoint_hash = sha256(args.checkpoint)
    overall_successes = sum(row["successes"] for row in result_rows)
    overall_episodes = sum(row["episodes"] for row in result_rows)
    table = "\n".join(
        f'| `{row["task"]}` | {row["successes"]}/{row["episodes"]} | {100*row["success_rate"]:.1f}% |'
        for row in result_rows
    )
    action_chunk = metadata.get("action_chunk", 50)
    physics_steps = metadata.get("physics_steps_per_action", 1)
    executed_actions = action_chunk if metadata.get("baseline") else 10
    schedule = (f"{action_chunk} predicted actions, all {executed_actions} executed before replanning; "
                f"each target held for {physics_steps} physics step{'s' if physics_steps != 1 else ''}")
    title_suffix = " — full-50/two-step schedule" if action_chunk == 50 and physics_steps == 2 else ""
    card = f"""---
pretty_name: UniVTAC SmolVLA Baseline Evaluation
tags:
- robotics
- tactile-sensing
- vision-language-action
- univtac
---

# UniVTAC SmolVLA baseline evaluation{title_suffix}

Closed-loop evaluation of the tactile-free SmolVLA baseline trained on all 800 downloaded UniVTAC Isaac45 demonstrations. Tactile observations are simulated for environment physics and success predicates but are never sent to the policy.

## Results

| Task | Successes | Success rate |
|---|---:|---:|
{table}
| **Macro/pooled total** | **{overall_successes}/{overall_episodes}** | **{100*overall_successes/overall_episodes:.1f}%** |

Each task uses {args.episodes_per_task} deterministic evaluation seeds beginning at 1,000,000. These evaluation rollouts are separate from the training demonstrations. Success is determined by UniVTAC's task-specific closed-loop predicate.

The `videos/` directory contains the first {args.videos_per_task} evaluation episodes for every task. Each MP4 shows the head and wrist RGB observations side by side at policy action cadence. `episodes.jsonl` contains every rollout, `results.csv` contains the per-task totals, and `summary.json` contains the machine-readable benchmark summary.

Checkpoint: `{args.checkpoint.name}`  
Checkpoint SHA-256: `{checkpoint_hash}`  
Execution schedule: {schedule}  
Policy inputs: head RGB, wrist RGB, robot joint state, language instruction  
Tactile policy inputs: none
"""
    (run_dir / "README.md").write_text(card)
    (run_dir / "checkpoint_sha256.txt").write_text(checkpoint_hash + "\n")

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    result = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=run_dir,
        commit_message=f"Upload complete SmolVLA baseline evaluation: {schedule}",
    )
    print(json.dumps({"repo_id": args.repo_id, "commit_url": result.commit_url}, indent=2), flush=True)


if __name__ == "__main__":
    main()
