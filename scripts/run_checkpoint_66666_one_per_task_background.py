"""Run one matched-60 Stage 2 rollout per UniVTAC task and upload videos."""

import json
import os
from pathlib import Path
import subprocess
import time

from huggingface_hub import CommitOperationAdd, HfApi


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs/tactile_refinement_trex_fullgrid_n0/expert/checkpoint_2.pt"
OUTPUT = ROOT / "runs/benchmark/stage2_checkpoint_66666_matched60_one_per_task_20260917"
STATUS = ROOT / "logs/stage2_checkpoint_66666_one_per_task_status.json"
REPO = "sungkyunner/univtac-stage2-tactile-ablation-checkpoint-33333"
TASKS = (
    "grasp_classify",
    "pull_out_key",
    "insert_HDMI",
    "lift_bottle",
    "lift_can",
    "insert_tube",
    "put_bottle_in_shelf",
    "insert_hole",
)
SEED = 2_000_000


def write_status(**values):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(values, indent=2) + "\n")
    os.replace(temporary, STATUS)


def evaluate():
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        OMNI_KIT_ACCEPT_EULA="YES",
        OMNI_KIT_ALLOW_ROOT="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="4",
    )
    command = [
        str(ROOT / ".venv-train/bin/python"),
        "-u",
        "-m",
        "tactile_refinement.ablation_rollout",
        "--checkpoint",
        str(CHECKPOINT),
        "--tasks",
        *TASKS,
        "--conditions",
        "correct",
        "--episodes",
        "1",
        "--start-seed",
        str(SEED),
        "--video-episodes",
        "1",
        "--schedule",
        "matched60",
        "--output",
        str(OUTPUT),
    ]
    if (OUTPUT / "metadata.json").exists():
        command.append("--resume")
    for attempt in range(1, 4):
        write_status(phase="evaluating", attempt=attempt, command=command, checked_at=time.time())
        result = subprocess.run(command, cwd=ROOT, env=env)
        if result.returncode == 0:
            return
        if "--resume" not in command:
            command.append("--resume")
        write_status(
            phase="retry_pending",
            attempt=attempt,
            returncode=result.returncode,
            checked_at=time.time(),
        )
        time.sleep(15)
    raise RuntimeError("Checkpoint 66,666 evaluation failed after three attempts")


def upload():
    rows = [json.loads(line) for line in (OUTPUT / "episodes.jsonl").read_text().splitlines()]
    expected = {(task, SEED, "correct") for task in TASKS}
    actual = {(row["task"], row["seed"], row["condition"]) for row in rows}
    if actual != expected or len(rows) != len(TASKS):
        raise RuntimeError(f"Expected exactly eight completed rollouts; got {len(rows)}: {actual}")

    prefix = "checkpoint_66666/matched60"
    operations = [
        CommitOperationAdd(
            path_in_repo=f"{prefix}/metadata.json",
            path_or_fileobj=str(OUTPUT / "metadata.json"),
        ),
        CommitOperationAdd(
            path_in_repo=f"{prefix}/episodes.jsonl",
            path_or_fileobj=str(OUTPUT / "episodes.jsonl"),
        ),
        CommitOperationAdd(
            path_in_repo=f"{prefix}/summary.json",
            path_or_fileobj=str(OUTPUT / "summary.json"),
        ),
    ]
    videos = []
    for task in TASKS:
        local = OUTPUT / "videos" / task / f"seed_{SEED}_correct.mp4"
        if not local.exists():
            raise FileNotFoundError(local)
        remote = f"{prefix}/videos/{task}/{local.name}"
        operations.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)))
        videos.append(remote)

    manifest = OUTPUT / "video_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "checkpoint_step": 66_666,
                "schedule": "matched60",
                "seed": SEED,
                "condition": "correct",
                "tasks": TASKS,
                "videos": videos,
            },
            indent=2,
        )
        + "\n"
    )
    operations.append(
        CommitOperationAdd(path_in_repo=f"{prefix}/video_manifest.json", path_or_fileobj=str(manifest))
    )
    result = HfApi().create_commit(
        repo_id=REPO,
        repo_type="dataset",
        operations=operations,
        commit_message="Upload checkpoint 66666 matched60 videos for all eight tasks",
    )
    return result.commit_url, videos


def main():
    try:
        evaluate()
        write_status(phase="uploading", checked_at=time.time())
        commit_url, videos = upload()
        write_status(
            phase="complete_and_uploaded",
            commit_url=commit_url,
            videos=videos,
            checked_at=time.time(),
        )
        print(json.dumps({"phase": "complete_and_uploaded", "commit_url": commit_url, "videos": videos}))
    except Exception as exc:
        write_status(phase="failed", error=repr(exc), checked_at=time.time())
        raise


if __name__ == "__main__":
    main()
