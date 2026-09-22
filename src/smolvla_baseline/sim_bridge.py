"""Private local IPC bridge to UniVTAC's separate Isaac Sim environment.

Only trusted child processes use this socket; it is never exposed on TCP.
The training Python environment is not modified to install simulator packages.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import socket
import struct
import subprocess
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_MESSAGE = 64 * 1024 * 1024


def _pack(value):
    # NumPy 1.x (Isaac) and 2.x (training) have incompatible pickle module paths.
    if isinstance(value, np.ndarray):
        return {'__array__': True, 'dtype': value.dtype.str, 'shape': value.shape,
                'data': np.ascontiguousarray(value).tobytes()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _pack(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_pack(v) for v in value]
    return value


def _unpack(value):
    if isinstance(value, dict):
        if value.get('__array__') is True:
            return np.frombuffer(value['data'], dtype=value['dtype']).reshape(value['shape']).copy()
        return {k: _unpack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unpack(v) for v in value]
    return value


def send_message(channel: socket.socket, value) -> None:
    data = pickle.dumps(_pack(value), protocol=5)
    if len(data) > MAX_MESSAGE:
        raise ValueError("simulator IPC message too large")
    channel.sendall(struct.pack("!Q", len(data)) + data)


def receive_message(channel: socket.socket):
    def read(size):
        result = bytearray()
        while len(result) < size:
            block = channel.recv(size-len(result))
            if not block:
                raise EOFError("simulator worker exited; inspect its log")
            result.extend(block)
        return result
    length, = struct.unpack("!Q", read(8))
    if length > MAX_MESSAGE:
        raise ValueError("simulator IPC message too large")
    return _unpack(pickle.loads(read(length)))


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def convert_observation(raw: dict, instruction: str) -> dict:
    result = {"instruction": instruction}
    for name in ("head", "wrist"):
        rgb = as_numpy(raw["observation"][name]["rgb"])
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"UniVTAC {name} RGB must be uint8 HWC, got {rgb.shape}/{rgb.dtype}")
        result[name] = rgb.copy()
    state = as_numpy(raw["embodiment"]["joint"]).reshape(-1).astype(np.float32)
    if state.shape != (9,) or not np.isfinite(state).all():
        raise ValueError("UniVTAC joint observation must contain nine finite values")
    result["state"] = state
    # Deliberately ignore raw['tactile']; it is never sent to the policy.
    return result


def action_to_qpos(action) -> np.ndarray:
    """Official qpos API broadcasts finger[0] to both finger joints."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (9,) or not np.isfinite(action).all():
        raise ValueError("SmolVLA action must be a finite 9-vector")
    return action[:8].copy()


def initial_state_digest(raw: dict) -> str:
    """Observed robot/actor state signature, NOT a full UIPC-state checksum.

    A 1e-5 tolerance avoids bytewise GPU float jitter. Hidden deformable states
    are controlled by reset seed/config but are not included in this signature.
    """
    def convert(value):
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in sorted(value.items())}
        return np.round(as_numpy(value).astype(np.float64), 5).tolist()
    state = {"joints": convert(raw["embodiment"]["joint"]), "actors": convert(raw["actor"])}
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


class UniVTACEnvironment:
    is_simulation = True

    def __init__(self, *, task: str, worker_module: str = 'smolvla_baseline.sim_worker', worker_args=()):
        from .instructions import DEFAULT_TASKS
        if task not in DEFAULT_TASKS:
            raise ValueError(f"unsupported UniVTAC task: {task}")
        simulator_python = PROJECT_ROOT / ".venv-benchmark/bin/python"
        simulator_root = PROJECT_ROOT / "UniVTAC"
        if not simulator_python.exists():
            raise FileNotFoundError("run scripts/setup_benchmark.sh first")
        if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").upper() not in {"YES", "Y", "1"}:
            raise RuntimeError("Isaac Sim EULA acceptance is required before simulator startup")
        parent, child = socket.socketpair()
        self.channel = parent
        self.channel.settimeout(900)
        log_dir = PROJECT_ROOT / "logs/univtac_workers"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log = (log_dir / f"{task}-{os.getpid()}.log").open("a")
        env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT / "src"),
                   OMP_NUM_THREADS="2", NO_ALBUMENTATIONS_UPDATE="1")
        command = [str(simulator_python), "-u", "-m", worker_module,
                   "--socket-fd", str(child.fileno()), "--task", task, *worker_args]
        try:
            self.process = subprocess.Popen(command, cwd=simulator_root, env=env,
                pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL, stdout=self.log, stderr=self.log)
            child.close()
            self._response()
        except BaseException:
            child.close()
            self.close()
            raise

    def _response(self):
        response = receive_message(self.channel)
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["value"]

    def reset(self, *, seed):
        send_message(self.channel, {"command": "reset", "seed": seed})
        return self._response()

    def step(self, action):
        send_message(self.channel, {"command": "step", "action": np.asarray(action)})
        return self._response()

    def close(self):
        try:
            if hasattr(self, "process") and self.process.poll() is None:
                try:
                    send_message(self.channel, {"command": "close"})
                    self.process.wait(timeout=20)
                except (OSError, subprocess.TimeoutExpired):
                    # Only this worker is ours; never signal other GPU jobs.
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
        finally:
            self.channel.close()
            self.log.close()


def make_env(*, task: str):
    return UniVTACEnvironment(task=task)
