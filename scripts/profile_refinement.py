"""Short real-data memory/timing probe; no checkpoints or production training."""
import argparse
import contextlib
import io
import json

import torch

from smolvla_baseline.data import ROOT
from tactile_refinement.train import active_stages, build_refiner, run_stage


class Capture(io.StringIO):
    def __init__(self, terminal):
        super().__init__(); self.terminal = terminal
    def write(self, value):
        self.terminal.write(value)
        return super().write(value)
    def flush(self):
        self.terminal.flush()


def main():
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--updates', type=int, default=1)
    args = parser.parse_args()
    config = json.loads((ROOT/'configs/tactile_refinement.json').read_text())
    torch.set_num_threads(config['cpu_threads']); torch.manual_seed(config['seed'])
    model, pre, _ = build_refiner(config)
    capture = Capture(sys.stdout)
    with contextlib.redirect_stdout(capture):
        for stage in active_stages(config):
            run_stage(model, pre, config, stage, {}, profile_updates=args.updates)
            torch.cuda.empty_cache()
    rows = [json.loads(line) for line in capture.getvalue().splitlines() if line.startswith('{')]
    report = {'gpu': torch.cuda.get_device_name(), 'batch_size': config['batch_size'],
              'gradient_accumulation_steps': config['gradient_accumulation_steps'],
              'note': 'Short setup-inclusive update timings while baseline evaluation also runs; not a convergence or uncontended throughput measurement.',
              'rows': rows}
    (ROOT/'logs/refinement_profile.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
