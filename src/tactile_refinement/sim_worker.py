"""Explicitly opt into depth observations and two-step target holds."""
from smolvla_baseline.sim_worker import main

if __name__ == '__main__':
    main(refinement=True)
