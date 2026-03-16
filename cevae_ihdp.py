"""Run CEVAE on the IHDP dataset.

This is a thin wrapper around run_experiment.py kept for backwards
compatibility.  Prefer using the unified runner directly::

    python run_experiment.py --dataset ihdp --models cevae --replications 10
"""

from run_experiment import main

if __name__ == "__main__":
    main(["--dataset", "ihdp", "--models", "cevae", "--replications", "10"])
