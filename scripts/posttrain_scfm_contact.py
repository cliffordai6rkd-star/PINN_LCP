"""Independent SCFM post-training for schema-10 carswm_v9 GRU checkpoints.

Uses the teacher's original 100 Hz data, preprocessing and normalizer.
The trained Flow is exported without any architecture change.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.posttrain_scfm_latent import main

if __name__ == "__main__":
    main(architecture="contact")
