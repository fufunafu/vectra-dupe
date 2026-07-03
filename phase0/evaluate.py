"""Multi-seed evaluation of the Phase 0 pipeline.

Runs the accuracy experiment (known 0.8 mL bump) and the noise-floor
experiment (no change) across several random seeds and reports statistics.

Run:  .venv/bin/python evaluate.py
"""

import time

import numpy as np
import open3d as o3d

from run_phase0 import EXCLUDE_RADIUS_MM, ROI_RADIUS_MM, reconstruct_session
from vectra3d import analyze, capture, register, synthetic

N_SEEDS = 8


def one_experiment(seed: int, head, head_after, bump):
    rng = np.random.default_rng(seed)
    sensor = capture.Sensor(rng)

    recon_before = reconstruct_session(head, sensor, rng)

    def compare_to_before(target_mesh):
        recon = reconstruct_session(
            target_mesh, sensor, rng, synthetic.random_session_transform(rng))
        transform = register.register_after_to_before(
            recon, recon_before, bump.center, EXCLUDE_RADIUS_MM)
        aligned = o3d.geometry.TriangleMesh(recon).transform(transform)
        field = analyze.signed_distance_field(recon_before, aligned)
        field = analyze.subtract_bias_field(field, bump.center, EXCLUDE_RADIUS_MM)
        return analyze.roi_volume_mm3(field, bump.center, ROI_RADIUS_MM)

    return compare_to_before(head_after), compare_to_before(head)


def main():
    t0 = time.time()
    head = synthetic.make_head()
    bump = synthetic.cheek_bump_spec(head, amplitude_mm=2.0, sigma_mm=8.0)
    head_after = synthetic.apply_bump(head, bump)
    gt = (synthetic.closed_mesh_volume(head_after)
          - synthetic.closed_mesh_volume(head))

    measured, nulls = [], []
    for seed in range(N_SEEDS):
        m, n = one_experiment(seed, head, head_after, bump)
        measured.append(m)
        nulls.append(n)
        print(f"seed {seed}: measured {m / 1000:6.3f} mL   null {n / 1000:+6.3f} mL")

    measured = np.array(measured) / 1000.0
    nulls = np.array(nulls) / 1000.0
    gt_ml = gt / 1000.0
    print()
    print(f"ground truth      : {gt_ml:.3f} mL")
    print(f"measured          : {measured.mean():.3f} +/- {measured.std():.3f} mL "
          f"(bias {100 * (measured.mean() - gt_ml) / gt_ml:+.1f}%)")
    print(f"null (truth 0)    : {nulls.mean():+.3f} +/- {nulls.std():.3f} mL")
    print(f"per-measurement sigma estimate: {nulls.std():.3f} mL")
    print(f"runtime: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
