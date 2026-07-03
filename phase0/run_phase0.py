"""Phase 0 feasibility experiment.

Simulates the full phone workflow twice and answers the two go/no-go questions:

1. ACCURACY  — capture a synthetic head before and after adding a cheek bump of
   exactly known volume (the "filler"); does the pipeline recover that volume?
2. NOISE FLOOR — capture the SAME unchanged head in two sessions; how much
   phantom volume does the pipeline report where the true answer is zero?

Run:  .venv/bin/python run_phase0.py
"""

import os
import time

import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from vectra3d import analyze, capture, fuse, register, synthetic

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ROI_RADIUS_MM = 24.0  # 3 sigma of the bump (contains 98.9% of its volume)
EXCLUDE_RADIUS_MM = 28.0  # ROI + margin, masked out of registration + bias fit


def reconstruct_session(mesh, sensor, rng, session_transform=None):
    """Capture 3 noisy views of (optionally transformed) mesh and fuse them."""
    scan_target = o3d.geometry.TriangleMesh(mesh)
    if session_transform is not None:
        scan_target.transform(session_transform)
    poses = capture.capture_session(scan_target, sensor, rng)
    return fuse.fuse_session(poses)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(7)
    sensor = capture.Sensor(rng)

    print("Building synthetic head + ground-truth bump ...")
    head = synthetic.make_head()
    bump = synthetic.cheek_bump_spec(head, amplitude_mm=2.0, sigma_mm=8.0)
    head_after = synthetic.apply_bump(head, bump)
    gt_mm3 = (synthetic.closed_mesh_volume(head_after)
              - synthetic.closed_mesh_volume(head))
    print(f"  ground-truth bump volume: {gt_mm3:.0f} mm^3 ({gt_mm3 / 1000:.3f} mL)")

    print("Session A: capture + fuse 'before' ...")
    recon_before = reconstruct_session(head, sensor, rng)

    print("Session B: capture + fuse 'after' (new sitting position) ...")
    recon_after = reconstruct_session(
        head_after, sensor, rng, synthetic.random_session_transform(rng))

    print("Registering B -> A on stable regions ...")
    transform = register.register_after_to_before(
        recon_after, recon_before, bump.center, EXCLUDE_RADIUS_MM)
    recon_after_aligned = o3d.geometry.TriangleMesh(recon_after).transform(transform)

    print("Measuring signed distance + ROI volume ...")
    field = analyze.signed_distance_field(recon_before, recon_after_aligned)
    field = analyze.subtract_bias_field(field, bump.center, EXCLUDE_RADIUS_MM)
    measured_mm3 = analyze.roi_volume_mm3(field, bump.center, ROI_RADIUS_MM)
    stable_rms = analyze.rms_distance_mm(field, bump.center, EXCLUDE_RADIUS_MM)

    print("Noise-floor run: re-scan the unchanged head (session C) ...")
    recon_repeat = reconstruct_session(
        head, sensor, rng, synthetic.random_session_transform(rng))
    transform_c = register.register_after_to_before(
        recon_repeat, recon_before, bump.center, EXCLUDE_RADIUS_MM)
    recon_repeat_aligned = o3d.geometry.TriangleMesh(recon_repeat).transform(transform_c)
    field_null = analyze.signed_distance_field(recon_before, recon_repeat_aligned)
    field_null = analyze.subtract_bias_field(field_null, bump.center, EXCLUDE_RADIUS_MM)
    null_mm3 = analyze.roi_volume_mm3(field_null, bump.center, ROI_RADIUS_MM)
    null_rms = analyze.rms_distance_mm(field_null)

    print("Writing outputs ...")
    analyze.save_heatmap_png(
        field, os.path.join(RESULTS, "heatmap_before_after.png"),
        f"Before vs After — ROI volume {measured_mm3 / 1000:.3f} mL "
        f"(truth {gt_mm3 / 1000:.3f} mL)")
    analyze.save_heatmap_png(
        field_null, os.path.join(RESULTS, "heatmap_noise_floor.png"),
        f"Repeat scan, no change — phantom volume {null_mm3 / 1000:.3f} mL")
    analyze.save_colored_mesh(
        recon_before, field, os.path.join(RESULTS, "before_with_heatmap.ply"))
    o3d.io.write_triangle_mesh(os.path.join(RESULTS, "recon_before.ply"), recon_before)
    o3d.io.write_triangle_mesh(
        os.path.join(RESULTS, "recon_after_aligned.ply"), recon_after_aligned)

    err_pct = 100 * (measured_mm3 - gt_mm3) / gt_mm3
    report = "\n".join([
        "PHASE 0 RESULTS",
        "===============",
        f"Ground-truth bump volume : {gt_mm3 / 1000:8.3f} mL",
        f"Measured ROI volume      : {measured_mm3 / 1000:8.3f} mL  ({err_pct:+.1f}%)",
        f"Noise-floor phantom vol  : {null_mm3 / 1000:8.3f} mL  (truth: 0)",
        f"Stable-region RMS (A vs B): {stable_rms:7.3f} mm",
        f"Repeat-scan RMS (A vs C)  : {null_rms:7.3f} mm",
        "",
        f"Verdict: noise floor is {abs(null_mm3 / gt_mm3) * 100:.1f}% of a "
        f"{gt_mm3 / 1000:.1f} mL effect.",
        f"Total runtime: {time.time() - t0:.1f} s",
    ])
    print("\n" + report)
    with open(os.path.join(RESULTS, "report.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
