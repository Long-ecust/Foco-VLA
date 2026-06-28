"""Sanity-check the Piper URDF against recorded data.

Loads piper_description.urdf, lumps the gripper as a fixed body on link6,
then on a longvla_raw_v2 HDF5 episode:

  1. Identifies static frames (joint velocities ~ 0).
  2. Computes pinocchio gravity torque g(q) at each static frame.
  3. Reports per-joint residual = filtered_effort - g(q).

Per-joint residual quantifies how much of the static torque is *not* explained
by raw CAD gravity. Large residual = CAD inertials are wrong (parameter
identification needed) and/or there is significant joint friction at standstill
(unlikely to be huge but possible from harmonic drives).

Convention: signs / axis directions in URDF are taken at face value. If a
joint's measured torque is consistently the negative of g(q), the data
collection or driver is reporting tau with opposite sign convention --
something to fix once, in one place.
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import pinocchio as pin


# right-arm slice in the bimanual schema
RIGHT_ARM_Q = slice(7, 13)   # 6 arm joints, exclude gripper (idx 13)
RIGHT_ARM_T = slice(7, 13)   # same for tau


def lump_gripper_into_link6(model: pin.Model) -> None:
    """Add a rigid body equivalent to {gripper_base + 2 fingers} fixed to link6.

    Numbers from piper_with_gripper_description.xacro:
      gripper_base : m=0.45,  com_in_link6 ~ (0, 0, 0.0321)
      gripper_link1: m=0.025, parent gripper_base, origin xyz=(0,0,0.1358)
      gripper_link2: m=0.025, parent gripper_base, origin xyz=(0,0,0.1358)

    Approximation: treat fingers at their nominal mid-stroke position. For a
    parameter-identification later we will replace this with a properly built
    model from the gripper xacro. For now we only need a reasonable g(q).
    """
    j6 = model.getJointId("joint6")
    # All offsets here are in the link6 frame (joint6 child link).
    # gripper_base inertial origin in xacro:
    base_com = np.array([-0.000183807, 8.05e-05, 0.0321436690])
    base_mass = 0.45
    base_I = np.diag([0.00092934, 0.00071447, 0.00039442])
    # parallel-axis from base_com
    I_base = pin.Inertia(base_mass, base_com, base_I)

    # finger masses, lumped at their mounting frame (z=0.1358 in link6 frame):
    finger_offset = np.array([0.0, 0.0, 0.1358])
    finger_mass = 0.025
    finger_inertia_local = np.diag([0.00007371, 0.00000781, 0.0000747])
    I_finger = pin.Inertia(finger_mass, finger_offset, finger_inertia_local)

    # combine all into one rigid body, then add to link6's inertia
    extra = I_base + I_finger + I_finger
    model.inertias[j6] = model.inertias[j6] + extra


def load_episode(path: str):
    with h5py.File(path, "r") as f:
        qpos = f["observations/qpos"][...]
        qvel = f["observations/qvel"][...]
        fil_effort = f["observations/filtered_effort"][...]
        attrs = dict(f.attrs)
    return qpos, qvel, fil_effort, attrs


def static_mask(qvel_arm: np.ndarray, vel_thresh: float = 0.02) -> np.ndarray:
    """Return boolean mask of frames where every arm joint is below thresh."""
    return np.max(np.abs(qvel_arm), axis=1) < vel_thresh


def compute_gravity_residual(
    model: pin.Model, data: pin.Data, q_arm: np.ndarray, tau_arm: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (g_at_static, residual = tau - g)."""
    idx = np.where(mask)[0]
    g_vals = np.zeros((len(idx), model.nv))
    for k, i in enumerate(idx):
        g_vals[k] = pin.computeGeneralizedGravity(model, data, q_arm[i])
    return g_vals, tau_arm[idx] - g_vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", help="path to a single .hdf5 episode")
    ap.add_argument(
        "--urdf",
        default="examples/piper_bimanual/urdf/piper/urdf/piper_description.urdf",
    )
    ap.add_argument("--vel-thresh", type=float, default=0.02, help="rad/s static threshold")
    args = ap.parse_args()

    model = pin.buildModelFromUrdf(args.urdf)
    lump_gripper_into_link6(model)
    data = model.createData()
    print(f"URDF loaded: {model.nq} dofs, {model.njoints} joints")
    total_mass = sum(model.inertias[i].mass for i in range(1, model.njoints))
    print(f"total mass after lumping gripper: {total_mass:.3f} kg")

    qpos, qvel, fil_effort, attrs = load_episode(args.episode)
    q_arm = qpos[:, RIGHT_ARM_Q]
    qd_arm = qvel[:, RIGHT_ARM_Q]
    tau_arm = fil_effort[:, RIGHT_ARM_T]
    print(f"\nepisode: {os.path.basename(args.episode)}  T={len(qpos)}  "
          f"task={attrs.get('task')!r}")

    # qpos / qvel range sanity (URDF expects rad)
    print("qpos arm range (rad):", q_arm.min(axis=0), "→", q_arm.max(axis=0))
    print("qvel arm max|.| (rad/s):", np.max(np.abs(qd_arm), axis=0))

    mask = static_mask(qd_arm, args.vel_thresh)
    print(f"\nstatic frames (max|qdot|<{args.vel_thresh} rad/s): "
          f"{mask.sum()}/{len(mask)}  ({100 * mask.mean():.1f}%)")
    if mask.sum() < 30:
        print("!! too few static frames, lower --vel-thresh"); return

    g_vals, residual = compute_gravity_residual(model, data, q_arm, tau_arm, mask)

    np.set_printoptions(formatter={"float": lambda x: f"{x:+7.3f}"})
    print("\nstatic-frame statistics (right arm joints j1..j6):")
    print(f"  measured tau   mean : {tau_arm[mask].mean(axis=0)}")
    print(f"  measured tau   std  : {tau_arm[mask].std(axis=0)}")
    print(f"  CAD g(q)       mean : {g_vals.mean(axis=0)}")
    print(f"  CAD g(q)       std  : {g_vals.std(axis=0)}")
    print(f"  residual       mean : {residual.mean(axis=0)}")
    print(f"  residual       std  : {residual.std(axis=0)}")
    print(f"  residual |.| p95    : {np.percentile(np.abs(residual), 95, axis=0)}")

    # also report a sign hint
    corr = np.array([
        np.corrcoef(tau_arm[mask, j], g_vals[:, j])[0, 1] if g_vals[:, j].std() > 1e-6 else np.nan
        for j in range(model.nv)
    ])
    print(f"\n  corr(tau, g(q)) per joint : {corr}")
    print("  (close to +1 means URDF gravity matches measurement up to params;")
    print("   close to -1 means torque sign convention flipped.)")


if __name__ == "__main__":
    main()
