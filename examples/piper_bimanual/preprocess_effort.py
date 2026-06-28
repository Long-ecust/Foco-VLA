#!/usr/bin/env python3
"""
数据预处理: 电流 → 力矩转换 + 重力补偿

Usage:
    uv run python examples/piper_bimanual/preprocess_effort.py \    
        examples/datasets/episode_*.hdf5 \
        --urdf examples/piper_bimanual/urdf/piper/urdf/piper_description.urdf \
        --kt-left 0.3 0.3 0.3 0.3 0.3 0.3 0.1 \
        --kt-right 0.3 0.3 0.3 0.3 0.3 0.3 0.1 \
        --gear-ratio 100 100 100 50 50 50 100 \
        --output-suffix _preprocessed
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False
    print("Warning: pinocchio not installed. Run: uv pip install pinocchio")


def lump_gripper_into_link6(model: pin.Model) -> None:
    """Add gripper mass to link6 (from validate_urdf_baseline.py)."""
    if not HAS_PINOCCHIO:
        return
    
    j6 = model.getJointId("joint6")
    base_com = np.array([-0.000183807, 8.05e-05, 0.0321436690])
    base_mass = 0.45
    base_I = np.diag([0.00092934, 0.00071447, 0.00039442])
    I_base = pin.Inertia(base_mass, base_com, base_I)

    finger_offset = np.array([0.0, 0.0, 0.1358])
    finger_mass = 0.025
    finger_inertia_local = np.diag([0.00007371, 0.00000781, 0.0000747])
    I_finger = pin.Inertia(finger_mass, finger_offset, finger_inertia_local)

    extra = I_base + I_finger + I_finger
    model.inertias[j6] = model.inertias[j6] + extra


def preprocess_episode(
    hdf5_path: str,
    kt_left: np.ndarray,
    kt_right: np.ndarray,
    gear_ratio: np.ndarray,
    model: Optional[pin.Model] = None,
    output_path: Optional[str] = None,
) -> dict:
    """
    Load episode and apply corrections:
    1. Current (A) → Torque (Nm): tau = current * kt * gear_ratio
    2. Gravity compensation: residual_tau = tau - g(q)
    """
    print(f"Processing: {Path(hdf5_path).name}")
    
    with h5py.File(hdf5_path, "r") as f:
        qpos = f["observations/qpos"][...]  # (T, 14)
        qvel = f["observations/qvel"][...]
        effort_A = f["observations/effort"][...]  # (T, 14) in Amperes
        fil_effort_A = f["observations/filtered_effort"][...]
        
        # Copy all other data
        other_data = {}
        for key in f["observations"]:
            if key not in ["qpos", "qvel", "effort", "filtered_effort"]:
                other_data[key] = f[f"observations/{key}"][...]
        
        attrs = dict(f.attrs)
    
    T = len(qpos)
    
    # Current → Torque conversion
    # kt_left and kt_right are per-joint torque constants (Nm/A)
    # gear_ratio is the same for all joints (same arm structure)
    kt = np.concatenate([kt_left, kt_right])  # (14,)
    gr = np.tile(gear_ratio, 2)  # (14,)
    
    coeff = kt * gr  # (14,)
    effort_Nm = effort_A * coeff[np.newaxis, :]  # (T, 14)
    fil_effort_Nm = fil_effort_A * coeff[np.newaxis, :]
    
    print(f"  Torque conversion applied: {coeff}")
    print(f"  effort_A  range: [{effort_A.min():.4f}, {effort_A.max():.4f}]")
    print(f"  effort_Nm range: [{effort_Nm.min():.4f}, {effort_Nm.max():.4f}]")
    
    # Gravity compensation (if model available)
    gravity_q = None
    residual = None
    
    if model is not None:
        print(f"  Computing gravity compensation...")
        data = model.createData()
        gravity_q = np.zeros((T, 14))
        
        # Right arm only (first arm is left, stationary in your data)
        right_arm_q = slice(7, 13)  # 6 DOF joints
        
        for t in range(T):
            # Build full config for pinocchio (need all joints)
            q_full = np.zeros(model.nq)
            q_full[right_arm_q] = qpos[t, right_arm_q]  # right arm
            
            g_full = pin.computeGeneralizedGravity(model, data, q_full)
            gravity_q[t] = g_full[:14]  # Extract first 14 components
        
        residual = fil_effort_Nm - gravity_q
        print(f"  Gravity compensation shape: {gravity_q.shape}")
        print(f"  Right arm gravity range: [{gravity_q[:, right_arm_q].min():.4f}, "
              f"{gravity_q[:, right_arm_q].max():.4f}]")
        print(f"  Residual (tau - g) range: [{residual.min():.4f}, {residual.max():.4f}]")
    
    # Save to output if specified
    if output_path:
        print(f"  Saving to: {output_path}")
        with h5py.File(output_path, "w") as fout:
            # Original observations with corrected effort
            obs_group = fout.create_group("observations")
            obs_group.create_dataset("qpos", data=qpos)
            obs_group.create_dataset("qvel", data=qvel)
            obs_group.create_dataset("effort_A", data=effort_A)
            obs_group.create_dataset("effort_Nm", data=effort_Nm)
            obs_group.create_dataset("filtered_effort_A", data=fil_effort_A)
            obs_group.create_dataset("filtered_effort_Nm", data=fil_effort_Nm)
            
            if gravity_q is not None:
                obs_group.create_dataset("gravity_q", data=gravity_q)
                obs_group.create_dataset("residual_tau", data=residual)
            
            # Other observations
            for key, val in other_data.items():
                obs_group.create_dataset(key, data=val)
            
            # Copy attributes
            for k, v in attrs.items():
                fout.attrs[k] = v
            fout.attrs["preprocessed"] = True
            fout.attrs["effort_corrected_to_Nm"] = True
            if gravity_q is not None:
                fout.attrs["gravity_compensated"] = True
    
    return {
        "qpos": qpos,
        "qvel": qvel,
        "effort_A": effort_A,
        "effort_Nm": effort_Nm,
        "filtered_effort_Nm": fil_effort_Nm,
        "gravity_q": gravity_q,
        "residual": residual,
        "coeff": coeff,
    }


def main():
    ap = argparse.ArgumentParser(description="Preprocess effort data: A→Nm + gravity compensation")
    ap.add_argument(
        "episodes",
        nargs="+",
        help="HDF5 episode file(s) or glob pattern",
    )
    ap.add_argument(
        "--urdf",
        default="examples/piper_bimanual/urdf/piper/urdf/piper_description.urdf",
        help="Path to robot URDF",
    )
    ap.add_argument(
        "--kt-left",
        type=float,
        nargs=7,
        default=[0.3] * 6 + [0.1],
        help="Torque constant (Nm/A) for left arm (7 values: 6 joints + gripper)",
    )
    ap.add_argument(
        "--kt-right",
        type=float,
        nargs=7,
        default=[0.3] * 6 + [0.1],
        help="Torque constant (Nm/A) for right arm",
    )
    ap.add_argument(
        "--gear-ratio",
        type=float,
        nargs=7,
        default=[100] * 6 + [100],
        help="Gear ratio for each joint type (7 values)",
    )
    ap.add_argument(
        "--output-suffix",
        default="_preprocessed",
        help="Suffix to add to output files (before .hdf5)",
    )
    ap.add_argument(
        "--skip-gravity",
        action="store_true",
        help="Skip gravity compensation (e.g., if pinocchio unavailable)",
    )
    
    args = ap.parse_args()
    
    # Resolve glob patterns
    files = []
    for pattern in args.episodes:
        if "*" in pattern or "?" in pattern:
            files.extend(glob.glob(pattern))
        elif os.path.isfile(pattern):
            files.append(pattern)
    
    if not files:
        print("No files found!")
        return
    
    print(f"Found {len(files)} episode(s)")
    
    # Load model if gravity compensation requested
    model = None
    if not args.skip_gravity:
        if HAS_PINOCCHIO:
            try:
                model = pin.buildModelFromUrdf(args.urdf)
                lump_gripper_into_link6(model)
                print(f"URDF loaded: {model.nq} DOFs, {model.njoints} joints")
            except Exception as e:
                print(f"Failed to load URDF: {e}")
                print("Will skip gravity compensation")
        else:
            print("pinocchio not available, skipping gravity compensation")
    
    # Process each file
    kt_left = np.array(args.kt_left)
    kt_right = np.array(args.kt_right)
    gear_ratio = np.array(args.gear_ratio)
    
    for hdf5_file in files:
        output_file = hdf5_file.replace(".hdf5", f"{args.output_suffix}.hdf5")
        
        result = preprocess_episode(
            hdf5_file,
            kt_left=kt_left,
            kt_right=kt_right,
            gear_ratio=gear_ratio,
            model=model,
            output_path=output_file,
        )
        print()


if __name__ == "__main__":
    main()
