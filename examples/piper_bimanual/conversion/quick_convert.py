#!/usr/bin/env python3
"""
快速数据转换脚本 - 针对 Piper 人机交互任务

不需要复杂的命令行参数，预设了合理的默认值。
"""

from pathlib import Path
import subprocess
import sys


def main():
    print("=" * 60)
    print("Piper 数据集转换 - 快速开始")
    print("=" * 60)
    print()
    
    openpi_root = Path(__file__).parent.parent.parent.parent
    raw_dir = openpi_root / "examples" / "datasets"
    urdf_path = openpi_root / "examples" / "piper_bimanual" / "urdf" / "piper" / "urdf" / "piper_description.urdf"
    
    # 检查数据
    if not raw_dir.exists():
        print(f"❌ 找不到数据目录: {raw_dir}")
        return 1
    
    episodes = list(raw_dir.glob("episode_*.hdf5"))
    if not episodes:
        print(f"❌ 在 {raw_dir} 中没有找到 episode_*.hdf5 文件")
        return 1
    
    print(f"✓ 发现 {len(episodes)} 个数据 episode")
    print()
    
    # 选择转换方式
    print("选择转换方式:")
    print("  1. 完整转换 (推荐) - Current→Nm + 重力补偿")
    print("  2. 基础转换 - Current→Nm (无重力补偿)")
    print("  3. 原始数据 - 无任何转换 (用于调试)")
    print()
    
    choice = input("请选择 (1-3): ").strip()
    
    if choice == "1":
        repo_id = "piper_bimanual_gravity_compensated"
        repo_id = input(f"输入数据集名称 [{repo_id}]: ").strip() or repo_id
        
        print()
        print("执行转换...")
        cmd = [
            "uv", "run", "python",
            str(openpi_root / "examples" / "piper_bimanual" / "conversion" / "convert_longvla_hdf5_to_lerobot.py"),
            "--raw-dir", str(raw_dir),
            "--repo-id", repo_id,
            "--task", "human_handover",
            "--kt", "0.3", "0.3", "0.3", "0.3", "0.3", "0.3", "0.1",
            "--gear-ratio", "100", "100", "100", "100", "100", "100", "100",
            "--urdf", str(urdf_path),
        ]
        
    elif choice == "2":
        repo_id = "piper_bimanual_basic"
        repo_id = input(f"输入数据集名称 [{repo_id}]: ").strip() or repo_id
        
        print()
        print("执行转换...")
        cmd = [
            "uv", "run", "python",
            str(openpi_root / "examples" / "piper_bimanual" / "conversion" / "convert_longvla_hdf5_to_lerobot.py"),
            "--raw-dir", str(raw_dir),
            "--repo-id", repo_id,
            "--task", "human_handover",
            "--no-gravity-compensation",
            "--kt", "0.3", "0.3", "0.3", "0.3", "0.3", "0.3", "0.1",
            "--gear-ratio", "100", "100", "100", "100", "100", "100", "100",
        ]
        
    elif choice == "3":
        repo_id = "piper_bimanual_raw"
        repo_id = input(f"输入数据集名称 [{repo_id}]: ").strip() or repo_id
        
        print()
        print("执行转换...")
        cmd = [
            "uv", "run", "python",
            str(openpi_root / "examples" / "piper_bimanual" / "conversion" / "convert_longvla_hdf5_to_lerobot.py"),
            "--raw-dir", str(raw_dir),
            "--repo-id", repo_id,
            "--task", "human_handover",
            "--no-current-to-nm",
            "--no-gravity-compensation",
        ]
    else:
        print("❌ 无效选择")
        return 1
    
    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("=" * 60)
        print(f"✓ 转换完成! 数据集: {repo_id}")
        print("=" * 60)
        print()
        print("后续步骤:")
        print(f"  1. 验证数据集:")
        print(f"     uv run python -c \"from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; ds = LeRobotDataset('{repo_id}'); print(f'Episodes: {{ds.num_episodes}}, Frames: {{ds.num_frames}}')")
        print()
        print(f"  2. 查看样本:")
        print(f"     uv run python -c \"from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; ds = LeRobotDataset('{repo_id}'); frame = ds[0]; import numpy as np; print('Right effort:', np.array(frame['observation.right_interaction']))\"")
        print()
        print(f"  3. 开始训练:")
        print(f"     uv run python scripts/train.py --dataset-name {repo_id} ...")
        print()
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 转换失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
