#!/usr/bin/env python3
"""检查数据集的结构和内容"""
import h5py
import numpy as np
from pathlib import Path

dataset_dir = Path('/home/kplnb049/openpi/examples/datasets')
hdf5_files = sorted(dataset_dir.glob('*.hdf5'))

print(f"发现 {len(hdf5_files)} 个数据集文件\n")

for hdf5_file in hdf5_files:
    print(f"{'='*60}")
    print(f"文件: {hdf5_file.name}")
    print(f"{'='*60}")
    
    try:
        with h5py.File(hdf5_file, 'r') as f:
            print(f"文件大小: {hdf5_file.stat().st_size / 1024 / 1024:.2f} MB")
            print(f"\n数据集结构:")
            
            def print_structure(name, obj):
                indent = "  " * (name.count('/') - 1)
                if isinstance(obj, h5py.Dataset):
                    print(f"{indent}├─ {Path(name).name}: shape={obj.shape}, dtype={obj.dtype}")
                else:
                    print(f"{indent}├─ {Path(name).name}/")
            
            f.visititems(print_structure)
            
            # 检查关键数据
            print(f"\n关键数据检查:")
            
            # 查找所有数据集
            datasets_info = {}
            
            def collect_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets_info[name] = {
                        'shape': obj.shape,
                        'dtype': obj.dtype,
                        'min': float(np.min(obj[:])) if obj.size > 0 else None,
                        'max': float(np.max(obj[:])) if obj.size > 0 else None,
                        'mean': float(np.mean(obj[:])) if obj.size > 0 else None,
                    }
            
            f.visititems(collect_datasets)
            
            for name, info in sorted(datasets_info.items()):
                print(f"\n  {name}:")
                print(f"    Shape: {info['shape']}, Dtype: {info['dtype']}")
                if info['min'] is not None:
                    print(f"    Range: [{info['min']:.4f}, {info['max']:.4f}]")
                    print(f"    Mean: {info['mean']:.4f}")
    
    except Exception as e:
        print(f"错误: {e}")
    
    print()
