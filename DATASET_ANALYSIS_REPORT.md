# 数据集分析报告

## 1. 数据集总体情况

**样本数据**: 6个 HDF5 episode 文件，每个约 14-16 秒，30 FPS
- 双臂机器人 (bimanual) 配置: 14 个自由度 (7 joint + 1 gripper × 2)
- 数据格式: `longvla_raw_v2` schema
- 任务: "hand the bottle of tea to me" (递茶瓶任务)

## 2. 检查结果总结

### ✓ 数据质量 - 良好
- **无 NaN/Inf 异常**: 所有数值数据都是有效的
- **数据完整性**: qpos, qvel, effort, filtered_effort, end_pose, 多相机图像都完整
- **结构一致**: 所有 episode 格式统一

### ⚠️ 发现的问题

#### A. **时间戳同步偏差** (需要注意)
```
drift joint_feedback_minus_image_ms: mean=±70ms std=70ms max=135ms
drift end_pose_minus_image_ms: mean=±310ms std=85ms max=370ms
```
- **影响**: 关节反馈与图像不完全同步，end_pose 延迟最大
- **原因**: 可能是不同传感器数据的异步采集或处理管道延迟
- **建议**: 训练时需要处理这种时间错配，可考虑对齐策略

#### B. **帧丢失** (轻微)
- 每个 episode 有 1-3 帧丢失 (总帧数的 0.3%-0.7%)
- **原因**: 网络抖动或处理延迟
- **影响**: 轻微，可接受

#### C. **左臂数据异常** (严重！)
```
Left arm joint range (ptp):   [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
Right arm joint range (ptp):  [0.24, 1.59, 1.34, 0.25, 1.97, 0.09, 0.04]
```
- **问题**: 左臂所有关节在整个 episode 中完全没有移动（qpos = 0）
- **可能原因**:
  1. 左臂未被激活/解锁
  2. 数据采集中左臂传感器故障
  3. 任务中左臂作为被控制对象而不是执行器
- **需要确认**: 这是否是预期行为？

#### D. **图像质量** (轻微问题)
- camera_l: 有 3-7 对"冻结"帧（连续帧内容基本相同）
- **原因**: 可能是相机输出帧率不足或某些时刻输出重复
- **影响**: 轻微

### E. **关节力矩 (Effort) 特性**

从 **static-frame 分析** (关节静止时的力矩):

```
Static-frame filtered_effort std (noise floor):
  L_j1-j6:  0.001-0.002  (非常小)
  L_grip:   0.001
  R_j1:     0.101  Nm
  R_j2:     1.114  Nm  ← 最大
  R_j3:     0.322  Nm
  R_j4:     0.083  Nm
  R_j5:     0.248  Nm
  R_j6:     0.042  Nm
  R_grip:   0.461  Nm
```

---

## 3. 关节力矩补偿分析

### 当前状态
- **数据格式**: 你的 `effort` 是**电流值**而非力矩值
- **已处理**: 数据包含 `filtered_effort` (已去噪)

### 补偿需求评估

#### A. **重力补偿 (Gravity Compensation)**

**需要！** 原因：
1. 右臂的静止力矩标准差很大（j2 有 1.1 Nm）
2. 这些不是噪声，而是重力和静摩擦的叠加
3. 使用 URDF 模型计算 `g(q)` 可以补偿这部分

**实现方式** (见 `validate_urdf_baseline.py`):
```python
# 1. 从 URDF 加载 Piper 机器人模型
# 2. 在每个关节配置 q 计算重力力矩 g(q)
# 3. residual_tau = measured_tau - g(q)
# 4. 使用 residual_tau 作为网络训练信号
```

#### B. **电流 → 力矩转换 (Current to Torque)**

**建议补偿步骤**:

```
1. 获取电机规格:
   - Piper 用什么电机？ (品牌/型号)
   - 每个关节的减速比是多少？
   
2. 建立转换公式:
   tau = kt × i × N  
   其中:
   - kt: 电机扭矩常数 (Nm/A)
   - i: 电流值 (A)  ← 你现在有的
   - N: 减速比
   
3. 应用转换 (在数据加载时做):
   effort_Nm = effort_A * kt * N
```

#### C. **静摩擦/库伦摩擦补偿**

从分析看，可能存在：
- **滞后摩擦 (Coulomb friction)**: 关节静止时的 1.1 Nm 可能包含这部分
- **粘性摩擦 (Viscous friction)**: 高速时线性增长

**建议**:
```
residual = measured_tau - g(q)
# 逐关节拟合摩擦模型:
# residual ≈ f_s * sign(qvel) + f_v * qvel
# 其中 f_s 是库伦摩擦，f_v 是粘性摩擦系数
```

---

## 4. 后续行动清单

### 立即需要做的 (必要)

- [ ] **确认左臂状态**: 
  - 左臂是否应该有运动？
  - 如果不应该，确认这些 episode 是否可用

- [ ] **获取电机参数**:
  - Piper 电机型号和扭矩常数 (kt)
  - 每个关节的减速比
  - 关节范围 (关节限制)

- [ ] **安装 pinocchio** (用于重力计算):
  ```bash
  uv pip install pinocchio
  ```

### 优化层面 (可选)

- [ ] 使用 `validate_urdf_baseline.py` 验证 URDF 中的惯性参数
- [ ] 实现静摩擦补偿
- [ ] 对齐时间戳用于多模态学习

---

## 5. 推荐的数据预处理流程

```python
import h5py
import numpy as np
import pinocchio as pin

def preprocess_episode(hdf5_path, model, kt_coeffs, gear_ratios):
    with h5py.File(hdf5_path, 'r') as f:
        qpos = f['observations/qpos'][...]  # (T, 14)
        qvel = f['observations/qvel'][...]
        effort_A = f['observations/effort'][...]  # 电流值 (A)
        fil_effort_A = f['observations/filtered_effort'][...]
    
    # 1. 电流 → 力矩转换
    effort_Nm = effort_A * (kt_coeffs[:, None] * gear_ratios[:, None])
    fil_effort_Nm = fil_effort_A * (kt_coeffs[:, None] * gear_ratios[:, None])
    
    # 2. 计算重力补偿
    data = model.createData()
    g_q = np.zeros_like(qpos)
    for t in range(len(qpos)):
        g_q[t] = pin.computeGeneralizedGravity(model, data, qpos[t])
    
    # 3. 去重力残差
    residual = fil_effort_Nm - g_q
    
    return {
        'qpos': qpos,
        'qvel': qvel,
        'effort_raw_Nm': effort_Nm,
        'effort_filtered_Nm': fil_effort_Nm,
        'gravity_estimate': g_q,
        'residual': residual,
    }
```

---

## 6. 问题和疑问

1. **左臂为什么完全没动？** → 需要确认
2. **电机规格在哪里？** → URDF中有吗？
3. **end_pose 的 310ms 延迟** → 是否需要对齐？
4. **期望的力矩范围是多少？** → 用于检验转换系数

---

## 总体评价

✅ **数据质量: 良好** - 无数据异常
⚠️ **需要补偿: 是的** - 尤其是重力和可能的摩擦
⚠️ **需要调查: 左臂数据** - 可能影响训练
💡 **建议**: 获取电机参数后立即实现电流→力矩转换和重力补偿
