#!/usr/bin/env python3
"""
Republish Piper joint feedback with filtered effort values.

Input topics:
  - /left/joint_feedback
  - /right/joint_feedback

Output topics:
  - /left/joint_feedback_filtered
  - /right/joint_feedback_filtered
"""

import argparse
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class EffortFilterState:
    def __init__(self, joint_dim: int, median_window: int, ema_alpha: float) -> None:
        self.median_window = median_window
        self.ema_alpha = ema_alpha
        self.buffers = [deque(maxlen=median_window) for _ in range(joint_dim)]
        self.ema = np.zeros(joint_dim, dtype=np.float64)
        self.initialized = False

    def filter(self, effort: np.ndarray) -> np.ndarray:
        output = np.zeros_like(effort, dtype=np.float64)
        for idx, value in enumerate(effort):
            self.buffers[idx].append(float(value))
            median_value = float(np.median(np.asarray(self.buffers[idx], dtype=np.float64)))
            if not self.initialized:
                self.ema[idx] = median_value
            else:
                self.ema[idx] = self.ema_alpha * median_value + (1.0 - self.ema_alpha) * self.ema[idx]
            output[idx] = self.ema[idx]
        self.initialized = True
        return output.astype(np.float32)


class FilteredJointFeedbackPublisher(Node):
    def __init__(self, median_window: int, ema_alpha: float) -> None:
        super().__init__("filtered_joint_feedback_publisher")
        self.filters = {
            "left": EffortFilterState(7, median_window, ema_alpha),
            "right": EffortFilterState(7, median_window, ema_alpha),
        }
        self.pub_left = self.create_publisher(JointState, "/left/joint_feedback_filtered", 20)
        self.pub_right = self.create_publisher(JointState, "/right/joint_feedback_filtered", 20)
        self.create_subscription(JointState, "/left/joint_feedback", lambda msg: self._callback("left", msg), 50)
        self.create_subscription(JointState, "/right/joint_feedback", lambda msg: self._callback("right", msg), 50)

    def _callback(self, side: str, msg: JointState) -> None:
        effort = np.asarray(msg.effort[:7], dtype=np.float32)
        filtered_effort = self.filters[side].filter(effort)

        out = JointState()
        out.header = msg.header
        out.name = list(msg.name)
        out.position = list(msg.position)
        out.velocity = list(msg.velocity)
        out.effort = [float(x) for x in filtered_effort] + [float(x) for x in msg.effort[7:]]
        if side == "left":
            self.pub_left.publish(out)
        else:
            self.pub_right.publish(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Republish Piper joint feedback with filtered effort.")
    parser.add_argument("--median-window", type=int, default=5, help="Median filter window size.")
    parser.add_argument("--ema-alpha", type=float, default=0.2, help="EMA factor in (0, 1].")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.median_window < 1 or args.median_window % 2 == 0:
        raise ValueError("--median-window must be a positive odd integer")
    if not 0.0 < args.ema_alpha <= 1.0:
        raise ValueError("--ema-alpha must be in (0, 1]")

    rclpy.init()
    node = FilteredJointFeedbackPublisher(args.median_window, args.ema_alpha)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
