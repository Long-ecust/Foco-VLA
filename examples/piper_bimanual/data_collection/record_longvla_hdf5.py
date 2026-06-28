import argparse
from datetime import datetime
from pathlib import Path
import select
import sys
import termios
import tty

import h5py
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState

try:
    import rerun as rr
except ImportError:
    rr = None


IMAGE_TOPIC_MAP = {
    "/camera_f/color/image_raw": "camera_f",
    "/camera_l/color/image_raw": "camera_l",
    "/camera_r/color/image_raw": "camera_r",
}

RAW_FEEDBACK_TOPIC_MAP = {
    "/left/joint_feedback": "left",
    "/right/joint_feedback": "right",
}

FILTERED_FEEDBACK_TOPIC_MAP = {
    "/left/joint_feedback_filtered": "left",
    "/right/joint_feedback_filtered": "right",
}

END_POSE_TOPIC_MAP = {
    "/left/end_pose": "left",
    "/right/end_pose": "right",
}

ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
END_POSE_COMPONENT_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw"]


def input_ready() -> bool:
    return bool(select.select([sys.stdin], [], [], 0.0)[0])


class KeyboardListener:
    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self.fd)

    def __enter__(self) -> "KeyboardListener":
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old_settings)

    def poll_key(self) -> str | None:
        if not input_ready():
            return None
        return sys.stdin.read(1)


class RerunVisualizer:
    def __init__(self, *, app_id: str, spawn: bool) -> None:
        if rr is None:
            raise RuntimeError("Rerun is not installed. Install it first, for example: pip install rerun-sdk")

        rr.init(app_id, spawn=spawn)
        self.frame_index = 0

    def _scalar(self, value: float):
        if hasattr(rr, "Scalar"):
            return rr.Scalar(value)
        if hasattr(rr, "Scalars"):
            return rr.Scalars([value])
        return value

    def log_frame(
        self,
        *,
        frame_time: float,
        images: dict[str, np.ndarray],
        qpos: np.ndarray,
        qvel: np.ndarray,
        effort: np.ndarray,
        filtered_effort: np.ndarray,
        end_pose: np.ndarray,
    ) -> None:
        if hasattr(rr, "set_time_seconds"):
            rr.set_time_seconds("frame_time", frame_time)
        elif hasattr(rr, "set_time_sequence"):
            rr.set_time_sequence("frame_idx", self.frame_index)

        for cam_name, frame in images.items():
            rr.log(f"cameras/{cam_name}", rr.Image(frame[:, :, ::-1]))

        for arm_idx, arm_name in enumerate(("left", "right")):
            base = arm_idx * 7
            for joint_offset, joint_name in enumerate(ARM_JOINT_NAMES):
                idx = base + joint_offset
                rr.log(f"joints/{arm_name}/{joint_name}/qpos", self._scalar(float(qpos[idx])))
                rr.log(f"joints/{arm_name}/{joint_name}/qvel", self._scalar(float(qvel[idx])))
                rr.log(f"joints/{arm_name}/{joint_name}/effort", self._scalar(float(effort[idx])))
                rr.log(f"joints/{arm_name}/{joint_name}/filtered_effort", self._scalar(float(filtered_effort[idx])))
            pose_base = arm_idx * 7
            for component_offset, component_name in enumerate(END_POSE_COMPONENT_NAMES):
                rr.log(
                    f"end_pose/{arm_name}/{component_name}",
                    self._scalar(float(end_pose[pose_base + component_offset])),
                )
        self.frame_index += 1


class HDF5EpisodeWriter:
    def __init__(self, output_path: Path, image_shapes: dict[str, tuple[int, int, int]], fps: float, prompt: str) -> None:
        self.output_path = output_path
        self.h5 = h5py.File(output_path, "w")
        self.h5.attrs["schema_version"] = "longvla_raw_v2"
        self.h5.attrs["fps"] = fps
        self.h5.attrs["created_at"] = datetime.now().isoformat()
        self.h5.attrs["prompt"] = prompt
        self.h5.attrs["task"] = prompt

        obs = self.h5.create_group("observations")
        imgs = obs.create_group("images")
        self.image_dsets = {}
        for cam_name, shape in image_shapes.items():
            self.image_dsets[cam_name] = imgs.create_dataset(
                cam_name,
                shape=(0, *shape),
                maxshape=(None, *shape),
                chunks=(1, *shape),
                compression="gzip",
                dtype=np.uint8,
            )

        self.qpos = obs.create_dataset("qpos", shape=(0, 14), maxshape=(None, 14), chunks=(256, 14), compression="gzip", dtype=np.float32)
        self.qvel = obs.create_dataset("qvel", shape=(0, 14), maxshape=(None, 14), chunks=(256, 14), compression="gzip", dtype=np.float32)
        self.effort = obs.create_dataset("effort", shape=(0, 14), maxshape=(None, 14), chunks=(256, 14), compression="gzip", dtype=np.float32)
        self.filtered_effort = obs.create_dataset("filtered_effort", shape=(0, 14), maxshape=(None, 14), chunks=(256, 14), compression="gzip", dtype=np.float32)
        self.end_pose = obs.create_dataset("end_pose", shape=(0, 14), maxshape=(None, 14), chunks=(256, 14), compression="gzip", dtype=np.float32)

        ts = self.h5.create_group("timestamps")
        self.frame_time = ts.create_dataset("frame", shape=(0,), maxshape=(None,), chunks=(1024,), compression="gzip", dtype=np.float64)
        self.image_time = ts.create_dataset("image", shape=(0, 3), maxshape=(None, 3), chunks=(1024, 3), compression="gzip", dtype=np.float64)
        self.feedback_time = ts.create_dataset("joint_feedback", shape=(0, 2), maxshape=(None, 2), chunks=(1024, 2), compression="gzip", dtype=np.float64)
        self.filtered_feedback_time = ts.create_dataset("joint_feedback_filtered", shape=(0, 2), maxshape=(None, 2), chunks=(1024, 2), compression="gzip", dtype=np.float64)
        self.end_pose_time = ts.create_dataset("end_pose", shape=(0, 2), maxshape=(None, 2), chunks=(1024, 2), compression="gzip", dtype=np.float64)

        meta = self.h5.create_group("meta")
        meta.create_dataset("arm_joint_names", data=np.asarray(ARM_JOINT_NAMES, dtype="S16"))
        meta.create_dataset("camera_names", data=np.asarray(["camera_f", "camera_l", "camera_r"], dtype="S16"))
        meta.create_dataset("end_pose_components", data=np.asarray(END_POSE_COMPONENT_NAMES, dtype="S16"))
        self.length = 0

    def append(
        self,
        *,
        frame_time: float,
        images: dict[str, np.ndarray],
        qpos: np.ndarray,
        qvel: np.ndarray,
        effort: np.ndarray,
        filtered_effort: np.ndarray,
        end_pose: np.ndarray,
        image_time: np.ndarray,
        feedback_time: np.ndarray,
        filtered_feedback_time: np.ndarray,
        end_pose_time: np.ndarray,
    ) -> None:
        idx = self.length
        self.length += 1

        for dset in self.image_dsets.values():
            dset.resize((self.length, *dset.shape[1:]))
        for dset, width in [
            (self.qpos, 14),
            (self.qvel, 14),
            (self.effort, 14),
            (self.filtered_effort, 14),
            (self.end_pose, 14),
        ]:
            dset.resize((self.length, width))
        for dset, width in [
            (self.frame_time, None),
            (self.image_time, 3),
            (self.feedback_time, 2),
            (self.filtered_feedback_time, 2),
            (self.end_pose_time, 2),
        ]:
            if width is None:
                dset.resize((self.length,))
            else:
                dset.resize((self.length, width))

        for cam_name, frame in images.items():
            self.image_dsets[cam_name][idx] = frame
        self.qpos[idx] = qpos
        self.qvel[idx] = qvel
        self.effort[idx] = effort
        self.filtered_effort[idx] = filtered_effort
        self.end_pose[idx] = end_pose
        self.frame_time[idx] = frame_time
        self.image_time[idx] = image_time
        self.feedback_time[idx] = feedback_time
        self.filtered_feedback_time[idx] = filtered_feedback_time
        self.end_pose_time[idx] = end_pose_time

    def close(self) -> None:
        self.h5.attrs["num_frames"] = self.length
        self.h5.close()


class LongVLARawRecorder(Node):
    def __init__(
        self,
        *,
        image_topics: list[str],
        raw_feedback_topics: list[str],
        filtered_feedback_topics: list[str],
        end_pose_topics: list[str],
        output_dir: Path,
        fps: float,
        prompt: str,
        rerun_visualizer: RerunVisualizer | None = None,
    ) -> None:
        super().__init__("longvla_raw_hdf5_recorder")
        self.bridge = CvBridge()
        self.output_dir = output_dir
        self.fps = fps
        self.prompt = prompt
        self.rerun_visualizer = rerun_visualizer
        self.latest_images = {}
        # 原始反馈只作为附加保留，不再作为启动录制的前置条件。
        self.latest_joint_feedback = {}
        self.latest_filtered_feedback = {}
        self.latest_end_pose = {}
        self.recording = False
        self.writer = None
        self.status_frames = 0

        for topic in image_topics:
            self.create_subscription(Image, topic, lambda msg, t=topic: self._image_callback(t, msg), 10)
        for topic in raw_feedback_topics:
            self.create_subscription(JointState, topic, lambda msg, t=topic: self._joint_feedback_callback(t, msg), 50)
        for topic in filtered_feedback_topics:
            self.create_subscription(JointState, topic, lambda msg, t=topic: self._filtered_joint_feedback_callback(t, msg), 50)
        for topic in end_pose_topics:
            self.create_subscription(Pose, topic, lambda msg, t=topic: self._end_pose_callback(t, msg), 20)

        self.record_timer = self.create_timer(1.0 / fps, self._record_tick)
        self.status_timer = self.create_timer(2.0, self._log_status)

    def _ros_time(self, stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _image_callback(self, topic: str, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().error(f"Failed to convert image {topic}: {exc}")
            return
        self.latest_images[IMAGE_TOPIC_MAP[topic]] = (frame, self._ros_time(msg.header.stamp))

    def _to_fixed_arm_vector(self, values, length: int = 7) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[0] >= length:
            return array[:length]
        padded = np.zeros(length, dtype=np.float32)
        padded[: array.shape[0]] = array
        return padded

    def _joint_feedback_callback(self, topic: str, msg: JointState) -> None:
        side = RAW_FEEDBACK_TOPIC_MAP[topic]
        self.latest_joint_feedback[side] = (
            self._to_fixed_arm_vector(msg.position),
            self._to_fixed_arm_vector(msg.velocity),
            self._to_fixed_arm_vector(msg.effort),
            self._ros_time(msg.header.stamp),
        )

    def _filtered_joint_feedback_callback(self, topic: str, msg: JointState) -> None:
        side = FILTERED_FEEDBACK_TOPIC_MAP[topic]
        self.latest_filtered_feedback[side] = (
            self._to_fixed_arm_vector(msg.position),
            self._to_fixed_arm_vector(msg.velocity),
            self._to_fixed_arm_vector(msg.effort),
            self._ros_time(msg.header.stamp),
        )

    def _end_pose_callback(self, topic: str, msg: Pose) -> None:
        side = END_POSE_TOPIC_MAP[topic]
        pose_vec = np.asarray(
            [
                msg.position.x,
                msg.position.y,
                msg.position.z,
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ],
            dtype=np.float32,
        )
        now = self.get_clock().now().nanoseconds / 1e9
        self.latest_end_pose[side] = (pose_vec, now)

    def _ready(self) -> bool:
        return (
            len(self.latest_images) == 3
            and len(self.latest_filtered_feedback) == 2
            and len(self.latest_end_pose) == 2
        )

    def _missing_streams(self) -> list[str]:
        missing = []
        for topic, cam_name in IMAGE_TOPIC_MAP.items():
            if cam_name not in self.latest_images:
                missing.append(topic)
        for topic, side in FILTERED_FEEDBACK_TOPIC_MAP.items():
            if side not in self.latest_filtered_feedback:
                missing.append(topic)
        for topic, side in END_POSE_TOPIC_MAP.items():
            if side not in self.latest_end_pose:
                missing.append(topic)
        return missing

    def start_recording(self) -> bool:
        if not self._ready():
            missing = ", ".join(self._missing_streams())
            self.get_logger().warning(f"Cannot start recording yet. Missing streams: {missing}")
            return False
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"episode_{timestamp}.hdf5"
        image_shapes = {name: frame.shape for name, (frame, _) in self.latest_images.items()}
        self.writer = HDF5EpisodeWriter(output_path, image_shapes=image_shapes, fps=self.fps, prompt=self.prompt)
        self.recording = True
        self.status_frames = 0
        self.get_logger().info(f"Recording started -> {output_path}")
        return True

    def stop_recording(self, *, save: bool = True) -> None:
        self.recording = False
        if self.writer is not None:
            output_path = self.writer.output_path
            frame_count = self.writer.length
            self.writer.close()
            self.writer = None
            if save:
                self.get_logger().info(f"Recording stopped -> {output_path} ({frame_count} frames)")
            else:
                if output_path.exists():
                    output_path.unlink()
                self.get_logger().info(f"Recording discarded -> {output_path} ({frame_count} frames)")

    def _record_tick(self) -> None:
        if not self.recording or self.writer is None or not self._ready():
            return

        images = {k: v[0] for k, v in self.latest_images.items()}
        image_time = np.asarray([self.latest_images[name][1] for name in ["camera_f", "camera_l", "camera_r"]], dtype=np.float64)

        # 训练用观测优先取 filtered topic；如果原始反馈缺失，则 effort 也回退到滤波值。
        left_qpos, left_qvel, left_filtered_effort, left_ffb_t = self.latest_filtered_feedback["left"]
        right_qpos, right_qvel, right_filtered_effort, right_ffb_t = self.latest_filtered_feedback["right"]
        left_raw_feedback = self.latest_joint_feedback.get("left")
        right_raw_feedback = self.latest_joint_feedback.get("right")
        qpos = np.concatenate([left_qpos, right_qpos], axis=0)
        qvel = np.concatenate([left_qvel, right_qvel], axis=0)
        filtered_effort = np.concatenate([left_filtered_effort, right_filtered_effort], axis=0)
        left_end_pose, left_end_pose_t = self.latest_end_pose["left"]
        right_end_pose, right_end_pose_t = self.latest_end_pose["right"]
        end_pose = np.concatenate([left_end_pose, right_end_pose], axis=0)
        if left_raw_feedback is not None and right_raw_feedback is not None:
            _, _, left_effort, left_fb_t = left_raw_feedback
            _, _, right_effort, right_fb_t = right_raw_feedback
            effort = np.concatenate([left_effort, right_effort], axis=0)
            feedback_time = np.asarray([left_fb_t, right_fb_t], dtype=np.float64)
        else:
            effort = filtered_effort.copy()
            feedback_time = np.asarray([left_ffb_t, right_ffb_t], dtype=np.float64)

        frame_time = self.get_clock().now().nanoseconds / 1e9
        self.writer.append(
            frame_time=frame_time,
            images=images,
            qpos=qpos,
            qvel=qvel,
            effort=effort,
            filtered_effort=filtered_effort,
            end_pose=end_pose,
            image_time=image_time,
            feedback_time=feedback_time,
            filtered_feedback_time=np.asarray([left_ffb_t, right_ffb_t], dtype=np.float64),
            end_pose_time=np.asarray([left_end_pose_t, right_end_pose_t], dtype=np.float64),
        )
        if self.rerun_visualizer is not None:
            self.rerun_visualizer.log_frame(
                frame_time=frame_time,
                images=images,
                qpos=qpos,
                qvel=qvel,
                effort=effort,
                filtered_effort=filtered_effort,
                end_pose=end_pose,
            )
        self.status_frames += 1

    def _log_status(self) -> None:
        if self.recording:
            self.get_logger().info(f"Recording... frames={self.status_frames}, prompt={self.prompt}")
        else:
            self.get_logger().info(
                "Waiting to start. Ready streams: "
                f"images={len(self.latest_images)}/3, filtered_fb={len(self.latest_filtered_feedback)}/2, "
                f"end_pose={len(self.latest_end_pose)}/2, "
                f"raw_fb(optional)={len(self.latest_joint_feedback)}/2"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a full LongVLA raw episode into HDF5.")
    parser.add_argument("--image-topics", nargs="+", default=list(IMAGE_TOPIC_MAP.keys()))
    parser.add_argument("--raw-feedback-topics", nargs="+", default=list(RAW_FEEDBACK_TOPIC_MAP.keys()))
    parser.add_argument("--filtered-feedback-topics", nargs="+", default=list(FILTERED_FEEDBACK_TOPIC_MAP.keys()))
    parser.add_argument("--end-pose-topics", nargs="+", default=list(END_POSE_TOPIC_MAP.keys()))
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--output-dir", default="logs/longvla_raw_hdf5")
    parser.add_argument("--prompt", required=True, help="Task prompt saved into HDF5 and later reused for training.")
    parser.add_argument("--rerun", action="store_true", help="Stream images and state to Rerun while recording.")
    parser.add_argument("--rerun-app-id", default="openpi.longvla_recorder")
    parser.add_argument(
        "--rerun-no-spawn",
        action="store_true",
        help="Do not auto-launch the Rerun viewer. Use this if you connect to an existing viewer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rerun_visualizer = None
    if args.rerun:
        rerun_visualizer = RerunVisualizer(app_id=args.rerun_app_id, spawn=not args.rerun_no_spawn)

    rclpy.init()
    node = LongVLARawRecorder(
        image_topics=args.image_topics,
        raw_feedback_topics=args.raw_feedback_topics,
        filtered_feedback_topics=args.filtered_feedback_topics,
        end_pose_topics=args.end_pose_topics,
        output_dir=output_dir,
        fps=args.fps,
        prompt=args.prompt,
        rerun_visualizer=rerun_visualizer,
    )

    try:
        print("LongVLA raw HDF5 recorder is ready.")
        print("Keyboard controls: s=start/save, r=discard and restart, q=quit")
        with KeyboardListener() as keyboard:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
                key = keyboard.poll_key()
                if key is None:
                    continue
                key = key.lower()

                if key == "s":
                    if node.recording:
                        node.stop_recording(save=True)
                        print("Episode saved. Press 's' to start next episode, 'r' to start fresh, 'q' to quit.")
                    else:
                        if not node.start_recording():
                            print("Still waiting for required streams. Press 's' again after the missing topics appear.")
                        else:
                            print("Recording started. Press 's' to save, 'r' to discard/restart, 'q' to quit.")
                elif key == "r":
                    if node.recording:
                        node.stop_recording(save=False)
                    if not node.start_recording():
                        print("Still waiting for required streams. Press 'r' again after the missing topics appear.")
                    else:
                        print("Recording restarted. Press 's' to save, 'r' to discard/restart, 'q' to quit.")
                elif key == "q":
                    if node.recording:
                        node.stop_recording(save=True)
                    print("Recorder exiting.")
                    break
    except KeyboardInterrupt:
        if node.recording:
            node.stop_recording(save=True)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
