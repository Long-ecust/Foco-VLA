import os
import cv2
import numpy as np
import pandas as pd
from datasets import Dataset


# =========================================================
# Video reader（frame-level）
# =========================================================
class VideoReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __getitem__(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            raise IndexError(idx)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def __len__(self):
        return self.length


# =========================================================
# LeRobot → OpenPI → HF builder
# =========================================================
class LeRobotToHFDatasetBuilder:
    def __init__(self, root_dir, horizon=16):
        self.root = root_dir
        self.horizon = horizon

        self.cam_map = {
            "observation.images.head_image": "top_head",
            "observation.images.right_wrist_image": "hand_right",
        }

        self.videos = {}

    # -----------------------------
    def _video(self, key):
        if key not in self.videos:
            path = os.path.join(
                self.root,
                f"videos/{key}/chunk-000/file-000.mp4"
            )
            self.videos[key] = VideoReader(path)
        return self.videos[key]

    # -----------------------------
    def _parquet(self):
        path = os.path.join(self.root, "data/chunk-000/file-000.parquet")
        return pd.read_parquet(path)

    # =====================================================
    def build(self):

        df = self._parquet()
        T = len(df)

        dataset = []

        for t in range(T - self.horizon):

            # ---------------- images ----------------
            images = {}

            for raw_k, openpi_k in self.cam_map.items():
                images[openpi_k] = self._video(raw_k)[t]

            # padding missing camera
            H, W = images["top_head"].shape[:2]
            images["hand_left"] = np.zeros((H, W, 3), dtype=np.uint8)

            # ---------------- state ----------------
            state = np.asarray(df["observation.state"].iloc[t], dtype=np.float32)

            # ---------------- actions ----------------
            actions = np.stack(
                df["action"].iloc[t:t + self.horizon]
            ).astype(np.float32)

            # ---------------- prompt ----------------
            prompt = (
                df["task"].iloc[t]
                if "task" in df.columns
                else "default"
            )

            dataset.append({
                "top_head": images["top_head"],
                "hand_right": images["hand_right"],
                "hand_left": images["hand_left"],
                "state": state,
                "actions": actions,
                "prompt": prompt,
            })

        return dataset


# =========================================================
# Save to HuggingFace Dataset
# =========================================================
def save_to_hf(dataset_list, out_dir):
    ds = Dataset.from_list(dataset_list)

    ds.save_to_disk(out_dir)
    print(f"[OK] saved HF dataset to: {out_dir}")


# =========================================================
# run
# =========================================================
if __name__ == "__main__":

    root = "/data/zhulin/shenxin/datasets/chip100_0321"
    out = "/data/zhulin/shenxin/datasets/chip100_0321_hf"

    builder = LeRobotToHFDatasetBuilder(root, horizon=16)

    data = builder.build()

    print("num samples:", len(data))

    save_to_hf(data, out)