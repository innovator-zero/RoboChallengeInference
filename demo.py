import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import einops
import numpy as np
from PIL import Image

sys.path.insert(0, "openpi/src")

from openpi.policies import policy_config
from openpi.training import config as _config

from robot.interface_client import InterfaceClient
from robot.job_worker import job_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
)

# Per-robot inference configuration.
# openpi_config: name passed to _config.get_config()
# action_type:   passed to get_state() and post_actions() — see README.md#robot-specific-notes
# image_type:    camera views requested from the robot
# image_mapping: maps robot camera names to openpi model input keys
# state_dim:     dimension of the robot state vector
ROBOT_CONFIGS = {
    "aloha": {
        "openpi_config": "pi05_RC_aloha",
        "action_type": "joint",  # dual-arm: no left/right prefix needed
        "image_type": ["cam_left_wrist", "cam_right_wrist", "cam_high"],
        "image_mapping": {
            "cam_left_wrist": "cam_left_wrist",
            "cam_right_wrist": "cam_right_wrist",
            "cam_high": "cam_high",
        },
        "state_dim": 14,
        "gripper_dims": [6, 13],
    },
    "arx5": {
        "openpi_config": "pi05_RC_arx5",
        "action_type": "leftjoint",  # single-arm: must use left prefix
        "image_type": ["cam_global", "cam_arm", "cam_side"],
        "image_mapping": {
            "cam_global": "cam_high",
            "cam_arm": "cam_left_wrist",
            "cam_side": "cam_right_wrist",
        },
        "state_dim": 7,
        "gripper_dims": [-1],
    },
    "w1": {
        "openpi_config": "pi05_RC_w1",
        "action_type": "joint",  # dual-arm: no left/right prefix needed
        "image_type": ["cam_left_wrist", "cam_right_wrist", "cam_high"],
        "image_mapping": {
            "cam_left_wrist": "cam_left_wrist",
            "cam_right_wrist": "cam_right_wrist",
            "cam_high": "cam_high",
        },
        "state_dim": 14,
        "gripper_dims": [6, 13],
    },
    "ur5": {
        "openpi_config": "pi05_RC_ur5",
        "action_type": "leftpos",  # single-arm: must use left prefix; 2 cameras only
        "image_type": ["cam_global", "cam_arm"],
        "image_mapping": {
            "cam_global": "cam_high",
            "cam_arm": "cam_left_wrist",
        },
        "state_dim": 8,
        "gripper_dims": [-1],
    },
}

class DummyPolicy:
    def __init__(self, checkpoint_path, robot, exec_horizon=50):
        cfg = ROBOT_CONFIGS[robot]
        if cfg["openpi_config"] is None:
            raise NotImplementedError(f"No openpi config defined for robot '{robot}'")
        train_config = _config.get_config(cfg["openpi_config"])
        self.policy = policy_config.create_trained_policy(train_config, checkpoint_path)
        self.image_mapping = cfg["image_mapping"]
        self.state_dim = cfg["state_dim"]
        self.gripper_dims = cfg["gripper_dims"]
        self.exec_horizon = exec_horizon

    def warmup(self):
        logging.info("Warming up policy with random input...")
        dummy_image = np.zeros((3, 224, 224), dtype=np.uint8)
        inputs = {
            "images": {model_key: dummy_image for model_key in self.image_mapping.values()},
            "prompt": "",
            "state": np.zeros(self.state_dim, dtype=np.float32),
        }
        self.policy.infer(inputs)
        logging.info("Warmup complete.")

    def resize_with_pad(self, image: np.ndarray, height: int, width: int) -> np.ndarray:
        cur_height, cur_width = image.shape[:2]
        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        pil_image = Image.fromarray(image)
        resized_image = pil_image.resize((resized_width, resized_height), resample=Image.BILINEAR)
        zero_image = Image.new(resized_image.mode, (width, height), 0)
        pad_height = max(0, int((height - resized_height) / 2))
        pad_width = max(0, int((width - resized_width) / 2))
        zero_image.paste(resized_image, (pad_width, pad_height))
        return np.array(zero_image)

    def decode(self, b: bytes, height: int = 224, width: int = 224):
        image = cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.resize_with_pad(image, height, width)
        return einops.rearrange(image, "h w c -> c h w")

    def _validate_input_data(self, input_data):
        if not input_data:
            raise ValueError("input_data is empty")
        if not isinstance(input_data, dict):
            raise ValueError(f"input_data must be a dict, got {type(input_data).__name__}")

        missing_keys = [key for key in ("images", "action") if key not in input_data]
        if missing_keys:
            raise ValueError(f"input_data missing required keys: {missing_keys}")

        images = input_data["images"]
        if not images:
            raise ValueError("input_data['images'] is empty")
        if not isinstance(images, dict):
            raise ValueError(f"input_data['images'] must be a dict, got {type(images).__name__}")

        missing_cameras = [camera_name for camera_name in self.image_mapping if camera_name not in images]
        if missing_cameras:
            raise ValueError(f"input_data['images'] missing cameras: {missing_cameras}")

        action = input_data["action"]
        if action is None:
            raise ValueError("input_data['action'] is None")
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.size == 0:
            raise ValueError("input_data['action'] is empty")
        if action_array.shape[-1] != self.state_dim:
            raise ValueError(
                f"input_data['action'] has shape {action_array.shape}, expected last dimension {self.state_dim}"
            )
        return images, action_array

    def run_policy(self, input_data, prompt=None):
        if prompt is None:
            raise ValueError("prompt must be passed to GPUClient.infer")
        input_images, action_array = self._validate_input_data(input_data)
        images = {
            model_key: self.decode(input_images[robot_key])
            for robot_key, model_key in self.image_mapping.items()
        }
        inputs = {
            "images": images,
            "prompt": prompt,
            "state": action_array,
        }
        action_chunk = self.policy.infer(inputs)["actions"][: self.exec_horizon]
        actions = np.array(action_chunk)
        for dim in self.gripper_dims:
            gripper = actions[:, dim]
            actions[:, dim] = np.where(gripper < 0.02, 0, gripper)
        return actions.tolist()


class VideoRecorder:
    def __init__(self, log_dir: Path, prefix: str, fps: float = 20.0):
        self.log_dir = log_dir
        self.prefix = prefix
        self.fps = fps
        self._writers: dict = {}

    def _get_writer(self, camera_name, frame_hw):
        if camera_name not in self._writers:
            h, w = frame_hw
            path = str(self.log_dir / f"{self.prefix}_{camera_name}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
            if not writer.isOpened():
                logging.warning(f"VideoWriter failed to open for camera '{camera_name}' at {path}")
            else:
                logging.info(f"Recording camera '{camera_name}' to {path}")
            self._writers[camera_name] = writer
        return self._writers[camera_name]

    def record(self, images_bytes):
        for camera_name, png_bytes in images_bytes.items():
            frame = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                logging.warning(f"Failed to decode frame for camera '{camera_name}'")
                continue
            writer = self._get_writer(camera_name, frame.shape[:2])
            if writer.isOpened():
                writer.write(frame)

    def close(self):
        for camera_name, writer in self._writers.items():
            if writer.isOpened():
                writer.release()
                logging.info(f"Closed video writer for camera '{camera_name}'")
        self._writers.clear()


class GPUClient:
    def __init__(self, policy, recorder=None):
        self.policy = policy
        self.recorder = recorder

    def infer(self, state, prompt=None):
        if self.recorder is not None and "images" in state:
            self.recorder.record(state["images"])
        return self.policy.run_policy(state, prompt=prompt)

    def close(self):
        if self.recorder is not None:
            self.recorder.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--user_token", type=str, required=True, help="User token")
    # parser.add_argument('--run_id', type=str, required=True, help='Run ID. Get it from the detail page of your submission')
    parser.add_argument('--submission_id', type=str, required=True, help='Submission ID. Get it from the detail page of your submission')
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument(
        "--robot",
        type=str,
        required=True,
        choices=list(ROBOT_CONFIGS),
        help="Robot type to run inference for",
    )
    parser.add_argument("--exec_horizon", type=int, default=50, help="Number of actions to output per inference")

    args = parser.parse_args()

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{timestamp}_{args.robot}.log"
    logging.getLogger().addHandler(logging.FileHandler(log_file))
    logging.info(f"Logging to {log_file}")

    robot_cfg = ROBOT_CONFIGS[args.robot]
    image_size = [320, 240]
    image_type = robot_cfg["image_type"]
    action_type = robot_cfg["action_type"]
    duration = 0.05

    client = InterfaceClient(args.user_token)
    policy = DummyPolicy(args.checkpoint, args.robot, args.exec_horizon)
    policy.warmup()
    recorder = VideoRecorder(log_dir, f"{timestamp}_{args.robot}", fps=1.0)
    gpu_client = GPUClient(policy, recorder)

    try:
        job_loop(client, gpu_client, args.submission_id, image_size, image_type, action_type, duration)
    finally:
        gpu_client.close()


if __name__ == "__main__":
    main()
