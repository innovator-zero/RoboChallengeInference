import argparse
import logging
import sys

import cv2
import einops
import numpy as np

sys.path.insert(0, "openpi/src")

from openpi.policies import policy_config
from openpi.training import config as _config

from robot.interface_client import InterfaceClient
from robot.job_worker import job_loop

logging.basicConfig(
    filename="mylogfile.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
)

# Per-robot inference configuration.
# openpi_config: name passed to _config.get_config()
# action_type:   passed to get_state() and post_actions() — see README.md#robot-specific-notes
# image_type:    camera views requested from the robot
# image_mapping: maps robot camera names to openpi model input keys
ROBOT_CONFIGS = {
    "aloha": {
        "openpi_config": "pi05_RC_aloha",
        "action_type": "joint",       # dual-arm: no left/right prefix needed
        "image_type": ["high", "left_hand", "right_hand"],
        "image_mapping": {
            "high": "cam_high",
            "left_hand": "cam_left_wrist",
            "right_hand": "cam_right_wrist",
        },
    },
    "arx5": {
        "openpi_config": "pi05_RC_arx5",
        "action_type": "leftjoint",   # single-arm: must use left prefix
        "image_type": ["high", "left_hand", "right_hand"],
        "image_mapping": {
            "high": "cam_right_wrist",
            "left_hand": "cam_left_wrist",
            "right_hand": "cam_high",
        },
    },
    "franka": {
        "openpi_config": "pi05_RC_franka",
        "action_type": "leftpos",     # single-arm eef: must use left prefix
        "image_type": ["high", "left_hand", "right_hand"],
        "image_mapping": {
            "high": "cam_right_wrist",
            "left_hand": "cam_left_wrist",
            "right_hand": "cam_high",
        },
    },
    "ur5": {
        "openpi_config": "pi05_RC_ur5",
        "action_type": "leftpos",     # single-arm: must use left prefix; 2 cameras only
        "image_type": ["left_hand", "right_hand"],
        "image_mapping": {
            "left_hand": "cam_left_wrist",
            "right_hand": "cam_high",
        },
    },
}

# Task-specific prompts.
TASK_PROMPTS = {
    "stack_bowls": "stack the two smaller bowls on top of the largest bowl one by one.",
    "fold_dishcloth": "fold the dishcloth in half twice, then place it in the position slightly to the front and left",
    "move_objects_into_box": "place all the clutter on the desk into the white box",
}


class DummyPolicy:
    def __init__(self, checkpoint_path, robot, prompt, exec_horizon=50):
        cfg = ROBOT_CONFIGS[robot]
        if cfg["openpi_config"] is None:
            raise NotImplementedError(f"No openpi config defined for robot '{robot}'")
        train_config = _config.get_config(cfg["openpi_config"])
        self.policy = policy_config.create_trained_policy(train_config, checkpoint_path)
        self.prompt = prompt
        self.image_mapping = cfg["image_mapping"]
        self.exec_horizon = exec_horizon

    def decode(self, b: bytes):
        image = cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return einops.rearrange(image, "h w c -> c h w")

    def run_policy(self, input_data):
        images = {
            model_key: self.decode(input_data["images"][robot_key])
            for robot_key, model_key in self.image_mapping.items()
        }
        inputs = {
            "images": images,
            "prompt": self.prompt,
            "state": np.array(input_data["action"], dtype=np.float32),
        }
        action_chunk = self.policy.infer(inputs)["actions"][:self.exec_horizon]
        return np.array(action_chunk).tolist()


class GPUClient:
    def __init__(self, policy):
        self.policy = policy

    def infer(self, state):
        return self.policy.run_policy(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_token", type=str, required=True, help="User token")
    parser.add_argument(
        "--run_id", type=str, required=True,
        help="Run ID. Get it from the detail page of your submission",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument(
        "--robot", type=str, required=True, choices=list(ROBOT_CONFIGS),
        help="Robot type to run inference for",
    )
    parser.add_argument(
        "--task", type=str, required=True, choices=list(TASK_PROMPTS),
        help="Task name that determines the language prompt",
    )
    parser.add_argument("--exec_horizon", type=int, default=50, help="Number of actions to output per inference")

    args = parser.parse_args()

    robot_cfg = ROBOT_CONFIGS[args.robot]
    prompt = TASK_PROMPTS[args.task]
    image_size = [224, 224]
    image_type = robot_cfg["image_type"]
    action_type = robot_cfg["action_type"]
    duration = 0.05

    client = InterfaceClient(args.user_token)
    policy = DummyPolicy(args.checkpoint, args.robot, prompt, args.exec_horizon)
    gpu_client = GPUClient(policy)

    job_loop(client, gpu_client, args.run_id, image_size, image_type, action_type, duration)


if __name__ == "__main__":
    main()
