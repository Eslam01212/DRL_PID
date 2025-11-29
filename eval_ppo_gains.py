#!/usr/bin/env python3
# Minimal PPO to learn UGV2 gains (k_wall, k_dist) from LiDAR observations only.
# Training + evaluation entrypoint.
# Usage:
#   python3 eval_ppo_gains.py
#   python3 eval_ppo_gains.py --eval 60 ppo_gains_best.zip

import math, time, sys, os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList


from ppo_gains_logging import (
    RLNode,
    clamp,
    right_distance,
    rear_distance,
    init_csv,
    log_ppo_row,
    save_path_plot,
    SaveBestCallback,
)

# Torch feature extractor for 25-D LiDAR+state vector
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor




class LidarFeatureExtractor(BaseFeaturesExtractor):
    """Map 25-D observation vector (20 LiDAR + 5 scalars) to 64-D features."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        n_input = int(np.prod(observation_space.shape))
        self.fc = nn.Linear(n_input, features_dim)
        self.activation = nn.Tanh()
        self._features_dim = features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations shape: (batch, 25)
        if observations.dim() > 2:
            x = observations.view(observations.size(0), -1)
        else:
            x = observations
        return self.activation(self.fc(x))

class UGV2GainEnv(gym.Env):
    """
    Single-env wrapper that:
      - Talks to ROS2 through RLNode
      - Learns k_dist for UGV2 (front robot)
      - Keeps log buffers (xy1, xy2, vhist1, vhist2, IAE_wall, IAE_rear)
    """

    metadata = {"render_modes": []}

    def __init__(self, step_dt: float = 0.05):
        super().__init__()

        # ROS2 node + shared CSV header
        import rclpy
        rclpy.init(args=None)
        self.rclpy = rclpy
        self.node = RLNode()
        init_csv()

        # logging buffers
        self.v1 = 0.0
        self.v2 = 0.0
        self.xy1 = []
        self.vhist1 = []
        self.xy2 = []
        self.vhist2 = []
        self.IAE_wall = 0.0
        self.IAE_rear = 0.0

        self.step_dt = step_dt

        # targets & limits
        self.wall_target = 2.0
        self.follow_target = 2.0
        self.v2_base, self.vmax, self.wmax = 0.0, 0.8, 1.2

        # UGV1 (fixed controller, sinusoidal v1)
        self.k_wall_ugv1 = 2.0
        self.v1_base = 0.5
        self.v1_amp = self.v1_base
        self.v1_freq = 0.20  # Hz
        self.w1max = 1.2
        self.t0 = time.time()

        # actions: k_wall, k_dist in [0.1, 100]
        self.action_space = spaces.Box(
            low=np.array([-1], np.float32),
            high=np.array([1], np.float32),
            dtype=np.float32,
        )

        # observations: 20 LiDAR ranges + 5 scalars = 25-D vector
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(25,),
            dtype=np.float32,
        )
        self.last_v = 0.0
        self.last_w = 0.0
        self.prev_err = None
        self.max_steps = 1000
        self.step_count = 0

    # ---------- small internal helpers ----------
    def _spin(self, tmax: float = 0.02):
        """Spin ROS callbacks for at most tmax seconds."""
        t0 = time.time()
        while (time.time() - t0) < tmax:
            self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def _obs(self):
        """Build 25-D observation vector + return (d_right, rear) for reward."""
        self._spin(0.02)

        # 20-D LiDAR slice from UGV2 scan (evenly spaced beams over 360°)
        scan2 = self.node.scan2
        n_beams = 20

        if scan2 is not None and getattr(scan2, "ranges", None):
            ranges = np.asarray(scan2.ranges, dtype=np.float32)
            n = len(ranges)
            if n_beams >= n:
                idxs = np.arange(n, dtype=int)
            else:
                idxs = np.linspace(0, n - 1, num=n_beams, dtype=int)
            rm = scan2.range_max if scan2.range_max > 0.0 else 10.0
            lidar_vals = []
            for i in idxs:
                r = float(ranges[i])
                if not math.isfinite(r) or r <= 0.0:
                    r = rm
                lidar_vals.append(r)
            # pad / trim to exactly 20
            lidar20 = np.asarray(lidar_vals, dtype=np.float32)
            if lidar20.shape[0] < n_beams:
                lidar20 = np.pad(lidar20, (0, n_beams - lidar20.shape[0]), constant_values=rm)
            elif lidar20.shape[0] > n_beams:
                lidar20 = lidar20[:n_beams]
        else:
            rm = 10.0
            lidar20 = np.full((n_beams,), rm, dtype=np.float32)

        # scalar distances and previous commands
        rm2 = scan2.range_max if (scan2 is not None and scan2.range_max > 0.0) else 10.0
        d_right_ugv1 = right_distance(self.node.scan1)
        d_right_ugv2 = right_distance(scan2)
        rear = rear_distance(scan2)

        if d_right_ugv1 is None:
            d_right_ugv1 = rm2 
        if d_right_ugv2 is None:
            d_right_ugv2 = rm2 
        if rear is None:
            rear = rm2 

        extra = np.array(
            [d_right_ugv1, d_right_ugv2, rear, self.last_v, self.last_w],
            dtype=np.float32,
        )

        obs_vec = np.concatenate([lidar20, extra], axis=0).astype(np.float32)
        return obs_vec, (d_right_ugv2, rear)

    def _drive_ugv1(self):
        """Wall-following controller for UGV1: w from P-control, v sinusoidal."""
        d1 = right_distance(self.node.scan1)
        rm = self.node.scan1.range_max if self.node.scan1 else 10.0
        if d1 is None:
            d1 = rm 
        e1 = d1 - self.wall_target
        w1 = clamp(-self.k_wall_ugv1 * e1, -self.w1max, self.w1max)

        t = time.time() - self.t0
        self.v1 = self.v1_base + self.v1_amp * math.sin(2.0 * math.pi * self.v1_freq * t)
        self.v1 = clamp(self.v1, 0.0, self.vmax)

        self.node.send_cmd1(self.v1, w1)

    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Reset Gazebo / sim
        os.system(
            "ros2 service call /reset_simulation std_srvs/srv/Empty {} || "
            "ros2 service call /gazebo/reset_simulation std_srvs/srv/Empty {}"
        )

        self.step_count = 0
        self.prev_err = None
        self.IAE_wall = 0.0
        self.IAE_rear = 0.0

        self.node.send_cmd2(0.0, 0.0)
        self.node.send_cmd1(0.0, 0.0)
        time.sleep(0.1)
        self._spin(0.1)

        obs, _ = self._obs()
        return obs, {}

    def step(self, action):
        self.step_count += 1

        # --- UGV1 moves independently ---
        self._drive_ugv1()

        # --- UGV2 (learned gains) ---
        k_wall = 2
        k_dist = 1.0 + 0.5 * float(action[0])
        #k_dist = 1.5

        obs, (d_right, rear) = self._obs()
        rm = self.node.scan2.range_max if self.node.scan2 else 10.0
        if d_right is None:
            d_right = rm 
        if rear is None:
            rear = rm 

        e_wall = d_right - self.wall_target
        w = clamp(-k_wall * e_wall, -self.wmax, self.wmax)

        v = 0.0
        if rear <= rm:
            v = self.v2_base + k_dist * (self.follow_target - rear)
        self.v2 = clamp(v, 0.0, self.vmax)

        self.node.send_cmd2(self.v2, w)
        time.sleep(self.step_dt)

        self.last_v, self.last_w = self.v2, w

        # reward & termination
        err = abs(rear - self.follow_target)
        d_err = 0.0 if (self.prev_err is None) else (self.prev_err - err)
        self.prev_err = err
        reward = -(err) + 10.0 * d_err + 10.0

        terminated = (rear >= 3.0 or rear <= 0.5)
        truncated = self.step_count >= self.max_steps

        self.IAE_wall += abs(e_wall)
        self.IAE_rear += abs(rear - self.follow_target)

        # log per-step abs errors to shared CSV in PPO columns
        log_ppo_row(abs(e_wall), abs(rear - self.follow_target))

        # store path + speeds for later plotting (if odom available)
        if getattr(self.node, "p1", None) is not None:
            self.xy1.append(self.node.p1)
            self.vhist1.append(self.v1)
        if getattr(self.node, "p2", None) is not None:
            self.xy2.append(self.node.p2)
            self.vhist2.append(self.v2)

        info = {
            "IAE_wall": self.IAE_wall,
            "IAE_rear": self.IAE_rear,
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            self.node.send_cmd2(0.0, 0.0)
            self.node.send_cmd1(0.0, 0.0)
        except Exception:
            pass
        self.node.destroy_node()
        self.rclpy.shutdown()


if __name__ == "__main__":
    # Create env
    base_env = UGV2GainEnv(step_dt=0.05)
    env = Monitor(base_env)

    # ---- quick evaluation path ----
    if len(sys.argv) >= 2 and sys.argv[1] == "--eval":
        sec = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
        model_path = sys.argv[3] if len(sys.argv) > 3 else "ppo_gains_best.zip"
        model = PPO.load(model_path, env=env, device="auto")

        obs, _ = env.reset()
        t_end = time.time() + sec
        info = {}
        terminated = False
        while time.time() < t_end  :
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

        # final path plot
        try:
            print(
                f"IAE_wall={info.get('IAE_wall', 0.0)/max(sec,1.0)} "
                f"IAE_rear={info.get('IAE_rear', 0.0)/max(sec,1.0)}"
            )
            save_path_plot(
                base_env.xy1,
                base_env.vhist1,
                base_env.xy2,
                base_env.vhist2,
                fname="/home/ws/ugv_ws_clone/path_plot_eval.png",
                stride=60,
            )
        except Exception:
            pass

        env.close()
        sys.exit(0)

    # ---- training path ----
    policy_kwargs = dict(
        features_extractor_class=LidarFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=64),
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )

    os.makedirs("models", exist_ok=True)

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=128,
        batch_size=256,
        gamma=0.99,
        tensorboard_log="./tb",
    )
    #model = PPO.load("ppo_gains_best", env=env)   # <- continue from here

    # save best model + periodic checkpoints
    best_callback = SaveBestCallback(save_path="ppo_gains_best", verbose=1)
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,          # every 10k env steps
        save_path="./SavedModels",      # folder to save into
        name_prefix="ppo_gains",   # filenames: ppo_gains_XXXX_steps.zip
    )
    callback = CallbackList([best_callback, checkpoint_callback])

    model.learn(total_timesteps=10_000_000, tb_log_name="ugv2_gains", callback=callback)

    model.save("ppo_gains_last")
    env.close()