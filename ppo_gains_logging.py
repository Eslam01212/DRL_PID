import math, time, os, csv, warnings
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback

# ---------------------------------------------------------------------
# Generic helpers + ROS2 node + logging + plotting + callback
# ---------------------------------------------------------------------

CSV_PATH = "/home/ws/ugv_ws_clone/IAE_compare.csv"


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _finite(v):
    return math.isfinite(v) and v > 0.0


class RLNode(Node):
    """Minimal ROS2 node used by the PPO env."""

    def __init__(self):
        super().__init__("rl_gains_node")
        self.scan1 = None
        self.scan2 = None
        self.p1 = None
        self.p2 = None

        self.pub2 = self.create_publisher(Twist, "/ugv2/cmd_vel", 10)
        self.pub1 = self.create_publisher(Twist, "/ugv1/cmd_vel", 10)

        self.create_subscription(LaserScan, "/ugv1/scan", self.cb_s1, 10)
        self.create_subscription(LaserScan, "/ugv2/scan", self.cb_s2, 10)
        self.create_subscription(Odometry, "/ugv1/odom", self.cb_odom1, 10)
        self.create_subscription(Odometry, "/ugv2/odom", self.cb_odom2, 10)

    # --- callbacks ---
    def cb_s1(self, msg):
        self.scan1 = msg

    def cb_s2(self, msg):
        self.scan2 = msg

    def cb_odom1(self, msg: Odometry):
        self.p1 = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def cb_odom2(self, msg: Odometry):
        self.p2 = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    # --- cmd helpers ---
    def send_cmd2(self, v, w):
        m = Twist()
        m.linear.x = float(v)
        m.angular.z = float(w)
        self.pub2.publish(m)

    def send_cmd1(self, v, w):
        m = Twist()
        m.linear.x = float(v)
        m.angular.z = float(w)
        self.pub1.publish(m)


# ---------- LiDAR + image helpers ----------
def right_distance(scan: LaserScan, desired_angle=-math.pi / 3, window=4):
    if scan is None or not scan.ranges or scan.angle_increment == 0.0:
        return None
    idx = int(round((desired_angle - scan.angle_min) / scan.angle_increment))
    idx = max(0, min(idx, len(scan.ranges) - 1))
    i0, i1 = max(0, idx - window), min(len(scan.ranges) - 1, idx + window)
    vals = [scan.ranges[i] for i in range(i0, i1 + 1) if _finite(scan.ranges[i])]
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def rear_distance(scan: LaserScan, thr: float = 1.0):
    if scan is None or not scan.ranges or not getattr(scan, "intensities", None):
        return None
    n = len(scan.ranges)
    idxs = list(range(max(0, n - 360), n)) + list(range(0, min(360, n)))
    bright = [
        scan.ranges[i]
        for i in idxs
        if i < len(scan.intensities) and _finite(scan.ranges[i]) and scan.intensities[i] > thr
    ]
    return min(bright) if bright else None


# ---------- CSV helpers ----------
def init_csv():
    """Create the shared IAE CSV file with header if it does not exist."""
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "IAE_wall_total",          # from follow.py (baseline)
                    "IAE_rear_total",          # from follow.py (baseline)
                    "IAE_wall_total_Ours",     # from PPO eval
                    "IAE_rear_total_Ours",     # from PPO eval
                ]
            )


def log_ppo_row(iae_wall_step: float, iae_rear_step: float) -> None:
    """Append one row with PPO controller data in columns 3–4 (baseline cols empty)."""
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "",  # baseline columns left empty here
                "",
                iae_wall_step,
                iae_rear_step,
            ]
        )


# ---------- plotting ----------
def save_path_plot(
    xy1,
    vhist1,
    xy2,
    vhist2,
    fname="path_plot.png",
    stride: int = 60,
    base_area: float = 70.0,
    max_area: float = 900.0,
):
    """Path plot with speed-encoded circle radii for UGV1/UGV2."""
    if not (xy1 or xy2):
        return

    def area_list(vhist):
        if not vhist:
            return []
        v = np.abs(np.asarray(vhist, dtype=float))
        vmax = np.percentile(v, 95)  # robust scale (ignores big spikes)
        if vmax <= 1e-9:
            vmax = 1.0
        a = base_area + max_area * np.clip(v / vmax, 0, 1)
        return a.tolist()

    fig, ax = plt.subplots(figsize=(12, 6))
    xs, ys = [], []

    # --- UGV1 (green) ---
    if xy1:
        x1 = [p[0] for p in xy1]
        y1 = [p[1] for p in xy1]
        ax.plot(x1, y1, "-", lw=2, color="green", label="UGV1 path (back)")
        idx = range(0, len(x1), stride)
        A1 = area_list(vhist1)
        s1 = [A1[i] for i in idx] if A1 else []
        ax.scatter(
            [x1[i] for i in idx],
            [y1[i] for i in idx],
            s=s1,
            alpha=0.35,
            color="green",
            edgecolors="none",
            label="UGV1 speed ∝ radius",
            clip_on=True,
        )
        xs += x1
        ys += y1

    # --- UGV2 (blue) ---
    if xy2:
        x2 = [p[0] for p in xy2]
        y2 = [p[1] for p in xy2]
        ax.plot(x2, y2, "-", lw=2, color="blue", label="UGV2 path (front)")
        idx = range(0, len(x2), stride)
        A2 = area_list(vhist2)
        s2 = [A2[i] for i in idx] if A2 else []
        ax.scatter(
            [x2[i] for i in idx],
            [y2[i] for i in idx],
            s=s2,
            alpha=0.35,
            color="royalblue",
            edgecolors="none",
            label="UGV2 speed ∝ radius",
            clip_on=True,
        )
        xs += x2
        ys += y2

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls="--", alpha=0.25)
    ax.set_xlabel("x [m]", fontsize=14)
    ax.set_ylabel("y [m]", fontsize=14)

    if xs and ys:
        pad = 0.6
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

    ax.legend(loc="best", framealpha=0.95, fontsize=14)
    fig.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- SB3 callback ----------
class SaveBestCallback(BaseCallback):
    """When a new best episode reward is seen during training, save the model."""

    def __init__(self, save_path="ppo_gains_best", verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.best_reward = -np.inf

    def _on_step(self) -> bool:
        # 'infos' is a list (one per env) with episode stats when an episode ends
        infos = self.locals.get("infos")
        if infos is not None:
            for info in infos:
                ep_info = info.get("episode")
                if ep_info is not None:
                    ep_r = ep_info.get("r")
                    if ep_r is not None and ep_r > self.best_reward:
                        self.best_reward = ep_r
                        if self.verbose:
                            print(f"[SaveBestCallback] New best reward {ep_r:.3f}, saving model.")
                        self.model.save(self.save_path)
        return True

