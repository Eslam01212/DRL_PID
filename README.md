# DRL-Based Adaptive Spacing Gain Controller for Dual-Robot Wall Following

This repository contains the full implementation of the adaptive spacing-gain controller used in:

**Eslam Mohamed, Armando Sousa, Filipe Santos**  
*Minimalistic navigation for leader–follower robotic system using deep reinforcement learning for controller adaptation*,  
Journal of Automation, Mobile Robotics and Intelligent Systems (JAMRIS), 2024.

The project focuses on a **leader–follower (front–rear) robot pair** navigating a **wavy corridor**.  
Both robots follow the wall using a **simple P-controller**, and the *front robot* must additionally keep a safe spacing distance to the rear robot.  
A **PPO agent** observes compact LiDAR-based features and **adapts the spacing gain online**, improving formation stability without requiring any global localization, mapping, or inter-robot communication.  

---

## 📂 Repository Contents

| File | Description |
|------|-------------|
| **`ppo_gains_logging.py`** | ROS 2 node handling LiDAR processing, wall-following control, spacing estimation, IAE logging, and training callbacks. |
| **`eval_ppo_gains.py`** | Custom Gymnasium environment (`UGV2GainEnv`), feature extractor, PPO training pipeline, and evaluation script. |
| **`ppo_gains_*.zip`** | Trained PPO models (e.g., `ppo_gains_5.zip`, `ppo_gains_best.zip`). |

---

## 🚀 Features

- Fully reactive **LiDAR-only** navigation—no maps, GPS, SLAM, or communication needed.
- PPO agent outputs **1 scalar action** → mapped to spacing gain.
- Internal controller remains a simple **P-controller**.
- Custom reward function encouraging spacing stability and smoothness.
- Online logging of wall and spacing IAEs.
- Optional real-time path visualization.

---

## 🧰 Requirements

### ROS 2 + Simulation
- ROS 2 Humble (or equivalent)
- Gazebo world with `/ugv1/scan`, `/ugv2/scan`, `/ugv1/odom`, `/ugv2/odom`, `/ugv1/cmd_vel`, `/ugv2/cmd_vel`

### Python Dependencies

```
pip install stable-baselines3==2.0.0 gymnasium torch matplotlib
```

---

## 🏋️‍♂️ Training

```
python3 eval_ppo_gains.py
```

Outputs:
- Trained models → `SavedModels/`
- Best model → `ppo_gains_best.zip`
- TensorBoard logs → `./tb`

---

## 🎯 Evaluation

```
python3 eval_ppo_gains.py --eval 60 ppo_gains_best.zip
```

Produces:
- Time-normalized IAE
- `path_plot_eval.png`
---

---

## 📚 Citation

```
@article{mohamed2024minimalistic,
  title   = {Minimalistic Navigation for Leader--Follower Robotic System Using Deep Reinforcement Learning for Controller Adaptation},
  author  = {Mohamed, Eslam and Sousa, Armando and Santos, Filipe},
  journal = {Journal of Automation, Mobile Robotics and Intelligent Systems},
  year    = {2024}
}
```

---

## 👤 Author

**Eslam Mohamed**  
INESC TEC & FEUP — University of Porto  
Email: *eng.eslam.m.f@gmail.com*
