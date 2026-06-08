# Smart Traffic Signal Control with Deep Reinforcement Learning

An intelligent traffic signal optimization system that uses **Deep Q-Networks (DQN)** with custom observation and reward functions to minimize vehicle waiting times — with built-in **emergency vehicle priority** (ambulances). Built on top of [SUMO](https://eclipse.dev/sumo/) and the [`sumo-rl`](https://github.com/LucasAlegre/sumo-rl) framework.

---

## Overview

Traditional traffic signals operate on fixed-time cycles, often leading to unnecessary congestion. This project replaces that with a **DQN-based reinforcement learning agent** that learns optimal signal switching strategies by observing real-time traffic conditions — including queue lengths, lane densities, and per-vehicle waiting times.

### Key Features

- **Custom Observation Function** — Injects per-lane maximum waiting times into the agent's state space alongside phase, density, and queue data.
- **Exponential Priority Reward** — Uses an exponential penalty curve (`wait^1.5`) to aggressively punish long waits; emergency vehicles receive quadratic penalties (`wait^2`) for even stronger prioritization.
- **Yellow Phase Forgiveness** — Divides penalties by 100× during yellow phases so the agent isn't afraid to switch signals.
- **Emergency Vehicle Handling** — 60 ambulances are injected into the simulation (one every 60 seconds), and the reward function explicitly prioritizes clearing them.
- **Tuned DQN Hyperparameters** — 50% exploration fraction, 2000-step warmup, and target network updates every 500 steps for stable learning.

---

## Project Structure

```
traffic-rl-optimization/
│
├── src/
│   ├── train.py            # Training pipeline (50,000 steps DQN + GUI test)
│   └── evaluate.py         # Load saved model & run with SUMO GUI + metrics
│
├── nets/
│   ├── intersection.net.xml  # SUMO network file (4-way intersection)
│   ├── routes.rou.xml        # Regular vehicle routes (~1000 vehicles/hour)
│   └── ambulances.rou.xml    # Emergency vehicle routes (60 ambulances)
│
├── models/
│   ├── dqn_traffic_model.zip       # Pre-trained DQN model
│   └── dqn_results_conn0_ep*.csv   # Per-episode training metrics
│
├── README.md
├── .gitignore
└── requirements.txt
```

---

## Prerequisites

- **Python** 3.10+
- **SUMO** (Simulation of Urban Mobility) — [Installation Guide](https://sumo.dlr.de/docs/Installing/index.html)
  - Make sure the `SUMO_HOME` environment variable is set
- **GPU (optional)** — PyTorch with CUDA for faster training

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/smart-traffic-rl.git
cd smart-traffic-rl
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Agent

```bash
cd src
python train.py
```

This will:
1. Train a DQN agent for **50,000 timesteps** (headless, no GUI)
2. Save the model to `models/dqn_traffic_model.zip`
3. Launch the SUMO GUI to **watch the trained agent** control the intersection

### 5. Evaluate a Pre-trained Model

```bash
cd src
python evaluate.py
```

This loads the saved model and runs inference with the SUMO GUI, printing performance metrics at the end:
- Total vehicles cleared (throughput)
- Average stop time per vehicle
- Average/maximum queue length
- Cumulative reward score

---

## How It Works

### Observation Space

The agent observes a feature vector containing:

| Feature | Description |
|---------|-------------|
| Phase ID | One-hot encoding of the current green phase |
| Min Green | Whether minimum green duration has elapsed |
| Lane Density | Normalized vehicle density per lane |
| Lane Queue | Normalized queue length per lane |
| **Wait Times** | **Max accumulated waiting time per lane** (normalized by 100) |

### Reward Function

```
reward = total_penalty / 1000 (to tackle exploding gradient problem)

where for each vehicle:
  - Regular vehicle:  penalty -= wait_time^1.5
  - Emergency vehicle: penalty -= wait_time^2.0

if yellow phase:
  total_penalty /= 100  (forgiveness factor, required otherwise agent won't switch phase)
```

### DQN Configuration

| Hyperparameter | Value |
|----------------|-------|
| Algorithm | DQN (MlpPolicy) |
| Learning Rate | 5e-4 |
| Learning Starts | 2,000 steps |
| Train Frequency | Every step |
| Target Update Interval | 500 steps |
| Exploration Fraction | 50% |
| Final Exploration Rate | 5% |
| Total Training Steps | 50,000 |

---

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Simulation Duration | 3,600 seconds (1 hour) |
| Yellow Phase Duration | 5 seconds |
| Minimum Green Time | 10 seconds |
| Decision Interval | 15 seconds |
| Regular Vehicle Spawn Rate | ~1 vehicle every 3.6 seconds |
| Ambulance Spawn Rate | 1 every 60 seconds |

---

## Results

After training for 50,000 steps (~200+ episodes), the agent was evaluated on a full 1-hour simulation with ~1,000 regular vehicles and 60 ambulances:

| Metric | Value |
|--------|-------|
| Total AI Decisions Made | 240 |
| Cumulative Reward Score | -316.28 |
| Total Vehicles Cleared (Throughput) | 1,041 vehicles |
| Average Stop Time per Vehicle | 9.22 seconds |
| Average Stopped Cars per Cycle | 7.5 cars |
| Maximum Gridlock (Worst-case queue) | 14 cars |

The agent successfully cleared over **1,000 vehicles** in a single hour while keeping the average stop time under **10 seconds** and preventing severe gridlock — the worst-case queue never exceeded 14 cars at any point during the simulation.

---

## Dependencies

| Package | Version |
|---------|---------|
| Python | 3.10+ |
| gymnasium | 1.2.3 |
| sumo-rl | 1.4.5 |
| stable-baselines3 | 2.8.0 |
| torch | 2.12.0 |
| numpy | 2.4.6 |
| pandas | 3.0.3 |
| SUMO | 1.27.0 |

---

## License

This project is for educational and research purposes.

---

## Acknowledgements

- [Eclipse SUMO](https://eclipse.dev/sumo/) — Traffic simulation platform
- [sumo-rl](https://github.com/LucasAlegre/sumo-rl) — Gymnasium-compatible RL environment for SUMO
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) — RL algorithm implementations
