# 🎮 A3C Ms. Pac-Man Agent

A Deep Reinforcement Learning agent trained to play **Ms. Pac-Man** using the **Asynchronous Advantage Actor-Critic (A3C)** algorithm.

![Gameplay Demo](recordings/gameplay.gif)

## 🚀 Features

- **A3C Algorithm** - Asynchronous training with multiple parallel environments
- **Optimized for 4GB GPU** - Efficient network architecture
- **Video Recording** - Save gameplay as GIF or MP4
- **Checkpoint System** - Auto-save best model and periodic checkpoints

## 📋 Requirements

```bash
pip install torch gymnasium ale-py imageio imageio-ffmpeg numpy tqdm opencv-python
```

## 🎯 Quick Start

### Training

```bash
# Train the agent (100,000 steps with 8 parallel environments)
python main.py --mode train --epochs 100000 --envs 8
```

### Demo (Live View)

```bash
# Watch the trained agent play
python main.py --mode demo --load-best
```

### Record Gameplay

```bash
# Save as GIF
python main.py --mode record --load-best --output recordings/gameplay.gif

# Save as MP4
python main.py --mode record --load-best --output recordings/gameplay.mp4 --max-steps 3000
```

## 🏗️ Project Structure

```
├── main.py                 # Entry point
├── Network/
│   ├── NN.py              # Neural network architecture
│   └── Agent.py           # A3C Agent with training logic
├── Environment/
│   └── Preprocess.py      # Atari frame preprocessing
├── utils/
│   ├── Training.py        # Training loop
│   ├── Render_model.py    # Demo and recording functions
│   ├── Make_Env.py        # Environment factory
│   └── Load_model.py      # Model loading utilities
├── models/                 # Saved model checkpoints
└── recordings/            # Gameplay recordings
```

## 🧠 Network Architecture

| Layer | Type | Output Shape |
|-------|------|--------------|
| Input | 4 stacked frames | (4, 42, 42) |
| Conv1 | Conv2d(4→32, k=4, s=2) | (32, 21, 21) |
| Conv2 | Conv2d(32→64, k=4, s=2) | (64, 11, 11) |
| Conv3 | Conv2d(64→64, k=3, s=1) | (64, 11, 11) |
| FC1 | Linear(7744→512) | 512 |
| FC2 | Linear(512→256) | 256 |
| Actor | Linear(256→9) | 9 actions |
| Critic | Linear(256→1) | 1 value |

## ⚙️ Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 3e-4 |
| Discount Factor (γ) | 0.99 |
| Entropy Coefficient | 0.01 |
| Gradient Clipping | 0.5 |
| Frame Stack | 4 |
| Frame Size | 42×42 |

## 📊 Training Progress

Models are saved automatically:
- `models/model_best.pth` - Highest reward achieved
- `models/model_step_*.pth` - Checkpoints every 10,000 steps
- `models/model_final.pth` - Final model after training

## 🎮 Actions

| ID | Action |
|----|--------|
| 0 | NOOP |
| 1 | UP |
| 2 | RIGHT |
| 3 | LEFT |
| 4 | DOWN |
| 5 | UPRIGHT |
| 6 | UPLEFT |
| 7 | DOWNRIGHT |
| 8 | DOWNLEFT |

## � Algorithm Comparison (Stable-Baselines3)

Compare custom A3C with industry-standard implementations:

### Install SB3

```bash
pip install stable-baselines3[extra]
```

### Train PPO or A2C

```bash
# Train with PPO
python train_sb3.py --mode train --algorithm PPO --timesteps 100000 --envs 8

# Train with A2C
python train_sb3.py --mode train --algorithm A2C --timesteps 100000 --envs 8
```

### Evaluate SB3 Models

```bash
# Demo mode
python train_sb3.py --mode demo --algorithm PPO

# Record gameplay
python train_sb3.py --mode record --algorithm PPO --output recordings/ppo_gameplay.gif
```

### Run Benchmark

```bash
# Compare all algorithms (Custom A3C, PPO, A2C)
python benchmark.py --episodes 10
```

This will output a comparison table like:

```
Algorithm            Mean        Std        Min        Max
------------------------------------------------------------
Custom_A3C           850.0      120.5      680.0     1100.0
SB3_PPO             1200.0      150.3      950.0     1450.0
SB3_A2C              980.0      130.2      780.0     1200.0
```

## �📚 References

- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) (Mnih et al., 2016)
- [OpenAI Gymnasium](https://gymnasium.farama.org/)
- [Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)

## 📝 License

MIT License
