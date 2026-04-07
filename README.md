# HelixDesk OpenEnv

HelixDesk OpenEnv is a Gymnasium-compatible RL environment built for the Meta PyTorch OpenEnv Hackathon where an agent manages a realistic customer email queue — classifying, prioritising, routing, and preventing complaint surges.

## Formulations

- **State (42-dim):** email features, queue state, team loads, SLA pressure, trend signals, time, episode progress.
- **Action (MultiDiscrete [3,4,6,3]):** classify, prioritise, assign, secondary.
- **Reward:** 12 signals, clipped to [-1, 1].

## Quick Start
```bash
pip install -r requirements.txt
pip install -e .
python train.py --agent rule --episodes 100
python evaluate.py --agent rule
```

## Hackathon Grade
```bash
python grade.py --episodes 20
```

## Docker
```bash
docker build -t helixdesk-openenv:latest .
docker run helixdesk-openenv:latest
```

## RL Configuration
Uses OpenEnv spec. You can train via Stable-Baselines3:
`python train.py --agent sb3 --episodes 50`
