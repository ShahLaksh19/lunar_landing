# 🚀 Rocket Lander AI: Training with Reward Shaping

## Overview
This project trains an AI agent to land a rocket safely using Reinforcement Learning. The agent learns through repeated interactions with the Lunar Lander simulation.

## Technologies Used
- Python
- Google Colab
- Gymnasium
- Stable-Baselines3
- Proximal Policy Optimization (PPO)
- Matplotlib and NumPy

## Project Objectives
- Train a new AI agent to land a rocket.
- Understand how reinforcement learning works.
- Experiment with different reward functions.
- Study reward shaping and reward hacking.
- Evaluate and compare model performance.

## How It Works
The agent observes the rocket's position, velocity, angle, and landing-leg status. It chooses an action, receives a reward, and updates its policy through PPO training.

## Experiments
1. Original reward: baseline training.
2. Time penalty: penalize taking too long.
3. Hover bonus: investigate whether a poorly designed reward encourages unwanted behaviour.

## How to Run
1. Open `Rocket_Lander_AI_Jupyter.ipynb` in Google Colab.
2. Run the installation cell.
3. Execute the notebook cells in order.
4. Train the models and evaluate their performance.
5. Compare the results and discuss the effects of reward shaping.

## Results
Add your actual training graphs, evaluation scores, and observations here after running the experiments.

## Learning Outcomes
This project demonstrates reinforcement learning, PPO training, reward design, model evaluation, and the importance of aligning AI rewards with the intended goal.
