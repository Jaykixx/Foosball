# Isaac Sim Foosball RL Environment
This repository implements a Foosball RL Environment for [Isaac Sim 2023.1.1](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html). The project builds upon the [OmniIsaacGymEnvs Project](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/tree/release/2023.1.1). Prior versions of Isaac Sim are viable, however they might require
code changes due to deprecation in newer versions of OmniIsaacGymEnvs. 

This project provides a base of several training scenarios for Foosball:
- Ball Tracking (White Keeper)
- Blocking (White Keeper)
- Scoring on incoming ball (White Keeper)
- Scoring on resting ball (White Keeper)
- Scoring on resting ball with obstacles (White Keeper + Stationary Opponents)
- Keeper vs. Keeper with Self Play
- Full Game with Self Play

## Installation
Please follow the installation instructions for [OmniIsaacGymEnvs](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/tree/release/2023.1.1) to set up the base environment.

Afterward clone this repository:
```bash
git clone https://github.com/Jaykixx/Foosball.git
```

## Usage
To run the various training scenarios locate the python executable in Isaac Sim as described in the installation tutorial for [OmniIsaacGymEnvs](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/tree/release/2023.1.1).

Navigate to the project folder, then run:
```bash
PYTHON_PATH main.py task=FoosballBlocking
```
The execution is otherwise identical to OmniIsaacGymEnvs. For more Information see [Link](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/tree/release/2023.1.1?tab=readme-ov-file#running-the-examples). Checkpoints will be saved in ``<path_to_project>/runs/<task_name>/nn/``.

The task names for the various scenarios are:
- FoosballTracking
- FoosballBlocking
- FoosballScoringIncoming
- FoosballScoringResting
- FoosballScoringRestingObstacles
- FoosballKeeperSelfPlay
- FoosballSelfPlay

The full list of settings can be found in ``<path_to_project>/cfg/``.

For tensorboard run:
```bash
PYTHON_PATH -m tensorboard.main --logdir <path_to_project>/runs/<task_name>/summaries
```

## Cite
TBD