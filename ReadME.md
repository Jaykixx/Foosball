# Infos
Verwendet IsaacSim 2022.1.1 + OmniIsaacGymEnvs vom 01.11.2022 (Commit 62c9bbf)

## Tensorboard Command

### Privat
``
doskey PYTHON_PATH=C:\Users\Janosch\AppData\Local\ov\pkg\isaac_sim-2022.1.1\python.bat $*
``

``
PYTHON_PATH -m tensorboard.main --logdir runs/Foosball/summaries
``


### Leistungsrechner
``
doskey PYTHON_PATH=C:\Users\"Janosch Moos"\AppData\Local\ov\pkg\isaac_sim-2022.1.1\python.bat $*
``

``
PYTHON_PATH -m tensorboard.main --logdir runs/Foosball/summaries
``

``
PYTHON_PATH main.py task=FrankaPushCube num_envs=4 test=True headless=False checkpoint='runs\FrankaPushCube\nn\FrankaPushCube.pth'
``