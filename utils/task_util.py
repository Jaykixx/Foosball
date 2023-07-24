def initialize_task(config, env, init_sim=True):
    from environments.Foosball.foosball_blocking import FoosballBlockingTask
    from environments.Foosball.foosball_goal_shot import FoosballGoalShotTask
    from environments.Foosball.foosball import FoosballGame

    # Mappings from strings to environments
    task_map = {
        "FoosballBlocking": FoosballBlockingTask,
        "FoosballGoalShot": FoosballGoalShotTask,
        "Foosball": FoosballGame  # TODO: Check self play functionality in RLGames
    }

    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
    sim_config = SimConfig(config)

    cfg = sim_config.config
    task = task_map[cfg["task_name"]](
        name=cfg["task_name"], sim_config=sim_config, env=env
    )

    env.set_task(
        task=task,
        sim_params=sim_config.get_physics_params(),
        backend="torch",
        init_sim=init_sim
    )

    return task