def initialize_task(config, env, init_sim=True):
    # Custom Environments
    from environments.foosball.foosball_blocking import FoosballBlockingTask
    from environments.foosball.foosball_scoring_incoming import FoosballScoringIncomingTask
    from environments.foosball.foosball_scoring_resting import FoosballScoringRestingTask
    from environments.foosball.foosball_scoring_resting_obstacles import FoosballScoringRestingObstacleTask
    from environments.foosball.foosball_selfplay import FoosballSelfPlay
    from environments.foosball.foosball_keeper_selfplay import FoosballKeeperSelfPlay
    from environments.foosball.foosball_mixed_selfplay import FoosballMixedSelfPlay

    # Mappings from strings to environments
    task_map = {
        # Custom Environments
        "FoosballBlocking": FoosballBlockingTask,
        "FoosballScoringIncoming": FoosballScoringIncomingTask,
        "FoosballScoringResting": FoosballScoringRestingTask,
        "FoosballScoringRestingObstacle": FoosballScoringRestingObstacleTask,
        "FoosballSelfPlay": FoosballSelfPlay,
        "FoosballKeeperSelfPlay": FoosballKeeperSelfPlay,
        "FoosballMixedSelfPlay": FoosballMixedSelfPlay
    }

    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
    sim_config = SimConfig(config)
    cfg = sim_config.config
    algo = cfg['train']['params']

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
