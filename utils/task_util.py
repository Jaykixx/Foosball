def initialize_task(config, env, init_sim=True):
    from environments.Foosball.foosball_camera import FoosballCameraTask
    from environments.Foosball.base import FoosballTask

    # Mappings from strings to environments
    task_map = {
        "FoosballCamera": FoosballCameraTask,
        "Foosball": FoosballTask
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