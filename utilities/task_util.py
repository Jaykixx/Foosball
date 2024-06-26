def initialize_task(config, env, init_sim=True):
    # Custom Environments
    from environments.Foosball.foosball_blocking import FoosballBlockingTask
    from environments.Foosball.foosball_scoring_incoming import FoosballScoringIncomingTask
    from environments.Foosball.foosball_scoring_resting import FoosballScoringRestingTask
    from environments.Foosball.foosball_scoring_resting_obstacles import FoosballScoringRestingObstacleTask
    from environments.Foosball.foosball_selfplay import FoosballSelfPlay
    from environments.Foosball.foosball_keeper_selfplay import FoosballKeeperSelfPlay
    from environments.Foosball.foosball_tracking import FoosballTrackingTask
    # Default Environments
    from omniisaacgymenvs.tasks.allegro_hand import AllegroHandTask
    from omniisaacgymenvs.tasks.ant import AntLocomotionTask
    from omniisaacgymenvs.tasks.anymal import AnymalTask
    from omniisaacgymenvs.tasks.anymal_terrain import AnymalTerrainTask
    from omniisaacgymenvs.tasks.ball_balance import BallBalanceTask
    from omniisaacgymenvs.tasks.cartpole import CartpoleTask
    from omniisaacgymenvs.tasks.factory.factory_task_nut_bolt_pick import FactoryTaskNutBoltPick
    from omniisaacgymenvs.tasks.franka_cabinet import FrankaCabinetTask
    from omniisaacgymenvs.tasks.humanoid import HumanoidLocomotionTask
    from omniisaacgymenvs.tasks.ingenuity import IngenuityTask
    from omniisaacgymenvs.tasks.quadcopter import QuadcopterTask
    from omniisaacgymenvs.tasks.shadow_hand import ShadowHandTask
    from omniisaacgymenvs.tasks.crazyflie import CrazyflieTask

    # Mappings from strings to environments
    task_map = {
        # Custom Environments
        "FoosballTracking": FoosballTrackingTask,
        "FoosballBlocking": FoosballBlockingTask,
        "FoosballScoringIncoming": FoosballScoringIncomingTask,
        "FoosballScoringResting": FoosballScoringRestingTask,
        "FoosballScoringRestingObstacle": FoosballScoringRestingObstacleTask,
        "FoosballSelfPlay": FoosballSelfPlay,
        "FoosballKeeperSelfPlay": FoosballKeeperSelfPlay,
        # Default Environments
        "AllegroHand": AllegroHandTask,
        "Ant": AntLocomotionTask,
        "Anymal": AnymalTask,
        "AnymalTerrain": AnymalTerrainTask,
        "BallBalance": BallBalanceTask,
        "Cartpole": CartpoleTask,
        "FactoryTaskNutBoltPick": FactoryTaskNutBoltPick,
        "FrankaCabinet": FrankaCabinetTask,
        "Humanoid": HumanoidLocomotionTask,
        "Ingenuity": IngenuityTask,
        "Quadcopter": QuadcopterTask,
        "Crazyflie": CrazyflieTask,
        "ShadowHand": ShadowHandTask,
        "ShadowHandOpenAI_FF": ShadowHandTask,
        "ShadowHandOpenAI_LSTM": ShadowHandTask
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
