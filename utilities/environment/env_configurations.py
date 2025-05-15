from rl_games.common.env_configurations import *


def get_extended_env_info(env):
    result_shapes = {}
    result_shapes['observation_space'] = env.observation_space
    result_shapes['action_space'] = env.action_space
    result_shapes['agents'] = 1
    result_shapes['value_size'] = 1

    if hasattr(env, 'has_gripper'):
        result_shapes['has_gripper'] = env.has_gripper

    if hasattr(env, "joint_observation_space"):
        result_shapes['joint_observation_space'] = env.joint_observation_space
    if hasattr(env, "task_observation_space"):
        result_shapes['task_observation_space'] = env.task_observation_space
    if hasattr(env, "joint_space"):
        result_shapes['joint_space'] = env.joint_space

    if hasattr(env, "get_number_of_agents"):
        result_shapes['agents'] = env.get_number_of_agents()
    if hasattr(env, "value_size"):
        result_shapes['value_size'] = env.value_size
    if hasattr(env, "num_object_types"):
        result_shapes['num_object_types'] = env.num_object_types
    print(result_shapes)
    return result_shapes


