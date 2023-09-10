import numpy as np
import tensorflow as tf

def state_trans(state_tensor, soldiers_id_list):
    """
    扩展 global view 到每个士兵
    :param state_tensor:
    :param soldiers_id_list:
    :return:
    """
    global_view = tf.repeat(state_tensor[0], repeats=[len(soldiers_id_list)], axis=1)
    many_state_tensor = (global_view, state_tensor[1], state_tensor[2])
    return many_state_tensor

def EnvState2Agent(state):
    map_tensor, sight_tensor, sight_state_tensor = state
    map_tensor_tf = tf.convert_to_tensor(map_tensor, dtype=tf.float32)
    sight_tensor_tf = tf.convert_to_tensor(sight_tensor, dtype=tf.float32)
    sight_state_tensor_tf = tf.convert_to_tensor(sight_state_tensor, dtype=tf.float32)
    return map_tensor_tf, sight_tensor_tf, sight_state_tensor_tf


def EnvBatchState2Agent(states):
    map_tensors = []
    sight_tensors = []
    sight_state_tensors = []
    for state in states:
        if state:
            map_tensors.append(state[0])
            sight_tensors.append(state[1])
            sight_state_tensors.append(state[2])
    # 堆叠状态以形成批次数据
    batch_map_tensor = np.stack(map_tensors, axis=0)
    batch_sight_tensor = np.stack(sight_tensors, axis=0)
    batch_sight_state_tensor = np.stack(sight_state_tensors, axis=0)
    return batch_map_tensor, batch_sight_tensor, batch_sight_state_tensor


def MultiEnvData2AgentTrain(episode_states, episode_actions, episode_rewards, episode_next_states, episode_dones):
    # batch size datas
    global_map_view, soldier_view, soldier_state_view = [], [], []
    batch_actions = []
    batch_rewards = []
    next_global_map_view, next_soldier_view, next_soldier_state_view = [], [], []
    batch_dones = []

    # 遍历每一步
    for step_idx, step in enumerate(episode_states):
        # 遍历每一个仍然活跃的环境
        for active_env_idx, active_env in enumerate(step):
            # 拆解状态中的三个部分
            state1, state2, state3 = active_env
            # 添加到相应的批量数据列表
            global_map_view.append(state1)
            soldier_view.append(state2)
            soldier_state_view.append(state3)

            # 获取相应的动作、奖励、下一个状态和完成标志
            action = episode_actions[step_idx][active_env_idx]
            reward = episode_rewards[step_idx][active_env_idx]
            next_state1, next_state2, next_state3 = episode_next_states[step_idx][active_env_idx]
            done = episode_dones[step_idx][active_env_idx]

            # 添加到批量数据列表
            batch_actions.append(action)
            batch_rewards.append(reward)
            next_global_map_view.append(next_state1)
            next_soldier_view.append(next_state2)
            next_soldier_state_view.append(next_state3)
            batch_dones.append(1.0 if done else 0.0)

    # 转换为NumPy数组
    batch_actions = np.array(batch_actions)

    # 转换为Tensor
    global_map_view = tf.convert_to_tensor(global_map_view)
    soldier_view = tf.convert_to_tensor(soldier_view)
    soldier_state_view = tf.convert_to_tensor(soldier_state_view)
    batch_actions = tf.convert_to_tensor(batch_actions)
    batch_rewards = tf.constant(batch_rewards, shape=(len(batch_rewards), 1), dtype=tf.float32)
    next_global_map_view = tf.convert_to_tensor(next_global_map_view)
    next_soldier_view = tf.convert_to_tensor(next_soldier_view)
    next_soldier_state_view = tf.convert_to_tensor(next_soldier_state_view)
    batch_dones = tf.constant(batch_dones, shape=(len(batch_dones), 1), dtype=tf.float32)
    return (global_map_view, soldier_view, soldier_state_view), batch_actions, batch_rewards, \
        (next_global_map_view, next_soldier_view, next_soldier_state_view), batch_dones
