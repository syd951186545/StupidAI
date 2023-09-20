import random
from collections import deque

import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field

from Algorithm.PPOAgent import PPOAgent
from Algorithm.SampleAgent import SampleAgent
from Environment.Agent2Json import parse_agent_actions
from Environment.Json2Agent import parse_env_state
from Environment.Logger import logger
from Environment.GameEnv import SoldierGameEnv
import config

random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)


class ReplayBufferQue:
    """DQN的经验回放池，每次采样batch_size个样本"""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def push(self, transitions):
        """_summary_
        Args:
            transitions (tuple): _description_
            (state, action, tf.constant(reward), next_state, tf.constant(done)
        """
        self.buffer.append(transitions)

    def sample(self):
        buffer = list(self.buffer)
        return buffer

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


replay_buffer = ReplayBufferQue(config.Training.trajectory_nums)


@dataclass
class Trajectory:
    state: tuple = field(default=None)
    action: tf.Tensor = field(default=None)
    next_state: tuple = field(default=None)
    reward: tf.Tensor = field(default=None)
    done: tf.Tensor = field(default=None)


def visi(agent):
    env = SoldierGameEnv()
    env.visible = True
    state_json_dict, soldier_reward, done = env.reset()
    state_tensor, soldiers_id_list = parse_env_state(state_json_dict)

    while env.cur_round < config.Training.MAX_ROUNDS and len(env.teamEnemy) != 0:
        action = agent.get_max_action(state_tensor)
        actions_json_dict = parse_agent_actions(action, soldiers_id_list)
        (next_state_json_dict, soldier_reward, done), can_not_move_soldier_ids, our_atted_enemies = env.step(
            actions_json_dict)
        next_state_tensor, next_soldiers_id_list = parse_env_state(next_state_json_dict, can_not_move_soldier_ids,
                                                                   our_atted_enemies)
        # 更新 新一轮环境
        state_tensor, soldiers_id_list = next_state_tensor, next_soldiers_id_list  # 士兵最后的结束状态是下步士兵的开始状态


def collect_data(env, agent):
    trajectories = []
    # 初始化环境，获得环境中的初始状态
    state_json_dict, soldier_reward, done = env.reset()
    state_tensor, soldiers_id_list = parse_env_state(state_json_dict)
    tra = Trajectory()
    tra.state = tf.squeeze(state_tensor, axis=0)

    while env.cur_round < config.Training.MAX_ROUNDS and len(env.teamEnemy) != 0:
        if config.Training.sample_agent:
            action = agent.get_max_action(state_tensor)
        elif config.Training.sample_type == 'categorical':
            action = agent.get_action(state_tensor)
        else:
            action = agent.get_max_action(state_tensor)
        # actions = agent.act(action_prob)
        actions_json_dict = parse_agent_actions(action, soldiers_id_list)
        tra.action = tf.constant([action])  # 保存该状态时执行的动作概率prob
        (next_state_json_dict, soldier_reward, done), can_not_move_soldier_ids, our_atted_enemies = env.step(
            actions_json_dict)
        next_state_tensor, next_soldiers_id_list = parse_env_state(next_state_json_dict, can_not_move_soldier_ids,
                                                                   our_atted_enemies)
        # 保存动作 、回报值 和 done
        for i, soldier in enumerate(soldiers_id_list):
            if soldier:
                id, _, _ = soldier
                if id in soldier_reward:
                    tra.reward = tf.constant([soldier_reward[id]])
        tra.done = tf.constant(done)

        # 保存下一个该状态
        tra.next_state = tf.squeeze(next_state_tensor, axis=0)

        # 更新 新一轮环境
        state_tensor, soldiers_id_list = next_state_tensor, next_soldiers_id_list  # 士兵最后的结束状态是下步士兵的开始状态
        trajectories.append(tra)
        # 保存新一轮tra
        del tra
        tra = Trajectory()
        tra.state = tf.squeeze(state_tensor, axis=0)
    replay_buffer.clear()
    for tra in trajectories:
        replay_buffer.push((tra.state, tra.action, tra.next_state, tra.reward, tra.done))
    return replay_buffer


def Start():
    env = SoldierGameEnv()
    agent = PPOAgent()
    # agent.load_models()
    if config.Training.sample_agent:
        sample_agent = SampleAgent()
        __TrainLoop(env, sample_agent, agent)
    else:
        __TrainLoop(env, None, agent)


# 创建数据生成器
def data_generator(tra_datasets):
    for item in tra_datasets.sample():
        yield item


def __TrainLoop(env, sample_agent, agent):
    for episode in range(config.Training.EPOCH):
        # 每一个回合开始都要重置隐藏状态
        if sample_agent:
            tra_datasets = collect_data(env, sample_agent)
        else:
            tra_datasets = collect_data(env, agent)

        # 创建数据集
        dataset = tf.data.Dataset.from_generator(lambda: data_generator(tra_datasets), output_signature=(
            tf.TensorSpec(shape=(24, 24, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(1), dtype=tf.int32),
            tf.TensorSpec(shape=(24, 24, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(1), dtype=tf.float32),
            tf.TensorSpec(shape=(1), dtype=tf.bool)
        )).batch(config.Training.batch_size, drop_remainder=False)

        # 100个batch size实际是画面时间步长
        for batch in dataset:
            batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = batch
            # batch_states = tuple([tf.transpose(s, [1, 0, 2, 3, 4]) for s in batch_states])
            # batch_actions = tf.transpose(batch_actions, [1, 0, 2])
            # batch_next_states = tuple([tf.transpose(s, [1, 0, 2, 3, 4]) for s in batch_next_states])
            # batch_rewards = tf.transpose(batch_rewards, [1, 0, 2])
            # batch_dones = tf.transpose(batch_dones, [1, 0, 2])
            # batch = (batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones)
            agent.update(episode, batch_inputs=batch)
            avg_reward = tf.reduce_mean(batch_rewards)
            with config.Training.training_tf_writer.as_default():
                tf.summary.scalar("Average_Reward", avg_reward, step=episode)
            logger.info(f"Episode: {episode}, Average Reward: {avg_reward}")
            negative_values = tf.boolean_mask(batch_rewards, batch_rewards < 0)
            logger.info(f"Episode: {episode}, negative values: {negative_values}")
        if episode % 10 == 0:
            agent.save_model_ckpt(episode // 10)
        if episode % 100 ==0:
            visi(agent)



if __name__ == '__main__':
    Start()
