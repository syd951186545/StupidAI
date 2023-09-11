import time
import random

import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field

from Environment.PPO import PPOAgent
from Environment.Agent2Json import parse_agent_actions
from Environment.Json2Agent import parse_env_json
from Environment.logger import logger
from Environment.GameEnv import SoldierGameEnv
import config
from Environment.DataProcessing import state_trans

random.seed(123)
np.random.seed(123)


@dataclass
class Trajectory:
    state: tuple = field(default=None)
    action: tf.Tensor = field(default=None)
    next_state: tuple = field(default=None)
    reward: tf.Tensor = field(default=None)
    done: tf.Tensor = field(default=None)


def collect_data(env, agent):
    trajectories = []
    replay_buffer = []
    # 初始化环境，获得环境中的初始状态
    state_json_dict, soldier_reward, done = env.reset()
    state_tensor, soldiers_id_list = parse_env_json(state_json_dict, False)
    state_tensor = state_trans(state_tensor, soldiers_id_list)
    tra = Trajectory()
    tra.state = tuple([tf.squeeze(s, axis=0) for s in state_tensor])

    while not all(done):
        num_soldiers = state_tensor[0].shape[1]
        if config.Training.sample_type == 'categorical':
            action = agent.get_action(state_tensor)
        else:
            action = agent.get_max_action(state_tensor)
        # actions = agent.act(action_prob)
        actions_json_dict = parse_agent_actions(action, soldiers_id_list)
        tra.action = tf.constant(action)  # 保存该状态时执行的动作概率prob
        next_state_json_dict, soldier_reward, done = env.step(actions_json_dict)
        next_state_tensor, next_soldiers_id_list = parse_env_json(next_state_json_dict, False)
        # 保存动作回报值
        soldier_action_rewards = np.zeros([num_soldiers, 1], dtype=np.float32)
        for i, soldier in enumerate(soldiers_id_list):
            if soldier:
                id, _, _ = soldier
                if id in soldier_reward:
                    soldier_action_rewards[i, 0] = soldier_reward[id]
        tra.reward = tf.constant(soldier_action_rewards)
        _done = tf.constant(done)
        _done = tf.expand_dims(done, axis=1)
        tra.done = tf.constant(_done)

        # 下一个该状态
        next_global_view = tf.repeat(next_state_tensor[0], repeats=[len(soldiers_id_list)], axis=1)
        next_state_tensor = (next_global_view, next_state_tensor[1], next_state_tensor[2])
        tra.next_state = tuple([tf.squeeze(s, axis=0) for s in next_state_tensor])

        # 保存未阵亡士兵回合trajectory
        # 更新 新一轮环境
        state_tensor, soldiers_id_list = next_state_tensor, next_soldiers_id_list  # 士兵最后的结束状态是下步士兵的开始状态
        trajectories.append(tra)
        # 保存新一轮tra
        del tra
        tra = Trajectory()
        tra.state = tuple([tf.squeeze(s, axis=0) for s in state_tensor])

    for tra in trajectories:
        replay_buffer.append((tra.state, tra.action, tra.next_state, tra.reward, tra.done))
    return replay_buffer


def Start():
    env = SoldierGameEnv()
    agent = PPOAgent()
    # agent.load_model()
    if config.Training.sample_agent:
        sample_agent = PPOAgent()
        __TrainLoop(env, sample_agent, agent)
    else:
        __TrainLoop(env, None, agent)


# 创建数据生成器
def data_generator(tra_datasets):
    for item in tra_datasets:
        yield item


def __TrainLoop(env, sample_agent, agent):
    for episode in range(config.Training.EPOCH):
        # 每一个回合开始都要重置隐藏状态
        if sample_agent:
            sample_agent.actor_model.reset_hidden_states()
            tra_datasets = collect_data(env, sample_agent)
        else:
            tra_datasets = collect_data(env, agent)

        # 创建数据集
        dataset = tf.data.Dataset.from_generator(lambda: data_generator(tra_datasets), output_signature=(
            (tf.TensorSpec(shape=(None, 21, 21, 3), dtype=tf.float32),
             tf.TensorSpec(shape=(None, 21, 21, 4), dtype=tf.float32),
             tf.TensorSpec(shape=(None, 21, 21, 3), dtype=tf.float32)),

            tf.TensorSpec(shape=(None, 1), dtype=tf.int32),

            (tf.TensorSpec(shape=(None, 21, 21, 3), dtype=tf.float32),
             tf.TensorSpec(shape=(None, 21, 21, 4), dtype=tf.float32),
             tf.TensorSpec(shape=(None, 21, 21, 3), dtype=tf.float32)),

            tf.TensorSpec(shape=(None, 1), dtype=tf.float32),

            tf.TensorSpec(shape=(None, 1), dtype=tf.bool)
        )).batch(100, drop_remainder=False)

        # 100个batch size实际是画面时间步长
        for batch in dataset:
            batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = batch
            batch_states = tuple([tf.transpose(s, [1, 0, 2, 3, 4]) for s in batch_states])
            batch_actions = tf.transpose(batch_actions, [1, 0, 2])
            batch_next_states = tuple([tf.transpose(s, [1, 0, 2, 3, 4]) for s in batch_next_states])
            batch_rewards = tf.transpose(batch_rewards, [1, 0, 2])
            batch_dones = tf.transpose(batch_dones, [1, 0, 2])
            batch = (batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones)

            agent.update(episode, batch_inputs=batch)
            avg_reward = tf.reduce_mean(batch_rewards)
            with config.Training.training_tf_writer.as_default():
                tf.summary.scalar("Average_Reward", avg_reward, step=episode)
            logger.info(f"Episode: {episode}, Average Reward: {avg_reward}")
            negative_values = tf.boolean_mask(batch_rewards, batch_rewards < 0)
            logger.info(f"Episode: {episode}, negative values: {negative_values}")
        if episode % 10 == 0:
            agent.save_model_ckpt(episode // 10)
            env.visible = False
            if sample_agent:
                sample_agent.load_model_ckpt(training=True)
        if episode % 50 == 0:
            env.visible = True


if __name__ == '__main__':
    Start()
