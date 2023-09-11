import os
import numpy as np
import tensorflow as tf
from datetime import datetime

import config
from Algorithm.ActionModel import SoldierActionModel
from Algorithm.CriticModel import SoldierCriticModel
from Environment.logger import logger


class PPOAgent:
    actor_model: SoldierActionModel
    critic_model: SoldierActionModel

    def __init__(self, _actor_model=SoldierActionModel(), _critic_model=SoldierCriticModel(), competition=False):
        # 初始化模型、优化器和其他参数
        self.actor_model = _actor_model  # 行动者模型（用于选择动作）
        self.critic_model = _critic_model  # 评价者模型（用于评估状态价值）
        self.actor_optimizer = config.Training.actor_optimizer  # 行动者的优化器
        self.critic_optimizer = config.Training.critic_optimizer  # 评价者的优化器
        self.actor_model.compile(optimizer=self.actor_optimizer)
        self.critic_model.compile(optimizer=self.critic_optimizer, loss=tf.keras.losses.MeanSquaredError())
        self.clip_epsilon = config.Training.clip_epsilon  # PPO算法的裁剪系数
        self.epoch = config.Training.epoch
        self.current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        # 如果 load_model 参数为 True，则从文件加载模型,否则训练状态
        if competition:
            # 加载比赛用模型
            self.load_model()

        # 创建检查点管理
        self.actor_checkpoint = tf.train.Checkpoint(optimizer=self.actor_optimizer, model=self.actor_model)
        self.critic_checkpoint = tf.train.Checkpoint(optimizer=self.actor_optimizer, model=self.actor_model)
        self.actor_checkpoint_dir = os.path.join(config.Training.TRAIN_MODELS_PATH, "actor_model")
        self.critic_checkpoint_dir = os.path.join(config.Training.TRAIN_MODELS_PATH, "critic_model")

    def load_model(self):
        # 从指定路径加载行动者和评价者模型
        logger.debug("agent loading ...")
        try:
            actor_checkpoint = tf.train.Checkpoint(optimizer=self.actor_optimizer, model=self.actor_model)
            actor_manager = tf.train.CheckpointManager(actor_checkpoint, directory=config.Predict.PREDICT_MODELS_PATH,
                                                       max_to_keep=5)
            actor_manager.restore_or_initialize()
        except Exception as e:
            logger.info(f"{str(e)}")
            raise Exception(f"加载模型异常")
        logger.debug('agent 模型已加载。')

    def save_model_ckpt(self, epoch):
        # 保存行动者和评价者模型到指定路径 step_counter、checkpoint_interval参数使用方式待研究
        try:
            actor_manager = tf.train.CheckpointManager(
                self.actor_checkpoint, directory=self.actor_checkpoint_dir,
                max_to_keep=5)
            critic_manager = tf.train.CheckpointManager(
                self.critic_checkpoint, directory=self.critic_checkpoint_dir,
                max_to_keep=5)
            actor_manager.save(epoch)
            critic_manager.save(epoch)
        except Exception as e:
            raise Exception(f"保存检查点异常错误: {e}")
        logger.debug('agent 模型已保存。')

    def load_model_ckpt(self, training=True):
        # 从指定路径加载行动者和评价者模型
        try:
            actor_manager = tf.train.CheckpointManager(
                self.actor_checkpoint, directory=self.actor_checkpoint_dir, max_to_keep=10)
            critic_manager = tf.train.CheckpointManager(
                self.critic_checkpoint, directory=self.critic_checkpoint_dir, max_to_keep=10)
            logger.info(f"加载最新检查点{actor_manager.checkpoints}")
            logger.info(f"加载最新检查点{critic_manager.checkpoints}")
            if training:
                actor_manager.restore_or_initialize()
                critic_manager.restore_or_initialize()
            else:
                actor_manager.restore_or_initialize().expect_partial()
                critic_manager.restore_or_initialize().expect_partial()
        except Exception as e:
            raise Exception(f"加载检查点异常错误:{e}")
        logger.debug('agent 模型已加载')

    @staticmethod
    def scalar_record(step=None, *args):
        if step is None:
            raise Exception("step illegal")
        with config.Training.training_tf_writer.as_default():  # 设置 SummaryWriter 为默认
            # 记录数据
            tf.summary.scalar("Actor Loss", args[0].numpy(), step=step)
            tf.summary.scalar("Critic Loss", args[1][0], step=step)

    def train(self, episode, batch_inputs):
        """
         手动循环，训练模型，梯度累计平均更新
        :param episode:
        :param batch_inputs:
        :return:
        """
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = batch_inputs
        batch_size = batch_dones.shape[0]
        accumulated_grads = [tf.zeros_like(var) for var in self.actor_model.trainable_variables]
        for t in range(batch_size):
            # 提取每一帧的数据，手动循环以更新隐藏状态
            state = tuple([s[t:t + 1] for s in batch_states])  # 注意：保持维度
            action = batch_actions[t:t + 1]
            next_state = tuple([s[t:t + 1] for s in batch_next_states])
            reward = batch_rewards[t:t + 1]
            done = batch_dones[t:t + 1]
            _input = (state, action, next_state, reward, done)
            grads = self.update(t, batch_inputs=_input)
            accumulated_grads = [acc_grad + grad for acc_grad, grad in zip(accumulated_grads, grads)]

        averaged_grads = [grad / batch_size for grad in accumulated_grads]
        # 应用累积梯度
        self.actor_optimizer.apply_gradients(zip(averaged_grads, self.actor_model.trainable_variables))

    def update(self, episode, batch_inputs):
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = batch_inputs
        batch_size, num_agents = batch_actions.shape[0], batch_actions.shape[1]
        # 训练模型

        batch_dones = tf.cast(batch_dones, dtype=tf.float32)

        values = self.critic_model.predict(batch_states)
        # 计算 target advantages
        next_values = self.critic_model.predict(batch_next_states)
        next_values *= (1.0 - batch_dones)  # Zero out the values for terminal states
        target = batch_rewards + config.Training.gamma * next_values
        delta = target - values
        advantages = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        # GAE 计算动作优势值
        advantage = tf.zeros(shape=(num_agents, 1))
        for t in reversed(range(delta.shape[0])):
            advantage = delta[t] + config.Training.gamma * config.Training.lambda_gae * advantage
            advantages = advantages.write(t, advantage)
        advantages = advantages.stack()
        # 计算旧动作索引及概率
        old_probs = self.actor_model(batch_states)
        batch_indices = tf.tile(tf.reshape(tf.range(batch_size), [-1, 1, 1]), [1, num_agents, 1])
        agent_indices = tf.tile(tf.reshape(tf.range(num_agents), [1, -1, 1]), [batch_size, 1, 1])
        indices = tf.stack([batch_indices, agent_indices, batch_actions], axis=-1)
        old_action_prob = tf.gather_nd(old_probs, indices)
        # 排除advantages、旧动作概率和目标估计值的梯度
        advantages = tf.stop_gradient(advantages)
        old_action_prob = tf.stop_gradient(old_action_prob)
        target = tf.stop_gradient(target)
        for i in range(self.epoch):
            # Update Critic model
            critic_history = self.critic_model.fit(batch_states, target, verbose=0)
            # Train Actor model
            with tf.GradientTape() as tape:
                # Forward pass
                new_probs = self.actor_model(batch_states)
                new_action_prob = tf.gather_nd(new_probs, indices)

                # 使用 PPO 的裁剪损失计算行动者（Actor）损失
                ratios = tf.exp(tf.math.log(new_action_prob + 1e-10) - tf.math.log(old_action_prob + 1e-10))
                surr1 = ratios * advantages
                surr2 = tf.clip_by_value(ratios, 1 - config.Training.clip_epsilon,
                                         1 + config.Training.clip_epsilon) * advantages
                entropy = -tf.reduce_sum(new_action_prob * tf.math.log(new_action_prob + 1e-10), axis=-1,
                                         keepdims=True)  # 计算每个时间步骤的熵
                actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2) - config.Training.entropy_beta * entropy)

            tape.watch(self.actor_model.trainable_variables)
            # Compute gradients and perform a policy update
            actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
            # 一整个批次应用梯度
            # self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))
            self.scalar_record(self.epoch * episode + i, actor_loss, critic_history.history['loss'], )
            logger.info(f"data t: {episode}:epoch: {i}, Actor Loss: {actor_loss}"
                        f"    Critic Loss: {critic_history.history['loss']}")
            return actor_grads

    def get_action(self, state):
        logger.debug("get actions with batch train state ... ")
        action_probs = self.actor_model(state)  # Shape is [batch_size, 10, 9]
        sampled_actions = tf.random.categorical(tf.math.log(action_probs[0]), num_samples=1)
        chosen_actions = tf.reshape(sampled_actions, shape=(10, 1)).numpy()

        return chosen_actions

    def get_max_action(self, state):
        logger.debug("get actions with batch train state ... ")
        action_probs = self.actor_model(state)  # Shape is [batch_size, 10, 9]
        sampled_actions = tf.argmax(action_probs[0], axis=-1).numpy()
        chosen_actions = tf.reshape(sampled_actions, shape=(10, 1)).numpy()

        return chosen_actions


if __name__ == '__main__':
    # 初始化模型和优化器
    actor_model = SoldierActionModel()  # 替换为你实际的模型初始化代码
    critic_model = SoldierCriticModel()  # 替换为你实际的模型初始化代码
    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # 初始化PPOAgent
    agent = PPOAgent(actor_model, critic_model)

    # 以下部分请替换为你实际的数据收集逻辑
    states, actions, rewards, next_states, dones = [], [], [], [], []

    # 将数据转换为NumPy数组或Tensor
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    dones = np.array(dones)

    # 训练代理
    agent.train(states, actions, rewards, next_states, dones)
