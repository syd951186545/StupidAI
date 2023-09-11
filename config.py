import logging
import os
from datetime import datetime

import tensorflow as tf
from tensorflow.python.eager.def_function import run_functions_eagerly
from tensorflow.python.ops.summary_ops_v2 import create_file_writer
from tensorflow import distribute
from tensorflow.keras.optimizers import Adam

PROJECT_ROOT_DIR: str = os.path.dirname(os.path.abspath(__file__))
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")


class Global:
    LOGGER: str = "logging"  # logging or print, 开发、训练应logging更快更便捷，但比赛环境只能print打印日志
    LOGGER_LEVEL = logging.INFO  # 如果使用logging,设定日志打印等级
    PROJECT_ROOT_DIR: str = PROJECT_ROOT_DIR
    MODELS_PATH: str = os.path.join(PROJECT_ROOT_DIR, "Models")
    PPO_LOG_DIR = os.path.join(PROJECT_ROOT_DIR, "Logs", "PPOAgent")
    USE_DEVICE = "CPU"  # CPU，GPU，ALL_AVAILABLE_GPUS


class Training:
    LOGGER_LEVEL: str = "debug"  # logging or print, 开发、训练应logging更快更便捷，但比赛环境只能print打印日志
    LOG_DIR: str = os.path.join(PROJECT_ROOT_DIR, "Logs", "Train")
    TRAIN_MODELS_DIR: str = os.path.join(PROJECT_ROOT_DIR, "Models", "Train")
    TRAIN_MODELS_PATH: str = os.path.join(Global.MODELS_PATH, "Train")

    training_tf_writer = create_file_writer(os.path.join(Global.PPO_LOG_DIR, current_time))  # 训练时用的全局记录器

    # memory-distribute-train
    DEVICE = "GPU"  # "CPU"则表示单卡cpu计算，"GPU" 模型则采用训练策略：MirroredStrategy为单机器多gpu或者单gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # 设置要使用的GPU设备
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    TF_FUNCTION = run_functions_eagerly(True)  # 全局启用tf.function计算图构建加速,调试时关闭
    strategy = None if DEVICE == "CPU" else distribute.MirroredStrategy()
    strategy_gpu_nums = None if DEVICE == "CPU" else strategy.num_replicas_in_sync
    BATCH_SIZE_PER_REPLICA = 100
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy_gpu_nums
    EPOCH: int = 10000  # 训练次数
    NUM_ENVS: int = 1  # 采样时并行运行的游戏环境数
    NUM_THREADS = NUM_ENVS  # 多线程采样,暂时先跟环境数保持一致，一个环境一个线程收集
    sample_type = "categorical"  # 采样类型，categorical或者greedy (categorical依照动作概率分布采样，greedy则依照概率最大采样)
    sample_agent = False  # 采样agent为True表示独立采样10次后更新一次agent，False表示每次采样都更新一次agent
    game_visible = True  # 检查状态转换成tensor是否正确
    game_visible_pause = 0.01  # int, 暂停多少秒
    check_trans_state = False  # 检查状态转换成tensor是否正确

    MAX_ROUNDS = 10000  # 每场对局最大的运行回合，指步数或者游戏帧数
    epoch = 10  # 每一场数据用来训练的次数
    trajectory_nums = 500  # 经验池中的trajectory数量
    entropy_beta = 0.1  # 动作概率分布熵，增大时 增加熵在loss中的占比，该值倾向探索
    clip_epsilon = 0.2  # PPO近端裁剪系数
    actor_optimizer = Adam(learning_rate=1e-4)  # 行动者的优化器
    critic_optimizer = Adam(learning_rate=1e-4)  # 评价者的优化器

    # Hyperparameters
    gamma = 0.98  # 计算优势值的值衰减系数
    lambda_gae = 0.95  # 计算优势值的值未来值占比系数


class Predict:
    LOG_DIR = os.path.join(PROJECT_ROOT_DIR, "Logs", "Predict")
    PREDICT_MODELS_PATH: str = os.path.join(Global.MODELS_PATH, "Predict", "actor_model")
