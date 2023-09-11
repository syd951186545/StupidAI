import tensorflow as tf
from keras.layers import LSTM, TimeDistributed

from Algorithm.Layers import ActorGlobalLiner, ActorLoacalLiner, OutputLiner, CNN, TimeDistributed_CNN


class SoldierActionModel(tf.keras.Model):
    def __init__(self):
        super(SoldierActionModel, self).__init__()
        self.actionNums = 9
        self.global_cnn = CNN(name='global_cnn', input_channels=3)
        self.sight_cnn = TimeDistributed_CNN(name='sight_cnn', input_channels=4)  # TimeDistributed
        self.global_liner = ActorGlobalLiner()
        self.local_liner = ActorLoacalLiner()  # TimeDistributed
        self.lstm = LSTM(128, return_state=True)
        # 初始化一个空的隐藏状态列表
        self.hidden_state = None
        self.out_liner = OutputLiner()
        self.__memory_cut = 0

    def reset_hidden_states(self):
        self.hidden_state = None

    def call(self, _inputs):
        self.__memory_cut += 1
        _map_tensor, _sight_tensor, _sight_state_tensor = _inputs
        # 全局信息处理
        global_tensor = _map_tensor[:, 0, :]
        global_features = self.global_cnn(global_tensor)
        global_features = self.global_liner(global_features)
        global_features = tf.expand_dims(global_features, axis=1)
        # 局部信息处理
        # _sight_tensor = tf.concat([_sight_tensor, _sight_state_tensor], axis=-1)
        sight_features = self.sight_cnn(_sight_tensor)
        sight_features = self.local_liner(sight_features)
        # 每10次训练或者输入10张画面后重置一次隐藏状态
        if self.__memory_cut % 10 == 0:
            self.reset_hidden_states()
        # 全局信息记忆
        if self.hidden_state is not None:
            hidden_state_h, hidden_state_c = self.hidden_state
            lstm_outputs = self.lstm(global_features, initial_state=[hidden_state_h, hidden_state_c])
        else:
            lstm_outputs = self.lstm(global_features)
        # lstm_outputs 是一个包含三个元素的列表：[输出, 新的 h 状态, 新的 c 状态] 并更新隐藏状态
        lstm_out, new_state_h, new_state_c = lstm_outputs
        self.hidden_state = [new_state_h, new_state_c]

        # 全局信息和局部信息融合
        lstm_out = tf.expand_dims(lstm_out, axis=1)
        global_features = tf.concat([lstm_out, global_features], axis=-1)
        global_features = tf.repeat(global_features, repeats=10, axis=1)
        features = tf.concat([global_features, sight_features], axis=-1)

        return self.out_liner(features)


if __name__ == '__main__':
    tf.config.run_functions_eagerly(False)
    # TensorBoard: 记录日志
    map_tensor = tf.random.normal([8, 10, 21, 21, 3])  # Global view
    sight_tensor = tf.random.normal([8, 10, 21, 21, 4])  # Sight for each of the 10 soldiers
    sight_state_tensor = tf.random.normal([8, 10, 21, 21, 3])  # State for each of the 10 soldiers

    # Create an instance of the model
    model = SoldierActionModel()
    for i in range(10):
        actions = model(_inputs=(map_tensor, sight_tensor, sight_state_tensor))
        print("Actions:", actions.shape)  # Should be (10, 10, 9)
    model.summary()

    # Run the model
    # print("Actions:", actions)
