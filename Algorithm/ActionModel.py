import tensorflow as tf
from keras.layers import LSTM, TimeDistributed

from Algorithm.Layers import ActorGlobalLiner, ActorLoacalLiner, OutputLiner, CNN, TimeDistributed_CNN


class SoldierActionModel(tf.keras.Model):
    def __init__(self):
        super(SoldierActionModel, self).__init__()
        self.actionNums = 9
        self.global_cnn = TimeDistributed_CNN(name='global_cnn', input_channels=3)
        self.sight_cnn = TimeDistributed_CNN(name='sight_cnn', input_channels=4)  # TimeDistributed
        self.global_liner = ActorGlobalLiner()
        self.local_liner = ActorLoacalLiner()  # TimeDistributed
        self.lstm = LSTM(128, return_state=False, return_sequences=True)
        # 初始化一个空的隐藏状态列表
        self.out_liner = OutputLiner()
    @tf.function
    def call(self, _inputs):
        _map_tensor, _sight_tensor, _sight_state_tensor = _inputs
        # 全局信息处理
        # global_tensor = _map_tensor[0, :, :]
        global_features = self.global_cnn(_map_tensor)
        global_features = self.global_liner(global_features)
        # 局部信息处理
        # _sight_tensor = tf.concat([_sight_tensor, _sight_state_tensor], axis=-1)
        sight_features = self.sight_cnn(_sight_tensor)
        sight_features = self.local_liner(sight_features)
        # 全局信息记忆
        lstm_outputs = self.lstm(global_features)

        # 全局信息和局部信息融合，以及记忆信息融合
        features = tf.concat([global_features, sight_features, lstm_outputs], axis=-1)

        return self.out_liner(features)


if __name__ == '__main__':
    tf.config.run_functions_eagerly(False)
    # TensorBoard: 记录日志
    map_tensor = tf.random.normal([10, 100, 21, 21, 3])  # Global view
    sight_tensor = tf.random.normal([10, 100, 21, 21, 4])  # Sight for each of the 10 soldiers
    sight_state_tensor = tf.random.normal([10, 100, 21, 21, 3])  # State for each of the 10 soldiers

    # Create an instance of the model
    model = SoldierActionModel()
    for i in range(10):
        actions = model(_inputs=(map_tensor, sight_tensor, sight_state_tensor))
        print("Actions:", actions.shape)  # Should be (10, 10, 9)
    model.summary()

    # Run the model
    # print("Actions:", actions)
