import tensorflow as tf
from keras.layers import TimeDistributed

from Algorithm.Layers import CNN_strides2, ActorLiner, CNN_strides1, AttentionModule


class SoldierActionModel(tf.keras.Model):
    def __init__(self):
        super(SoldierActionModel, self).__init__()
        self.actionNums = 9
        # 全局视野21*21*3特征提取（敌、我、山地)
        self.global_cnn = CNN_strides1(name='global_cnn', input_channels=3)
        # 士兵视野21*21*4（敌、友、山地、自己） 特征提取
        self.sight_cnn = CNN_strides1(name='sight_cnn', input_channels=4)
        self.self_attention = TimeDistributed(AttentionModule(128))
        self.liner = ActorLiner()

    @tf.function
    def call(self, _inputs):
        _map_tensor, _sight_tensor, _sight_state_tensor = _inputs
        # 获取全局和局部特征
        global_features = self.global_cnn(_map_tensor)
        sight_features = self.sight_cnn(_sight_tensor)
        global_features = self.self_attention(global_features)
        sight_features = self.self_attention(sight_features)
        features = tf.concat([global_features, sight_features], axis=-1)
        return self.liner(features)


if __name__ == '__main__':
    tf.config.run_functions_eagerly(False)
    # TensorBoard: 记录日志
    map_tensor = tf.random.normal([8, 10, 21, 21, 3])  # Global view
    sight_tensor = tf.random.normal([8, 10, 21, 21, 4])  # Sight for each of the 10 soldiers
    sight_state_tensor = tf.random.normal([8, 10, 21, 21, 3])  # State for each of the 10 soldiers

    # Create an instance of the model
    model = SoldierActionModel()
    actions = model(_inputs=(map_tensor, sight_tensor, sight_state_tensor))

    model.summary()

    # Run the model
    print("Actions:", actions.shape)  # Should be (10, 10, 9)
    # print("Actions:", actions)
