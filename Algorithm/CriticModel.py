import tensorflow as tf
from keras.layers import TimeDistributed

from Algorithm.Layers import CriticLiner, TimeDistributed_CNN, CNN


class SoldierCriticModel(tf.keras.Model):
    def __init__(self):
        super(SoldierCriticModel, self).__init__()
        # 全局视野21*21*3特征提取（敌、我、山地)
        self.global_cnn = CNN(name='global_cnn', input_channels=3)
        # 士兵视野21*21*4（敌、友、山地、自己） 特征提取
        self.sight_cnn = TimeDistributed_CNN(name='sight_cnn', input_channels=4)
        self.liner = CriticLiner()

    # @tf.function
    def call(self, _inputs):
        _map_tensor, _sight_tensor, _sight_state_tensor = _inputs
        # 获取全局特征
        global_tensor = _map_tensor[:, 0, :]
        global_features = self.global_cnn(global_tensor)
        global_features = tf.expand_dims(global_features, axis=1)
        global_features = tf.repeat(global_features, repeats=10, axis=1)

        # 获取局部特征
        # _sight_tensor = tf.concat([_sight_tensor, _sight_state_tensor], axis=-1)
        sight_features = self.sight_cnn(_sight_tensor)

        features = tf.concat([global_features, sight_features], axis=-1)
        return self.liner(features)


if __name__ == '__main__':
    # Create an instance of the Critic model
    critic_model = SoldierCriticModel()
    # Create some dummy data, similar to what you did for the Actor model
    map_tensor = tf.random.normal([8, 10, 21, 21, 3])  # Global view
    sight_tensor = tf.random.normal([8, 10, 21, 21, 4])  # Sight for each of the 10 soldiers
    sight_state_tensor = tf.random.normal([8, 10, 21, 21, 3])  # State for each of the 10 soldiers
    inputs = (map_tensor, sight_tensor, sight_state_tensor)
    # Run the Critic model
    value_output = critic_model(inputs)
    critic_model.summary()
    # Check the output
    print("Value Output Shape:", value_output.shape)  # Should be (10, 1)
    # print("Value Output:", value_output)
