import numpy as np
import tensorflow as tf
import ujson as json

import config
from Environment.check_data import check_map
from Environment.logger import logger

HEIGHT, WIDTH = 21, 21  # 地图大小


def _generate_map(json_dict=None):
    height = json_dict['mapInfo']['height']
    width = json_dict['mapInfo']['width']
    map_ndarray = np.zeros((width, height, 3))

    map_zones = json_dict['mapInfo']['zones']
    for item in map_zones:
        if item['roleType'] == 'mountain':
            map_ndarray[item['pos']['x'], item['pos']['y'],  0] = 1
    return map_ndarray


def _generate_teams(json_dict=None):
    our = json_dict['players']['teamOur']['roles']
    # 采用列表来保证输入数据与输出动作与士兵id对应,我方士兵状态的收集顺序始终是2special 3artillery 5infantry
    # 对局过程中可能士兵数不足10人 ，所有无士兵的位置为空
    team_our_list = [[]]*10
    special_index, artillery_index, infantry_index = 0, 2, 5
    for i, item in enumerate(our):
        pos = (item['pos']['x'], item['pos']['y'])
        if item['roleType'] == 'special':
            attack_power = 11 / 10
            xxx = (item['health'], attack_power, item['life'])
            team_our_list[special_index] = [item['id'], pos, xxx]
            special_index += 1
        elif item['roleType'] == 'artillery':
            attack_power = 12 / 10
            xxx = (item['health'], attack_power, item['life'])
            team_our_list[artillery_index] = [item['id'], pos, xxx]
            artillery_index += 1
        else:
            attack_power = 10 / 10
            xxx = (item['health'], attack_power, item['life'])
            team_our_list[infantry_index] = [item['id'], pos, xxx]
            infantry_index += 1

    enemy = json_dict['players']['teamEnemy']['posList']
    team_enemy = {}
    for i, item in enumerate(enemy):
        team_enemy[str(i)] = (item['x'], item['y'])
    return team_our_list, team_enemy


def _update_map_with_teams(map_ndarray, team_our_list, team_enemy, team_our_state_ndarray):
    for soldier in team_our_list:
        if soldier:  # 对局过程中可能士兵数不足10人
            _id, pos, state = soldier
            x, y = pos
            map_ndarray[x, y, 1] = 1
            team_our_state_ndarray[x, y, :] = state

    for k, v in team_enemy.items():
        x, y = v
        map_ndarray[x, y, 2] = 1
    return map_ndarray, team_our_state_ndarray


def _generate_sight_maps(team_our_list, map_ndarray, team_our_state_ndarray, sight_size=21):
    sight_list = []
    sight_state_list = []

    for soldier in team_our_list:
        if soldier:  # 对局过程中可能士兵数不足10人
            _id, pos, state = soldier
            x, y = pos
            sight = np.zeros((sight_size, sight_size, 4))
            sight_state = np.zeros((sight_size, sight_size, 3))
            for dx in range(-10, 11):
                for dy in range(-10, 11):
                    new_x = x + dx
                    new_y = y + dy
                    if 0 <= new_x < HEIGHT and 0 <= new_y < WIDTH:
                        sight[dx + 10, dy + 10, :3] = map_ndarray[new_x, new_y, :]
                        sight_state[dx + 10, dy + 10, :] = team_our_state_ndarray[new_y, new_x, :]
                    else:
                        sight[dx + 10, dy + 10, :] = -1
                        sight_state[dx + 10, dy + 10, :] = -1

            sight[10, 10, 3] = 1
        else:
            sight = np.zeros((sight_size, sight_size, 4))
            sight_state = np.zeros((sight_size, sight_size, 3))
        sight_list.append(sight)
        sight_state_list.append(sight_state)
    return sight_list, sight_state_list


def _trans_state_2_agent_tensor(_map_ndarray, _sight_ndarray, _sight_state_ndarray):
    """
    该方法主要讲环境返回的state进行预处理成Tensor指定的格式，包括扩展第一个维度
    state : _map_ndarray [21*21*3],_sight_ndarray[10*21*21*4],_sight_state_ndarray[10*21*21*3]
    :param _map_ndarray:
    :param _sight_ndarray:
    :param _sight_state_ndarray:
    :return: tf tensor  [1*1*21*21*3],[1*10*21*21*4],[1*10*21*21*3]
    """
    map_tensor_tf = tf.constant(_map_ndarray, dtype=tf.float32)
    map_tensor_tf = tf.expand_dims(map_tensor_tf, 0)
    map_tensor_tf = tf.expand_dims(map_tensor_tf, 0)

    sight_tensor_tf = tf.constant(_sight_ndarray, dtype=tf.float32)
    sight_tensor_tf = tf.expand_dims(sight_tensor_tf, 0)

    sight_state_tensor_tf = tf.constant(_sight_state_ndarray, dtype=tf.float32)
    sight_state_tensor_tf = tf.expand_dims(sight_state_tensor_tf, 0)

    return map_tensor_tf, sight_tensor_tf, sight_state_tensor_tf


def parse_env_json(json_dict, dict_is_file=False):
    """
    解析环境中返回的json流，转换为模型需要的三层视野和状态数据 tf tensor  [1*21*21*3],[1*10*21*21*4],[1*10*21*21*3]
    shape = 模型input
    :param json_str:
    :param dict_is_file:
    :return:
    """
    logger.debug("start to parse_env_json ...")
    if dict_is_file:
        try:
            with open(json_dict, 'r') as f:
                json_str = f.read()
                json_dict = json.loads(json_str)
        except FileExistsError:
            logger.info("parse json failed when input json is file")
            return None
    map_ndarray = _generate_map(json_dict)
    team_our_list, team_enemy = _generate_teams(json_dict)
    team_our_state_ndarray = np.zeros((HEIGHT, WIDTH, 3))

    map_ndarray, team_our_state_ndarray = _update_map_with_teams(map_ndarray, team_our_list, team_enemy,
                                                                 team_our_state_ndarray)
    sight_ndarray, sight_state_ndarray = _generate_sight_maps(team_our_list, map_ndarray, team_our_state_ndarray)
    if config.Training.check_trans_state:
        check_map(map_ndarray)
        for one_sight_ndarray in sight_ndarray:
            check_map(one_sight_ndarray)
    return _trans_state_2_agent_tensor(map_ndarray, sight_ndarray, sight_state_ndarray), team_our_list


if __name__ == '__main__':
    parse_env_json("request.json", dict_is_file=True)