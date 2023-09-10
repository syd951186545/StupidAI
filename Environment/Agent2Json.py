import ujson as json
import numpy as np

from Environment.logger import logger


def parse_agent_actions(_actions, _team_our_list, write_file=False):
    """
    传入参数为
    :param _actions: agent返回的_actions [10],十个士兵的1个动作
    :param _team_our_list: 解析环境返回时保存的team_our_list，包含 id, initial_coordinates, state
    :param write_file: 训练时是否写文件来验证
    :return:
    """
    logger.debug("start to parse agent actions")
    soldier_output_list = []
    # 坐标的变化字典，键是0-8的整数，值是(x,y)的坐标变化
    direction_dict = {
        0: (-1, -1),
        1: (-1, 0),
        2: (-1, 1),
        3: (0, -1),
        4: (0, 0),
        5: (0, 1),
        6: (1, -1),
        7: (1, 0),
        8: (1, 1),
    }
    for i, soldier in enumerate(_team_our_list):
        if soldier:
            s_id, initial_coordinates, _ = soldier
            x, y = initial_coordinates
            new_coordinates_list = []
            dx, dy = direction_dict[_actions[i, 0]]
            x, y = x + dx, y + dy
            new_coordinates_list.append({'x': x, 'y': y})
            # 将每个单位的信息添加到输出列表中
            soldier_output_list.append({'roleId': s_id, 'posList': new_coordinates_list})
        output_dict = {'soldiers': soldier_output_list}
        # 将字典转换为JSON文件
        if write_file:
            with open('./Environment/action_response.json', 'w') as f:
                json.dump(output_dict, f)
        logger.debug("return agent actions dict")
    return output_dict


if __name__ == '__main__':
    actions = np.random.randint(0, 9, (10, 10))
    soldier_pos_dict = {
        'id0': (0, 0),
        'id1': (1, 1),
        'id2': (2, 2),
        'id3': (3, 3),
        'id4': (4, 4),
        'id5': (5, 5),
        'id6': (6, 6),
        'id7': (7, 7),
        'id8': (8, 8),
        'id9': (9, 9),
    }
    parse_agent_actions(actions, soldier_pos_dict)