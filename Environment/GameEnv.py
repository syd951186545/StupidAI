import copy
import random
import matplotlib
from sklearn.cluster import KMeans
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

import config


class Soldier:
    type_attributes = {
        "special": {"attack": 11, "max_respawn": 2},
        "artillery": {"attack": 12, "max_respawn": 2},
        "infantry": {"attack": 10, "max_respawn": 2},
    }

    def __init__(self, soldier_id, soldier_type, x, y, health=100, life=0):
        self.id = soldier_id
        self.type = soldier_type
        self.x = x
        self.y = y
        self.hp = health
        self.attack = self.type_attributes[soldier_type]["attack"]
        self.max_respawn = self.type_attributes[soldier_type]["max_respawn"]
        self.life = life  # 当前角色剩余命数，相当于复活次数
        self.exception = False
        self.be_attacked = {}


def find_enemy_center(teamEnemy):
    """
    :param teamEnemy: Soldier Map {id:Soldier(id)}
    :return: center point list
    """
    coordinates = np.array([[soldier.x, soldier.y] for soldier in teamEnemy.values()])
    if len(teamEnemy) > 3:
        n_clusters = 3
    else:
        n_clusters = len(teamEnemy)
    # 使用K-means算法进行聚类，找出三个中心
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(coordinates)
    centers = kmeans.cluster_centers_
    # centers = [(int(x), int(y)) for x, y in centers]
    return centers


def distances_enemies(old_pos, new_pos, enemy_points):
    def manhattan_distance(pos1, pos2):
        return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))

    original_distances = [manhattan_distance(old_pos, enemy_pos) for enemy_pos in enemy_points]
    new_distances = [manhattan_distance(new_pos, enemy_pos) for enemy_pos in enemy_points]

    return any(new_d < original_d for new_d, original_d in zip(new_distances, original_distances))


def distances_centers(old_pos, new_pos, centers):
    def distance(pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    # 假设 cluster_centers 是一个包含三个聚类中心坐标的 NumPy 数组
    cluster_centers = np.array(centers)
    # 当前点的坐标
    # 计算到每个中心的距离
    distances = np.array([distance(old_pos, (cx, cy)) for cx, cy in cluster_centers])
    # 找出最近的中心
    closest_center = cluster_centers[np.argmin(distances)]
    closest_distance = np.min(distances)

    # 计算新的点到最近中心的距离
    new_distance = distance(new_pos, closest_center)

    # 判断点是否朝向最近的中心靠近
    if new_distance < closest_distance:
        return True
    else:
        return False


matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.ion()
plt.show(block=False)


def step_visual_decorator(step):
    def visual_step(self, actions_json_dict):
        if not self.visible:
            return step(self, actions_json_dict)
        """
        如果可视化，绘制地图
        """
        for element in self.elements_to_clear:
            element.remove()
        self.elements_to_clear.clear()
        if not self.title_round:
            self.title_round = self.fig.text(0.02, 0.98, f'cur round is : {self.cur_round}', verticalalignment='top',
                                             horizontalalignment='left')
        else:
            self.title_round.remove()
            self.title_round = self.fig.text(0.02, 0.98, f'cur round is : {self.cur_round}', verticalalignment='top',
                                             horizontalalignment='left')
        # 1、绘制敌我士兵
        old_enemy_soldiers = {role_id: copy.deepcopy(soldier) for role_id, soldier in self.teamEnemy.items()}
        old_our_soldiers = {role_id: copy.deepcopy(soldier) for role_id, soldier in self.teamOur.items()}
        for role_id, soldier in old_enemy_soldiers.items():
            hp = soldier.hp
            side_length = max(0.2, hp / 100)
            enemy = plt.Rectangle((soldier.x, soldier.y), 1, side_length, color='darkred')
            self.ax.add_patch(enemy)
            self.elements_to_clear.append(enemy)

            hp_label = plt.text(soldier.x, soldier.y, hp, fontsize=9, color='gold')
            self.elements_to_clear.append(hp_label)
        for role_id, soldier in old_our_soldiers.items():
            hp = soldier.hp
            side_length = max(0.2, hp / 100)
            our = plt.Rectangle((soldier.x, soldier.y), 1, side_length, color='#88D8B0')
            self.ax.add_patch(our)
            self.elements_to_clear.append(our)

            hp_label = plt.text(soldier.x, soldier.y, hp, fontsize=9, color='gold')
            self.elements_to_clear.append(hp_label)

        # 2、地图更新移动到下一个状态
        result = step(self, actions_json_dict)

        # 3、绘制我方移动箭头和移动获取的reward
        for role_id, soldier in self.teamOur.items():
            reward = self.soldier_reward[role_id]
            old_x, old_y = old_our_soldiers[role_id].x, old_our_soldiers[role_id].y
            tar_x, tar_y = soldier.x, soldier.y
            arrow = FancyArrowPatch((old_x + 0.5, old_y + 0.5), (tar_x + 0.5, tar_y + 0.5),
                                    arrowstyle='-|>', mutation_scale=15, color='#2A0481')
            self.ax.add_patch(arrow)
            self.elements_to_clear.append(arrow)

            reward_label = self.ax.text(tar_x, tar_y, str(reward), fontsize=9, color='#2A0481')
            self.elements_to_clear.append(reward_label)
        plt.draw()
        plt.pause(config.Training.game_visible_pause)
        for element in self.elements_to_clear:
            element.remove()
        self.elements_to_clear.clear()

        # 清除元素后 重绘制敌我士兵
        new_enemy_soldiers = {role_id: soldier for role_id, soldier in self.teamEnemy.items()}
        new_our_soldiers = {role_id: soldier for role_id, soldier in self.teamOur.items()}
        for role_id, soldier in new_enemy_soldiers.items():
            hp = soldier.hp
            side_length = max(0.2, hp / 100)
            enemy = plt.Rectangle((soldier.x, soldier.y), 1, side_length, color='darkred')
            self.ax.add_patch(enemy)
            self.elements_to_clear.append(enemy)

            hp_label = plt.text(soldier.x, soldier.y, hp, fontsize=9, color='gold')
            self.elements_to_clear.append(hp_label)
        for role_id, soldier in new_our_soldiers.items():
            hp = soldier.hp
            side_length = max(0.2, hp / 100)
            our = plt.Rectangle((soldier.x, soldier.y), 1, side_length, color='#88D8B0')
            self.ax.add_patch(our)
            self.elements_to_clear.append(our)

            hp_label = plt.text(soldier.x, soldier.y, hp, fontsize=9, color='gold')
            self.elements_to_clear.append(hp_label)
        plt.draw()
        plt.pause(config.Training.game_visible_pause)
        return result

    return visual_step


def visual_enemy_decorator(enemy_step):
    def visual_render(self):
        if not self.visible:
            return enemy_step(self)
        else:
            for element in self.elements_to_clear:
                element.remove()
            self.elements_to_clear.clear()
            # 1、绘制我方士兵和敌方士兵
            our_soldiers = {role_id: soldier for role_id, soldier in self.teamOur.items()}
            for role_id, soldier in our_soldiers.items():
                if role_id not in self.teamOur:
                    continue
                if soldier.hp <= 0:
                    continue
                hp = soldier.hp
                side_length = max(0.2, hp / 100)
                our = plt.Rectangle((soldier.x, soldier.y), 1, side_length, color='#88D8B0')
                self.ax.add_patch(our)
                self.elements_to_clear.append(our)

                hp_label = plt.text(soldier.x, soldier.y, hp, fontsize=9, color='gold')
                self.elements_to_clear.append(hp_label)
            old_enemy_soldiers = {role_id: soldier for role_id, soldier in self.teamEnemy.items()}
            for role_id, soldier in old_enemy_soldiers.items():
                if role_id not in self.teamEnemy:
                    continue
                if soldier.hp <= 0:
                    continue
                hp = soldier.hp
                side_length = max(0.2, hp / 100)
                enemy = plt.Rectangle((soldier.x, soldier.y), 1, side_length, color='darkred')
                self.ax.add_patch(enemy)
                self.elements_to_clear.append(enemy)

                hp_label = plt.text(soldier.x, soldier.y, hp, fontsize=9, color='gold')
                self.elements_to_clear.append(hp_label)
            plt.draw()
            plt.pause(config.Training.game_visible_pause)
            # 2、模拟移动和攻击
            enemy_step(self)
            # 3、绘制敌方移动箭头
            new_enemy_soldiers = {role_id: soldier for role_id, soldier in self.teamEnemy.items()}
            for role_id, soldier in old_enemy_soldiers.items():
                if role_id in new_enemy_soldiers:
                    hp = self.teamEnemy[role_id].hp
                    old_x, old_y = old_enemy_soldiers[role_id].x, old_enemy_soldiers[role_id].y
                    tar_x, tar_y = new_enemy_soldiers[role_id].x, new_enemy_soldiers[role_id].y
                    arrow = FancyArrowPatch((old_x + 0.5, old_y + 0.5), (tar_x + 0.5, tar_y + 0.5),
                                            arrowstyle='-|>', mutation_scale=15, color='#2A0481')
                    self.elements_to_clear.append(arrow)
                    self.ax.add_patch(arrow)

                    hp_label = self.ax.text(old_x, old_y, str(hp), fontsize=9, color='gold')
                    self.elements_to_clear.append(hp_label)
            plt.draw()
            plt.pause(config.Training.game_visible_pause)
            # 4、清除元素后重新绘制敌方士兵位置和绘制我方士兵
            for element in self.elements_to_clear:
                element.remove()
            self.elements_to_clear.clear()
            our_soldiers = {role_id: soldier for role_id, soldier in self.teamOur.items()}
            for role_id, soldier in our_soldiers.items():
                if role_id not in self.teamOur:
                    continue
                if soldier.hp <= 0:
                    continue
                hp = soldier.hp
                side_length = max(0.2, hp / 100)
                our = plt.Rectangle((soldier.x, soldier.y), 1, side_length, color='#88D8B0')
                self.ax.add_patch(our)
                self.elements_to_clear.append(our)

                hp_label = plt.text(soldier.x, soldier.y, hp, fontsize=9, color='gold')
                self.elements_to_clear.append(hp_label)
            for role_id, soldier in new_enemy_soldiers.items():
                if role_id not in self.teamEnemy:
                    continue
                if soldier.hp <= 0:
                    continue
                hp = soldier.hp
                side_length = max(0.2, hp / 100)
                enemy = plt.Rectangle((soldier.x, soldier.y), 1, side_length, color='darkred')
                self.ax.add_patch(enemy)
                self.elements_to_clear.append(enemy)

                hp_label = plt.text(soldier.x, soldier.y, hp, fontsize=9, color='gold')
                self.elements_to_clear.append(hp_label)

    return visual_render


class SoldierGameEnv:
    MAX_ROUNDS = config.Training.MAX_ROUNDS
    REWARD_WIN = 1
    REWARD_LOSE = -20
    REWARD_DEFAULT = 0
    REWARD_ATTACK_ENEMY = 1
    REWARD_WALK_TO_ENEMY = 0
    REWARD_WALK = 0
    REWARD_NOT_WALK = 0
    REWARD_DANGER = 0
    REWARD_CONFLICT = 0
    REWARD_EXCEPTION = -1
    height, width = 22, 22
    MOUNTAIN_NUMS = 50

    def __init__(self):
        self.visible = config.Training.game_visible
        self.__initialize_state()

    def __initialize_state(self):
        self.map = None
        self.teamOur = {}
        self.teamEnemy = {}
        self.our_attack_dict = {}
        self.enemy_attack_dict = {}
        self.cur_round = 0
        self.totalScore = 0
        self.soldier_reward = defaultdict(int)
        self.soldier_done = defaultdict(bool)
        self.teamId = "10011"
        self.teamName = "GGBang"
        self.elements_to_clear = []

    def __init_visual_map(self):
        if not self.visible:
            return
        plt.close()
        self.fig, self.ax = plt.subplots()
        self.title_round = self.fig.text(0.02, 0.98, f'cur round is : {self.cur_round}', verticalalignment='top',
                                         horizontalalignment='left')
        self.ax.set_xticks(np.arange(0, self.width + 1, 5))
        self.ax.set_yticks(np.arange(0, self.height + 1, 5))
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid()
        # 绘制地图网格
        for x in range(self.width + 1):
            self.ax.axvline(x=x, linestyle='--', color='lightgrey')
        for y in range(self.height + 1):
            self.ax.axhline(y=y, linestyle='--', color='lightgrey')
        # 绘制地图元素
        for x in range(self.width):
            for y in range(self.height):
                if self.map[x, y] == -1:  # Mountain
                    self.ax.add_patch(plt.Rectangle((x, y), 1, 1, color='black'))
                elif self.map[x, y] == 1:  # Our soldier
                    our = plt.Rectangle((x, y), 1, 1, color='#88D8B0')
                    self.ax.add_patch(our)
                    self.elements_to_clear.append(our)
                elif self.map[x, y] == 2:  # Enemy soldier
                    enemy = plt.Rectangle((x, y), 1, 1, color='darkred')
                    self.ax.add_patch(enemy)
                    self.elements_to_clear.append(enemy)

    def __init_step(self):
        self.cur_round += 1
        # 清除reward
        self.soldier_reward.clear()
        for _id, soldier in self.teamOur.items():
            loss_hp = (soldier.max_respawn - soldier.life) * 100 + (100 - soldier.hp)
            loss_hp /= 100
            self.soldier_reward[_id] = self.REWARD_DEFAULT

    def __init_map(self):
        self.map = np.zeros((self.width, self.height), dtype=int)
        all_coordinates = [(x, y) for x in range(self.width) for y in range(self.height)]
        # 划分两个区域，一个用于我方士兵（靠近x轴），另一个用于敌方士兵（靠近width）
        our_side_coordinates = [(x, y) for x in range(self.width // 7) for y in range(self.height)]
        enemy_side_coordinates = [(x, y) for x in range(self.width // 7 * 5, self.width) for y in
                                  range(self.height // 7 * 6, self.height)]

        # 随机打乱两个区域的坐标点列表
        np.random.shuffle(our_side_coordinates)
        np.random.shuffle(enemy_side_coordinates)
        np.random.shuffle(all_coordinates)

        # 是否双方各自区域内随机选取士兵
        two_side = False
        if two_side:
            # 在各自区域内选取10个位置
            coordinates_our = our_side_coordinates[:1]
            coordinates_enemy = enemy_side_coordinates[:10]
        else:
            coordinates_our = all_coordinates[:1]
            coordinates_enemy = all_coordinates[10:20]

        # 生成所有可能的坐标，并从中移除已经被选为士兵的坐标
        available_coordinates = [coord for coord in all_coordinates if
                                 coord not in coordinates_our and coord not in coordinates_enemy]

        # 从剩下的坐标中随机选择MOUNTAIN_NUMS个作为障碍物
        np.random.shuffle(available_coordinates)
        coordinates_mountain = available_coordinates[:self.MOUNTAIN_NUMS]

        # 在地图上设置这些值
        for i, (x, y) in enumerate(coordinates_our):
            soldier_type = 'special' if i < 2 else 'artillery' if i < 5 else 'infantry'
            self.map[x, y] = 1  # 我方士兵
            cur_soldier = Soldier(i, soldier_type, x, y)
            self.teamOur[i] = cur_soldier
            self.soldier_reward[cur_soldier.id] = self.REWARD_DEFAULT
        for i, (x, y) in enumerate(coordinates_enemy):
            soldier_type = 'special' if i < 2 else 'artillery' if i < 5 else 'infantry'
            self.map[x, y] = 2  # 敌方士兵
            self.teamEnemy[i] = Soldier(i, soldier_type, x, y, 100, 0)

        for x, y in coordinates_mountain:
            self.map[x, y] = -1  # 障碍物山地

    def reset(self):
        """
        重置环境，随机初始化环境
        :return:
        """
        self.__initialize_state()
        self.__init_map()
        self.__init_visual_map()
        return self.render()

    def __build_zone_info(self, row, col):
        role_type = 'mountain' if self.map[row][col] == -1 else 'space'
        return {'pos': {'x': row, 'y': col}, 'roleType': role_type}

    @staticmethod
    def __build_team_info(team_dict):
        team_list = []
        for role_id, soldier in team_dict.items():
            team_list.append({
                'roleType': soldier.type,
                'id': soldier.id,
                'pos': {'x': soldier.x, 'y': soldier.y},
                'health': soldier.hp,
                'life': soldier.life
            })
        return team_list

    @staticmethod
    def __build_enemy_info(team_dict):
        return [{'x': soldier.x, 'y': soldier.y} for role_id, soldier in team_dict.items()]

    def render(self):
        map_list = [self.__build_zone_info(row, col) for row in range(self.width) for col in range(self.height)]
        team_our_list = self.__build_team_info(self.teamOur)
        team_enemy_list = self.__build_enemy_info(self.teamEnemy)

        data = {
            'roundNo': self.cur_round,
            'mapInfo': {
                'height': self.height,
                'width': self.width,
                'zones': map_list
            },
            'players': {
                'teamOur': {
                    'totalScore': self.totalScore,
                    'teamId': self.teamId,
                    'teamName': self.teamName,
                    'roles': team_our_list
                },
                'teamEnemy': {
                    'posList': team_enemy_list
                }
            }
        }
        return data, self.soldier_reward, self.__is_game_done()

    @step_visual_decorator
    def step(self, actions_json_dict):
        self.__init_step()
        target_soldier_pos, can_not_move_soldier_ids = self.__pre_check(actions_json_dict['soldiers'])
        self.__move(target_soldier_pos, can_not_move_soldier_ids)
        # if self.cur_round % 10 == 0:
        #     # 敌军随机移动10步，我们的智能体面对的场景都是敌军不动的
        #     for i in range(10):
        #         self.__enemy_move()
        our_atted_enemies = {}
        # {our_id:{enemy_id:rounds}),}
        for our_id, enemy in self.our_attack_dict.items():
            can_att_enemy_ids_pos = []
            for enemy_id, rounds in enemy.items():
                if abs(self.cur_round - rounds) <= 10 and enemy_id in self.teamEnemy:
                    can_att_enemy_ids_pos.append((enemy_id, self.teamEnemy[enemy_id].x, self.teamEnemy[enemy_id].y))
            our_atted_enemies[our_id] = can_att_enemy_ids_pos
        return self.render(), can_not_move_soldier_ids, our_atted_enemies

    def __is_game_done(self):
        # TODO Reward
        # done = []
        # for soldier_id, _ in self.teamOur.items():
        #     if self.soldier_reward[soldier_id] != 0:
        #         done.append(True)
        #     else:
        #         done.append(False)
        # return done

        num_our = len(self.teamOur)
        if len(self.teamEnemy) == 0:
            for _id, soldier in self.soldier_reward.items():
                self.soldier_reward[_id] += self.REWARD_WIN
                # self.soldier_reward[_id] += self.REWARD_WIN * (1 - self.cur_round / self.MAX_ROUNDS)
            return [True, ] * num_our
        elif num_our == 0:
            return [True, ] * num_our
        elif self.cur_round >= self.MAX_ROUNDS:
            return [False, ] * num_our
        else:
            return [False, ] * num_our

    def __pre_check(self, soldiers):
        movement_mapping = {}  # 存储每个士兵的旧位置和目标位置
        final_positions = {}  # 存储每个士兵的最终位置
        can_not_move_soldier_ids = {}
        # 初始化 movement_mapping 和 final_positions
        for soldier in soldiers:
            role_id = soldier["roleId"]
            old_pos = (self.teamOur[role_id].x, self.teamOur[role_id].y)
            tar_pos = (soldier["posList"][-1]["x"], soldier["posList"][-1]["y"])

            movement_mapping[role_id] = {'old_pos': old_pos, 'tar_pos': tar_pos}
            final_positions[role_id] = tar_pos

        # 循环直到没有更多冲突
        while True:
            conflicts = {}  # 存储冲突信息
            # 检查冲突和异常
            for role_id, positions in movement_mapping.items():
                tar_pos = final_positions[role_id]  # 使用最近的最终位置
                if self.__check_exception(tar_pos):
                    self.soldier_reward[role_id] = self.REWARD_EXCEPTION
                    can_not_move_soldier_ids[role_id] = (self.REWARD_EXCEPTION, positions['old_pos'], tar_pos)
                    final_positions[role_id] = positions['old_pos']
                if tar_pos in conflicts:
                    conflicts[tar_pos].append(role_id)
                else:
                    conflicts[tar_pos] = [role_id]
            # 解决冲突并更新最终位置
            conflict_happened = False
            for tar_pos, role_ids in conflicts.items():
                if len(role_ids) > 1:
                    conflict_happened = True
                    for role_id in role_ids:
                        # 检查是否士兵原地不动
                        if movement_mapping[role_id]['old_pos'] != tar_pos:
                            self.soldier_reward[role_id] = self.REWARD_CONFLICT
                            can_not_move_soldier_ids[role_id] = (-1, movement_mapping[role_id]['old_pos'], tar_pos)
                            final_positions[role_id] = movement_mapping[role_id]['old_pos']

            # 如果没有更多冲突，跳出循环
            if not conflict_happened:
                break

        # 生成最终结果
        result = []
        for role_id, pos in final_positions.items():
            result.append({'role_id': role_id, 'old_pos': movement_mapping[role_id]['old_pos'], 'rel_pos': pos})

        return result, can_not_move_soldier_ids

    def __check_exception(self, pos):
        if min(pos) < 0 or max(pos) >= min(self.width, self.height):
            return True
        target_pos = self.map[pos[0]][pos[1]]
        return target_pos == -1 or target_pos == 2

    def _attack_enemies(self, pos):
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
        enemy_ids_be_attacked = []
        for dx, dy in directions:
            new_x, new_y = pos[0] + dx, pos[1] + dy
            # 超出地图范围
            if min(pos) < 0 or max(pos) >= min(self.width, self.height):
                continue
            # 检查相邻点的值是否为2
            if self.map[new_x][new_y] == 2:
                enemy_ids_be_attacked.append()

        return True

    def __move(self, target_soldier_pos, not_move_soldier_ids):
        # center_points = find_enemy_center(self.teamEnemy)
        enemies_pos = [(soldier.x, soldier.y) for _, soldier in self.teamEnemy.items()]
        for soldier in target_soldier_pos:
            soldier_id = soldier['role_id']
            if soldier_id not in not_move_soldier_ids:
                old_pos = soldier['old_pos']
                rel_pos = soldier['rel_pos']
                # 地图更新
                self.map[old_pos[0]][old_pos[1]] = 0
                self.map[rel_pos[0]][rel_pos[1]] = 1
                # 对于可移动的士兵，也包含非异常非冲突只是移动在原地的点
                # 防止陷入局部最优，站立不动不扣分
                if old_pos == rel_pos:
                    self.soldier_reward[soldier_id] += self.REWARD_NOT_WALK
                # 向敌人中心移动distances_centers / 向任一敌人移动distances_enemies
                # elif distances_centers(old_pos, rel_pos, enemies_pos):
                #     self.soldier_reward[soldier_id] += self.REWARD_WALK_TO_ENEMY
                # 没有向敌人中心移动
                else:
                    self.soldier_reward[soldier_id] += self.REWARD_WALK

                our_soldier = self.teamOur[soldier_id]
                our_soldier.x = rel_pos[0]
                our_soldier.y = rel_pos[1]
                self.__attack(our_soldier)

    def __find_possible_moves(self, enemy_soldier):
        possible_moves = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                # 排除士兵自己的位置 若最终possible_moves为空，则士兵不动
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = enemy_soldier.x + dx, enemy_soldier.y + dy
                # 检查新位置是否在地图范围内并且没有障碍物
                if 0 <= new_x < self.width and 0 <= new_y < self.height:
                    if self.map[new_x][new_y] == 0:
                        possible_moves.append((new_x, new_y))
        return possible_moves

    # @visual_enemy_decorator
    def __enemy_move(self):
        enemy_ids = list(self.teamEnemy.keys())
        for _id in enemy_ids:
            enemy_soldier = self.teamEnemy[_id]
            possible_moves = self.__find_possible_moves(enemy_soldier)
            if possible_moves:
                new_x, new_y = random.choice(possible_moves)
                # 更新士兵的位置
                self.map[enemy_soldier.x][enemy_soldier.y] = 0
                self.map[new_x][new_y] = 2
                enemy_soldier.x = new_x
                enemy_soldier.y = new_y
                self.__enemy_attack(enemy_soldier)

    def __attack(self, soldier: Soldier):
        """
        soldier 攻击周围8个点位，目前假设只要攻击到就可以有回报(必须移动后才会触发攻击)，敌人不会死也不会攻击
        :param soldier:
        :return:
        """
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
        enemy_ids_be_killed = []
        for dx, dy in directions:
            att_x, att_y = soldier.x + dx, soldier.y + dy
            # 攻击超出地图范围，跳过
            if min(att_x, att_y) < 0 or max(att_x, att_y) >= min(self.width, self.height):
                continue
            # 检查相邻点的敌人,并攻击，此处仅触发攻击回报
            if self.map[att_x][att_y] == 2:
                # 找到这个敌方士兵，攻击若杀掉则清除他
                for id, enemy_soldier in self.teamEnemy.items():
                    if (enemy_soldier.x, enemy_soldier.y) == (att_x, att_y):
                        # 判断该我是否在10个回合内攻击过该士兵,True则攻击
                        if self.__check_and_attack(soldier, enemy_soldier):
                            enemy_soldier.hp -= soldier.attack
                        if enemy_soldier.hp <= 0:
                            enemy_ids_be_killed.append(enemy_soldier.id)
        for e_id in enemy_ids_be_killed:
            self.teamEnemy[e_id].life -= 1
            self.teamEnemy[e_id].hp = 100
            if self.teamEnemy[e_id].life < 0:
                self.map[self.teamEnemy[e_id].x][self.teamEnemy[e_id].y] = 0
                self.teamEnemy.pop(e_id)

    def __enemy_attack(self, soldier: Soldier):
        """
        soldier 敌方随机游走攻击周围8个点位，
        :param soldier: enemy
        :return:
        """
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
        our_ids_be_killed = []
        for dx, dy in directions:
            att_x, att_y = soldier.x + dx, soldier.y + dy
            # 攻击超出地图范围，跳过
            if min(att_x, att_y) < 0 or max(att_x, att_y) >= min(self.width, self.height):
                continue
            # 检查相邻点的敌人,并攻击，此处仅触发攻击回报
            if self.map[att_x][att_y] == 1:
                # 找到这个敌方士兵，攻击若杀掉则清除他
                for id, our_soldier in self.teamOur.items():
                    if (our_soldier.x, our_soldier.y) == (att_x, att_y):
                        # 判断该我是否在10个回合内攻击过该士兵
                        if self.__check_and_attack(soldier, our_soldier, our_att=False):
                            our_soldier.hp -= soldier.attack
                        if our_soldier.hp <= 0:
                            self.map[att_x][att_y] = 0
                            our_ids_be_killed.append(our_soldier.id)
        for o_id in our_ids_be_killed:
            self.teamOur[o_id].life -= 1
            self.teamOur[o_id].hp = 100
            if self.teamOur[o_id].life < 0:
                self.teamOur.pop(o_id)

    def __check_and_attack(self, our_soldier, enemy_soldier, our_att=True):
        """
        维护一个攻击列表{our_id:{enemy_id:rounds}),}
        表示该敌方Soldier在rounds回合时被我方our_id攻击过这个id，10个rounds内再攻击不得分。
        :param our_soldier: 我士兵
        :param enemy_soldier: 待判断当前敌方士兵是否可以攻击
        :return: 可攻击则攻击并返回True，不可攻击则False
        """
        if our_att:
            _dict = self.our_attack_dict
        else:
            _dict = self.enemy_attack_dict
        if our_soldier.id not in _dict:
            # 当前我士兵不在攻击列表中，则记录攻击时的回合数并加分
            _dict.update({our_soldier.id: {enemy_soldier.id: self.cur_round}})
            if our_att:
                self.soldier_reward[our_soldier.id] += self.REWARD_ATTACK_ENEMY
            return True
        else:
            check_enemy_id_in_att_list = [_id_with_his_round for _id_with_his_round in _dict[our_soldier.id].keys()]
            # 我方士兵有攻击列表，但攻击列表中没有该敌方士兵，则攻击一下
            if enemy_soldier.id not in check_enemy_id_in_att_list:
                _dict[our_soldier.id].update({enemy_soldier.id: self.cur_round})
                if our_att:
                    self.soldier_reward[our_soldier.id] += self.REWARD_ATTACK_ENEMY
                return True
            else:
                # 我方士兵有攻击列表，且攻击列表中有该敌方士兵，则判断是否在10个回合内攻击过
                for i, (enemy_soldier_id, his_round) in enumerate(_dict[our_soldier.id].items()):
                    if enemy_soldier.id == enemy_soldier_id:
                        if self.cur_round - his_round > 10:
                            # 更新攻击时的回合数，并增加奖励
                            _dict[our_soldier.id][enemy_soldier_id] = self.cur_round
                            if our_att:
                                self.soldier_reward[our_soldier.id] += self.REWARD_ATTACK_ENEMY
                            return True
                        # 在10个回合内攻击过该敌方士兵，则不得分,且此刻处在危险区域
                        else:
                            self.soldier_reward[our_soldier.id] += self.REWARD_DANGER
                            return False
