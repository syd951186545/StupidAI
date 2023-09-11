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

    def __init__(self, soldier_id, soldier_type, x, y, health=100, life=2):
        self.id = soldier_id
        self.type = soldier_type
        self.x = x
        self.y = y
        self.hp = health
        self.attack = self.type_attributes[soldier_type]["attack"]
        self.max_respawn = self.type_attributes[soldier_type]["max_respawn"]
        self.life = life  # 当前角色剩余命数，相当于复活次数


def find_enemy_center(teamEnemy):
    """
    :param teamEnemy: Soldier Map {id:Soldier(id)}
    :return: center point list
    """
    coordinates = np.array([[soldier.x, soldier.y] for soldier in teamEnemy.values()])
    if len(teamEnemy) > 5:
        n_clusters = 5
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
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

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
            self.title_round = self.fig.text(0.02, 0.98, f'当前回合={self.cur_round}', verticalalignment='top',
                                             horizontalalignment='left')
        else:
            self.title_round.remove()
            self.title_round = self.fig.text(0.02, 0.98, f'当前回合={self.cur_round}', verticalalignment='top',
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
            our = plt.Rectangle((soldier.x, soldier.y), 1, side_length, color='darkblue')
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
                                    arrowstyle='-|>', mutation_scale=15, color='purple')
            self.ax.add_patch(arrow)
            self.elements_to_clear.append(arrow)

            reward_label = self.ax.text(tar_x, tar_y, str(reward), fontsize=9, color='purple')
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
            our = plt.Rectangle((soldier.x, soldier.y), 1, side_length, color='darkblue')
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
                our = plt.Rectangle((soldier.x, soldier.y), 1, side_length, color='darkblue')
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
                                            arrowstyle='-|>', mutation_scale=15, color='purple')
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
                our = plt.Rectangle((soldier.x, soldier.y), 1, side_length, color='darkblue')
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
    REWARD_WIN = 100
    REWARD_LOSE = -20
    REWARD_DEFAULT = 0
    REWARD_ATTACK_ENEMY = 2
    REWARD_WALK_TO_ENEMY = 0
    REWARD_WALK = 0
    REWARD_NOT_WALK = -1
    REWARD_DANGER = -1
    REWARD_CONFLICT = -1
    REWARD_EXCEPTION = -2
    height, width = 21, 21
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
        self.title_round = self.fig.text(0.02, 0.98, f'当前回合={self.cur_round}', verticalalignment='top',
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
                    our = plt.Rectangle((x, y), 1, 1, color='darkblue')
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
        our_side_coordinates = [(x, y) for x in range(self.width // 3) for y in range(self.height)]
        enemy_side_coordinates = [(x, y) for x in range(self.width // 3 * 2, self.width) for y in range(self.height)]

        # 随机打乱两个区域的坐标点列表
        np.random.shuffle(our_side_coordinates)
        np.random.shuffle(enemy_side_coordinates)
        np.random.shuffle(all_coordinates)

        # 是否双方各自区域内随机选取士兵
        two_side = False
        if two_side:
            # 在各自区域内选取10个位置
            coordinates_our = our_side_coordinates[:10]
            coordinates_enemy = enemy_side_coordinates[:10]
        else:
            coordinates_our = all_coordinates[:10]
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
            self.teamEnemy[i] = Soldier(i, soldier_type, x, y, 100, 2)

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
        target_soldier_pos, conflicts, exception = self.__pre_check(actions_json_dict['soldiers'])
        self.__move(target_soldier_pos)
        # if self.cur_round % 10 == 0:
        #     # 敌军随机移动10步，我们的智能体面对的场景都是敌军不动的
        #     for i in range(10):
        #         self.__enemy_move()
        return self.render()

    def __is_game_done(self):
        # TODO Reward
        if len(self.teamEnemy) == 0:
            for _id, soldier in self.soldier_reward.items():
                self.soldier_reward[_id] += self.REWARD_WIN * (1 - self.cur_round / self.MAX_ROUNDS)
            return [True, ] * 10
        elif len(self.teamOur) == 0:
            return [True, ] * 10
        elif self.cur_round >= self.MAX_ROUNDS:
            return [True, ] * 10
        else:
            return [False, ] * 10

    def __pre_check(self, soldiers):
        target_pos_soldier = {}  # 无冲突无异常，可移动的点位：id{(x,y):id} 用以记录非冲突点位
        hold_soldier = {}  # 确定停在原地的士兵 点位：id{(x,y):id}
        soldiers_move_to_target = {}  # 无冲突无异常，同步记录id加点位{id:(x,y)} 并返回这个
        conflicts = set()  # 冲突点位（x,y）
        exception = []  # 异常的士兵 id
        for soldier in soldiers:
            role_id = soldier["roleId"]
            for pos in soldier["posList"]:
                tar_x, tar_y = pos["x"], pos["y"]
                # 检测到超边界、山地、冲撞敌方士兵的点位，将其加入异常列表
                if self.__check_exception((tar_x, tar_y)):
                    exception.append(role_id)
                    # 异常回报
                    self.soldier_reward[role_id] += self.REWARD_EXCEPTION
                    # 异常士兵【最终在原地】
                    hold_x, hold_y = self.teamOur[role_id].x, self.teamOur[role_id].y
                    hold_soldier[(hold_x, hold_y)] = role_id
                # 检测到当前目标与【最终在原地】不动的士兵冲突
                elif (tar_x, tar_y) in hold_soldier:
                    conflicts.add((tar_x, tar_y))
                    self.soldier_reward[role_id] += self.REWARD_CONFLICT
                    # 冲撞友方士兵，移动在原地
                    hold_soldier[(tar_x, tar_y)] = role_id
                # 无冲突、无冲撞，但是目标点位也有士兵要去
                elif (tar_x, tar_y) in target_pos_soldier:
                    conflicts.add((tar_x, tar_y))
                    conflict_id = target_pos_soldier.pop((tar_x, tar_y))
                    # 冲突回报
                    self.soldier_reward[role_id] += self.REWARD_CONFLICT
                    self.soldier_reward[conflict_id] += self.REWARD_CONFLICT
                else:
                    target_pos_soldier[(tar_x, tar_y)] = role_id
        for pos, _id in target_pos_soldier.items():
            soldiers_move_to_target.update({_id: pos})

        return soldiers_move_to_target, conflicts, exception

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

    def __move(self, target_soldier_pos):
        center_points = find_enemy_center(self.teamEnemy)
        enemies_pos = [(soldier.x, soldier.y) for _, soldier in self.teamEnemy.items()]
        for _, soldier in self.teamOur.items():
            # 对于所有士兵，在地图中擦除
            self.map[soldier.x][soldier.y] = 0
        for _, soldier in self.teamOur.items():
            # 对于所有士兵，若移动则更新其位置
            if soldier.id in target_soldier_pos.keys():
                # 防止陷入局部最优，站立不动不扣分
                if (soldier.x, soldier.y) == target_soldier_pos[soldier.id]:
                    self.soldier_reward[soldier.id] += self.REWARD_NOT_WALK
                # 向敌人中心移动distances_centers / 向任一敌人移动distances_enemies
                elif distances_enemies((soldier.x, soldier.y), target_soldier_pos[soldier.id], enemies_pos):
                    self.soldier_reward[soldier.id] += self.REWARD_WALK_TO_ENEMY
                # 没有向敌人中心移动
                else:
                    self.soldier_reward[soldier.id] += self.REWARD_WALK

                soldier.x = target_soldier_pos[soldier.id][0]
                soldier.y = target_soldier_pos[soldier.id][1]
                self.__attack(soldier)

        for _, soldier in self.teamOur.items():
            # 对于所有更新后士兵，再将其放在地图上
            self.map[soldier.x][soldier.y] = 1

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
                        # 判断该我是否在10个回合内攻击过该士兵,True则
                        if self.__check_and_attack(soldier, enemy_soldier):
                            enemy_soldier.hp -= soldier.attack
                        if enemy_soldier.hp <= 0:
                            enemy_ids_be_killed.append(enemy_soldier.id)
        for e_id in enemy_ids_be_killed:
            self.teamEnemy[e_id].life -= 1
            self.teamEnemy[e_id].hp = 100
            if self.teamEnemy[e_id].life <= 0:
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
            if self.teamOur[o_id].life <= 0:
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
