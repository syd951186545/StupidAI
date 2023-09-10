import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# 创建示例的21x21x3 ndarray，假设它包含了士兵和山地的位置信息
# 0表示空地，1表示我方士兵，2表示敌方士兵，-1表示山地

def check_map(data):
    # 创建一个21x21的栅格图
    grid_size = 21
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(0, grid_size, 1))
    ax.set_yticks(np.arange(0, grid_size, 1))
    ax.grid(linestyle='--', color='lightgrey')
    # 绘制栅格图
    for x in range(grid_size):
        for y in range(grid_size):
            if data[x, y, 0] == 1 or data[x, y, 0] == -1:  # Mountain
                ax.add_patch(Rectangle((x, y), 1, 1, color='black'))
            elif data[x, y, 1] == 1:  # Our soldier
                ax.add_patch(Rectangle((x, y), 1, 1, color='blue'))
            elif data[x, y, 2] == 1:  # Enemy soldier
                ax.add_patch(Rectangle((x, y), 1, 1, color='red'))

    # 设置坐标轴范围和纵横比
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal', adjustable='box')

    plt.show()
