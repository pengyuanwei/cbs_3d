#!/usr/bin/env python3
'''
Modified based on [Multi-Agent Path Finding](https://github.com/GavinPHR/Multi-Agent-Path-Finding)
Copyright (c) 2020 [Haoran Peng]
Copyright (c) 2025 [Pengyuan Wei]
Released under the MIT License
'''
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from cbs_3d.planner import Planner


class Simulator:
    def __init__(self, n_agent: int=4, three_dimensional: bool=False):
        self.n_agent = n_agent
        # Transform the vertices to be border-filled rectangles
        if not three_dimensional:
            static_obstacles = self.vertices_to_obsts(RECT_OBSTACLES)
        else:
            static_obstacles = self.vertices_to_obsts_3d(RECT_OBSTACLES)

        # Call cbs_3d to plan
        self.planner = Planner(GRID_SIZE, ROBOT_RADIUS, static_obstacles, three_dimensional)
        before = time.time()
        # Synchronous paths. A numpy: (agent_idx, path_length, xy)
        self.path = self.planner.plan(START, GOAL, debug=True)
        after = time.time()
        self.elapsed_time = after - before  # 计算运行时间
        print('Time elapsed:', "{:.4f}".format(after-before), 'second(s)')

        self.success = self.path.size > 0

    # Transform opposite vertices of rectangular obstacles into obstacles
    @staticmethod
    def vertices_to_obsts(obsts):
        def drawRect(v0, v1):
            o = []
            base = abs(v0[0] - v1[0])
            side = abs(v0[1] - v1[1])
            for xx in range(0, base, GRID_SIZE):
                o.append((v0[0] + xx, v0[1]))
                o.append((v0[0] + xx, v0[1] + side - 1))
            o.append((v0[0] + base, v0[1]))
            o.append((v0[0] + base, v0[1] + side - 1))
            for yy in range(0, side, GRID_SIZE):
                o.append((v0[0], v0[1] + yy))
                o.append((v0[0] + base - 1, v0[1] + yy))
            o.append((v0[0], v0[1] + side))
            o.append((v0[0] + base - 1, v0[1] + side))
            return o
        
        static_obstacles = []
        for vs in obsts.values():
            static_obstacles.extend(drawRect(vs[0], vs[1]))
        return static_obstacles

    # Transform opposite vertices of cuboid obstacles into obstacle points on edges
    @staticmethod
    def vertices_to_obsts_3d(obsts):
        def drawCube(v0, v1):
            o = []
            # 规范坐标确保有序
            x_min, x_max = sorted([v0[0], v1[0]])
            y_min, y_max = sorted([v0[1], v1[1]])
            z_min, z_max = sorted([v0[2], v1[2]])
            # 生成12条边的起点和终点
            edges = [
                # 底面（z=z_min）
                ((x_min, y_min, z_min), (x_max, y_min, z_min)),  # 底边x轴
                ((x_max, y_min, z_min), (x_max, y_max, z_min)),  # 右边y轴
                ((x_min, y_max, z_min), (x_max, y_max, z_min)),  # 顶边x轴反向
                ((x_min, y_min, z_min), (x_min, y_max, z_min)),  # 左边y轴反向
                # 顶面（z=z_max）
                ((x_min, y_min, z_max), (x_max, y_min, z_max)),
                ((x_max, y_min, z_max), (x_max, y_max, z_max)),
                ((x_min, y_max, z_max), (x_max, y_max, z_max)),
                ((x_min, y_min, z_max), (x_min, y_max, z_max)),
                # 垂直边（连接底面和顶面）
                ((x_min, y_min, z_min), (x_min, y_min, z_max)),  # 左侧z轴
                ((x_max, y_min, z_min), (x_max, y_min, z_max)),  # 右侧z轴
                ((x_max, y_max, z_min), (x_max, y_max, z_max)),  # 右侧z轴
                ((x_min, y_max, z_min), (x_min, y_max, z_max)),  # 左侧z轴
            ]
            # 沿边生成离散点
            for (start, end) in edges:
                if start[0] != end[0]:  # x轴边
                    current = start[0]
                    while current <= end[0]:
                        o.append((current, start[1], start[2]))
                        current += GRID_SIZE
                    if o[-1] != end:  # 补充终点
                        o.append(end)
                elif start[1] != end[1]:  # y轴边
                    current = start[1]
                    while current <= end[1]:
                        o.append((start[0], current, start[2]))
                        current += GRID_SIZE
                    if o[-1] != end:
                        o.append(end)
                elif start[2] != end[2]:  # z轴边
                    current = start[2]
                    while current <= end[2]:
                        o.append((start[0], start[1], current))
                        current += GRID_SIZE
                    if o[-1] != end:
                        o.append(end)
            return o

        static_obstacles = []
        for vs in obsts.values():
            static_obstacles.extend(drawCube(vs[0], vs[1]))
        return static_obstacles

    @staticmethod
    def load_scenario(fd):
        with open(fd, 'r', encoding='utf-8') as f:
            global GRID_SIZE, ROBOT_RADIUS, RECT_OBSTACLES, START, GOAL
            data = yaml.load(f, Loader=yaml.FullLoader)
            GRID_SIZE = data['GRID_SIZE']
            ROBOT_RADIUS = data['ROBOT_RADIUS']
            RECT_OBSTACLES = data['RECT_OBSTACLES']
            START = data['START']
            GOAL = data['GOAL']
        if RECT_OBSTACLES is None:
            raise ValueError("RECT_OBSTACLES cannot be None. At least the map boundaries need to be defined.")

    @staticmethod
    def random_scenario(n_agents:int, anonymous_mapf:bool=True):
        global GRID_SIZE, ROBOT_RADIUS, RECT_OBSTACLES, START, GOAL
        GRID_SIZE = 20
        ROBOT_RADIUS = 75
        RECT_OBSTACLES = {0: [[0, 0, 0], [1349, 1349, 1349]]}
        def create_points(N, cube_size=1349, max_attempts=1000):
            points = []
            attempts = 0
            while len(points) < N and attempts < max_attempts:
                # 在[76, 1274)范围内随机生成一个点
                point = np.random.randint(76, 1274, 3)  # 生成一个 NumPy 数组
                point_tuple = tuple(point.tolist())  # 转换为标准 Python int 类型
                # 检查与已生成点之间的椭球体距离
                if all(np.sum(((point - p) / np.array([150, 150, 300]))**2) > 1 for p in points):
                    points.append(point_tuple)
                attempts += 1
                if attempts == max_attempts:
                    raise RuntimeError("Reached maximum attempts without generating enough points.")
            return np.array(points)
        start_positions = create_points(n_agents)
        target_positions = create_points(n_agents)
        if anonymous_mapf:
            row_indices, col_indices = Simulator.optimal_pairing(start_positions, target_positions)
            START = start_positions[row_indices]
            GOAL = target_positions[col_indices]
        else:
            START = start_positions
            GOAL = target_positions

    @staticmethod
    def optimal_pairing(coords1, coords2):
        # 计算两组坐标之间的距离矩阵
        distance_matrix = cdist(coords1, coords2, metric='euclidean')
        # 使用匈牙利算法返回最优匹配
        return linear_sum_assignment(distance_matrix)

    @staticmethod
    def given_scenario(sources, targets):
        global GRID_SIZE, ROBOT_RADIUS, RECT_OBSTACLES, START, GOAL
        GRID_SIZE = 20
        ROBOT_RADIUS = 75
        RECT_OBSTACLES = {0: [[0, 0, 1200], [1349, 1349, 2549]]}
        START = sources
        GOAL = targets

    def show(self):
        particle_paths = self.path

        # 创建 3D 图形
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 初始化粒子和轨迹
        particles, = ax.plot([], [], [], 'ro', markersize=8)
        trajectories = [ax.plot([], [], [], lw=1)[0] for _ in range(self.n_agent)]

        # 设置坐标轴范围
        ax.set_xlim([RECT_OBSTACLES[0][0][0], RECT_OBSTACLES[0][1][0]])
        ax.set_ylim([RECT_OBSTACLES[0][0][1], RECT_OBSTACLES[0][1][1]])
        ax.set_zlim([RECT_OBSTACLES[0][0][2], RECT_OBSTACLES[0][1][2]])
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title(f"{self.n_agent} Particles Moving in Different Trajectories")

        # 初始化函数
        def init():
            particles.set_data([], [])
            particles.set_3d_properties([])
            for trajectory in trajectories:
                trajectory.set_data([], [])
                trajectory.set_3d_properties([])
            return [particles] + trajectories

        # 更新函数
        def update(frame):
            x, y, z = particle_paths[:, frame, 0], particle_paths[:, frame, 1], particle_paths[:, frame, 2]
            particles.set_data(x, y)
            particles.set_3d_properties(z)
            
            for i, trajectory in enumerate(trajectories):
                trajectory.set_data(particle_paths[i, :frame+1, 0], particle_paths[i, :frame+1, 1])
                trajectory.set_3d_properties(particle_paths[i, :frame+1, 2])
            
            return [particles] + trajectories

        # 显示图形，并等待用户交互
        plt.draw()  # 先绘制静态图像
        plt.waitforbuttonpress()  # 等待用户点击窗口或按键
        # 创建动画
        ani = animation.FuncAnimation(fig, update, frames=particle_paths.shape[1], init_func=init, blit=False, interval=50, repeat=False)
        # 保存动画为 MP4
        ani.save("animation.gif", writer=PillowWriter(fps=20))
        plt.show()