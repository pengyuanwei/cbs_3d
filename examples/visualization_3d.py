#!/usr/bin/env python3
'''
Copyright (c) 2025 [Pengyuan Wei]
Released under the MIT License
'''
import time
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cbs_3d.planner import Planner


class Simulator:
    def __init__(self):
        # Transform the vertices to be border-filled rectangles
        static_obstacles = self.vertices_to_obsts(RECT_OBSTACLES)

        # Call cbs_3d to plan
        self.planner = Planner(GRID_SIZE, ROBOT_RADIUS, static_obstacles)
        before = time.time()
        # A numpy: (agent_idx, path_length, xy)
        self.path = self.planner.plan(START, GOAL, debug=False)
        after = time.time()
        print('Time elapsed:', "{:.4f}".format(after-before), 'second(s)')

    '''
    Transform opposite vertices of rectangular obstacles into obstacles
    '''
    @staticmethod
    def vertices_to_obsts(obsts):
        def drawRect(v0, v1):
            o = []
            base = abs(v0[0] - v1[0])
            side = abs(v0[1] - v1[1])
            for xx in range(0, base, 30):
                o.append((v0[0] + xx, v0[1]))
                o.append((v0[0] + xx, v0[1] + side - 1))
            o.append((v0[0] + base, v0[1]))
            o.append((v0[0] + base, v0[1] + side - 1))
            for yy in range(0, side, 30):
                o.append((v0[0], v0[1] + yy))
                o.append((v0[0] + base - 1, v0[1] + yy))
            o.append((v0[0], v0[1] + side))
            o.append((v0[0] + base - 1, v0[1] + side))
            return o
        static_obstacles = []
        for vs in obsts.values():
            static_obstacles.extend(drawRect(vs[0], vs[1]))
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
        
    def show(self):
        # 通过填充0将路径改为三维路径
        particle_paths = np.pad(self.path, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)

        # 创建 3D 图形
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 初始化粒子和轨迹
        particles, = ax.plot([], [], [], 'ro', markersize=8)
        trajectories = [ax.plot([], [], [], lw=1)[0] for _ in range(4)]

        # 设置坐标轴范围
        ax.set_xlim([RECT_OBSTACLES[0][0][0], RECT_OBSTACLES[0][1][0]])
        ax.set_ylim([RECT_OBSTACLES[0][0][1], RECT_OBSTACLES[0][1][1]])
        ax.set_zlim([-2, 2])
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("Four Particles Moving in Different Trajectories")

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
        plt.show()


if __name__ == '__main__':
    # From command line, call: python3 visualizer.py scenario1.yaml
    # The approach supports defining rectangular obstacles. The first obstacle is the boundary of the map.
    Simulator.load_scenario(sys.argv[1])

    r = Simulator()
    r.show()