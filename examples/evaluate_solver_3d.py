#!/usr/bin/env python3
'''
Modified based on [Multi-Agent Path Finding](https://github.com/GavinPHR/Multi-Agent-Path-Finding)
Copyright (c) 2020 [Haoran Peng]
Copyright (c) 2025 [Pengyuan Wei]
Released under the MIT License
'''
import time
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

from cbs_3d.planner import Planner
from visualization_3d import Simulator


if __name__ == '__main__':

    n_agents = [4, 6, 8, 10]
    mean_time_list = []
    std_time_list = []
    mean_length_list = []
    std_length_list = []
    success_num_list = []
    for i in n_agents:
        computation_times = []
        path_lengths = []

        mean_time = 0.0
        std_time = 0.0
        mean_length = 0.0
        std_length = 0.0
        success_num = 0
        for iter_ in range(200):
            print("The senario No.", iter_)
            # 随机生成起点和终点
            Simulator.random_scenario(i)
            r = Simulator(i, three_dimensional=True)
            computation_times.append(r.elapsed_time)
            if r.path.size > 0:  # 判断是否为空
                path_lengths.append(len(r.path[0]))
            else:
                print("Warning: r.path is empty!")
            success_num += int(r.success)

        mean_time = np.mean(computation_times)
        std_time = np.std(computation_times)
        mean_length = np.mean(path_lengths)
        std_length = np.std(path_lengths)

        mean_time_list.append(mean_time)
        std_time_list.append(std_time)
        mean_length_list.append(mean_length)
        std_length_list.append(std_length)
        success_num_list.append(success_num)

    for i in range(len(n_agents)):
        print("--------------------------------------------")
        print(f"Scenario: {n_agents[i]} particles")
        print(f"平均时间: {mean_time_list[i]:.6f} 秒")
        print(f"标准差: {std_time_list[i]:.6f} 秒")
        print(f"平均长度: {mean_length_list[i]}")
        print(f"标准差: {std_length_list[i]}") 
        print(f'The success number: {success_num_list[i]}')