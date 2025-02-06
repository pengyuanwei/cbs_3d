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
    n_agent = 8

    computation_times = []
    path_lengths = []
    success_num = 0
    for iter_ in range(500):
        print("The senario No.", iter_)
        # 随机生成起点和终点
        Simulator.random_scenario(n_agent)
        r = Simulator(n_agent, three_dimensional=True)
        computation_times.append(r.elapsed_time)
        if r.path.size > 0:  # 判断是否为空
            path_lengths.append(len(r.path[0]))
        else:
            print("Warning: r.path is empty!")
        success_num += int(r.success)

    mean_time = np.mean(computation_times)
    std_time = np.std(computation_times)
    print(f"平均时间: {mean_time:.6f} 秒")
    print(f"标准差: {std_time:.6f} 秒")

    mean_length = np.mean(path_lengths)
    std_length = np.std(path_lengths)
    print(f"平均长度: {mean_length}")
    print(f"标准差: {std_length}") 

    print(f'The success number: {success_num}')