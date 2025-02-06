#!/usr/bin/env python3
'''
Modified based on [Multi-Agent Path Finding](https://github.com/GavinPHR/Multi-Agent-Path-Finding)
Copyright (c) 2020 [Haoran Peng]
Copyright (c) 2025 [Pengyuan Wei]
Released under the MIT License
'''
from cbs_3d.visualizor import Simulator


if __name__ == '__main__':
    # From command line, call: python3 visualizer.py scenario1.yaml
    # The approach supports defining rectangular obstacles. The first obstacle is the boundary of the map.
    # Simulator.load_scenario(sys.argv[1])

    # 随机生成起点和终点
    n_agent = 8
    Simulator.random_scenario(n_agent)
    r = Simulator(n_agent, three_dimensional=True)
    r.show()