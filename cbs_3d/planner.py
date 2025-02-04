#!/usr/bin/env python3
'''
Modified based on [Multi-Agent Path Finding](https://github.com/GavinPHR/Multi-Agent-Path-Finding)
Copyright (c) 2020 [Haoran Peng]
Copyright (c) 2025 [Pengyuan Wei]
Released under the MIT License

An implementation of multi-agent path finding using conflict-based search [Sharon et al., 2015]
'''
from typing import List, Tuple, Dict, Callable, Set
import multiprocessing as mp
from heapq import heappush, heappop
from itertools import combinations
from copy import deepcopy
import numpy as np

# The low level planner for CBS is the Space-Time A* planner
# https://github.com/pengyuanwei/space_time_a_star
from space_time_a_star.planner import Planner as STPlanner

from .constraint_tree import CTNode
from .constraints import Constraints
from .agent import Agent
from .assigner import *

class Planner:
    def __init__(self, 
                 grid_size: int,
                 robot_radius: int,
                 static_obstacles: List[Tuple[int, ...]],
                 three_dimensional: bool=False):

        self.robot_radius = robot_radius
        self.three_dimensional = three_dimensional
        self.st_planner = STPlanner(grid_size, robot_radius, static_obstacles, three_dimensional)

    '''
    You can use your own assignment function, the default algorithm greedily assigns
    the closest goal to each start.
    '''
    def plan(self, starts: List[Tuple[int, ...]],
                   goals: List[Tuple[int, ...]],
                   assign: Callable = min_cost,
                   max_iter: int = 200,
                   low_level_max_iter: int = 100,
                   max_process: int = 10,
                   debug: bool = False) -> np.ndarray:

        self.low_level_max_iter = low_level_max_iter
        self.debug = debug

        # Do goal assignment
        self.agents = assign(starts, goals)

        constraints = Constraints()

        # Compute path for each agent using low level planner
        solution = dict((agent, self.calculate_path(agent, constraints, None)) for agent in self.agents)

        open = []  # 存储待扩展节点的优先队列（最小堆）
        if all(len(path) != 0 for path in solution.values()):
            # Make root node
            node = CTNode(constraints, solution)
            # Min heap for quick extraction
            open.append(node)

        manager = mp.Manager()  # 跨进程数据共享
        iter_ = 0
        while open and iter_ < max_iter:
            iter_ += 1

            results = manager.list([])  # manager.list(): 生成一个进程安全的共享列表，所有子进程可向其追加数据，主进程通过该列表汇总所有子进程的计算结果
            processes = []  # 存储创建的进程对象，便于统一管理

            # 启动多个进程处理节点 (Default to 10 processes maximum)      
            # 每次循环从 open 队列中弹出一个节点（heappop），创建一个进程处理该节点：
            #   target=self.search_node：指定子进程执行的方法。
            #   args=[node, results]：传递当前节点和共享结果列表。      
            for _ in range(min(max_process, len(open))):
                p = mp.Process(target=self.search_node, args=[heappop(open), results])
                # 将进程对象存入 processes 列表，并立即启动（p.start()）
                processes.append(p)
                p.start()

            # 等待进程完成并收集结果
            for p in processes:
                # 在 multiprocessing 中，.join() 的作用为“等待子进程完成任务后再继续主进程”，是多进程编程中协调并行任务的核心机制。
                p.join()

            for result in results:
                if len(result) == 1:
                    if debug:
                        print('cbs_3d: Paths found after about {0} iterations'.format(4 * iter_))
                    return result[0]
                if result[0]:
                    heappush(open, result[0])
                if result[1]:
                    heappush(open, result[1])

        if debug:
            print('cbs_3d: Open set is empty, no paths found.')
        return np.array([])

    '''
    Abstracted away the cbs search for multiprocessing.
    The parameters open and results MUST BE of type ListProxy to ensure synchronization.
    '''
    def search_node(self, best: CTNode, results):
        agent_i, agent_j, time_of_conflict = self.validate_paths(self.agents, best)

        # If there is not conflict, validate_paths returns (None, None, -1)
        if agent_i is None:
            results.append((self.reformat(self.agents, best.solution),))
            return
        # Calculate new constraints
        agent_i_constraint = self.calculate_constraints(best, agent_i, agent_j, time_of_conflict)
        agent_j_constraint = self.calculate_constraints(best, agent_j, agent_i, time_of_conflict)

        # Calculate new paths
        agent_i_path = self.calculate_path(agent_i,
                                           agent_i_constraint,
                                           self.calculate_goal_times(best, agent_i, self.agents))
        agent_j_path = self.calculate_path(agent_j,
                                           agent_j_constraint,
                                           self.calculate_goal_times(best, agent_j, self.agents))

        # Replace old paths with new ones in solution
        solution_i = best.solution
        solution_j = deepcopy(best.solution)
        solution_i[agent_i] = agent_i_path
        solution_j[agent_j] = agent_j_path

        node_i = None
        if all(len(path) != 0 for path in solution_i.values()):
            node_i = CTNode(agent_i_constraint, solution_i)

        node_j = None
        if all(len(path) != 0 for path in solution_j.values()):
            node_j = CTNode(agent_j_constraint, solution_j)

        results.append((node_i, node_j))


    '''
    Pair of agent, point of conflict
    '''
    def validate_paths(self, agents, node: CTNode):
        # Check collision pair-wise
        for agent_i, agent_j in combinations(agents, 2):  # 生成所有智能体对
            time_of_conflict = self.safe_distance(node.solution, agent_i, agent_j)
            # time_of_conflict=-1 if there is not conflict
            if time_of_conflict == -1:
                continue
            return agent_i, agent_j, time_of_conflict
        return None, None, -1


    def safe_distance(self, solution: Dict[Agent, np.ndarray], agent_i: Agent, agent_j: Agent) -> int:
        for idx, (point_i, point_j) in enumerate(zip(solution[agent_i], solution[agent_j])):
            if self.elliptical_distance_sq(point_i, point_j) > 1:
                continue
            return idx
        return -1

    @staticmethod
    def dist(point1: np.ndarray, point2: np.ndarray) -> int:
        return int(np.linalg.norm(point1-point2, 2))  # L2 norm

    def elliptical_distance_sq(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        计算椭球体归一化距离平方
        - 水平方向安全半径: 2 * self.robot_radius
        - 垂直方向安全半径: 4 * self.robot_radius (水平的两倍)
        """
        dx = (p1[0] - p2[0]) / (2 * self.robot_radius)  # 水平方向缩放
        dy = (p1[1] - p2[1]) / (2 * self.robot_radius)
        dz = 0
        if self.three_dimensional and p1.size > 2 and p2.size > 2:
            dz = (p1[2] - p2[2]) / (4 * self.robot_radius)  # 垂直方向缩放为水平的两倍
        return dx**2 + dy**2 + dz**2
    
    def calculate_constraints(self, node: CTNode,
                                    constrained_agent: Agent,
                                    unchanged_agent: Agent,
                                    time_of_conflict: int) -> Constraints:
        contrained_path = node.solution[constrained_agent]
        unchanged_path = node.solution[unchanged_agent]
        pivot = unchanged_path[time_of_conflict]  # 冲突坐标
        conflict_end_time = time_of_conflict

        # 扩展冲突时间段（直到两路径分离）
        try:
            while self.elliptical_distance_sq(contrained_path[conflict_end_time], pivot) <= 1:
                conflict_end_time += 1
        except IndexError:
            pass  # 路径越界时终止

        # 生成时空约束：禁止 constrained_agent 在 [time_of_conflict, conflict_end_time) 时间段内进入 pivot 位置
        return node.constraints.fork(constrained_agent, tuple(pivot.tolist()), time_of_conflict, conflict_end_time)

    def calculate_goal_times(self, node: CTNode, agent: Agent, agents: List[Agent]):
        '''
        记录其他智能体到达目标的时间和位置，作为semi_dynamic_obstacles
        '''
        solution = node.solution
        goal_times = dict()
        for other_agent in agents:
            if other_agent == agent:
                continue
            time = len(solution[other_agent]) - 1
            goal_times.setdefault(time, set()).add(tuple(solution[other_agent][time]))
        return goal_times

    '''
    Calculate the paths for all agents with space-time constraints
    '''
    def calculate_path(self, agent: Agent, 
                       constraints: Constraints, 
                       goal_times: Dict[int, Set[Tuple[int, ...]]]) -> np.ndarray:
        return self.st_planner.plan(agent.start, 
                                    agent.goal, 
                                    constraints.setdefault(agent, dict()), 
                                    semi_dynamic_obstacles=goal_times,
                                    max_iter=self.low_level_max_iter, 
                                    debug=self.debug)

    '''
    Reformat the solution to a numpy array
    '''
    @staticmethod
    def reformat(agents: List[Agent], solution: Dict[Agent, np.ndarray]):
        solution = Planner.pad(solution)
        reformatted_solution = []
        for agent in agents:
            reformatted_solution.append(solution[agent])
        return np.array(reformatted_solution)

    '''
    Pad paths to equal length, inefficient but well..
    '''
    @staticmethod
    def pad(solution: Dict[Agent, np.ndarray]):
        max_ = max(len(path) for path in solution.values())
        for agent, path in solution.items():
            if len(path) == max_:
                continue
            padded = np.concatenate([path, np.array(list([path[-1]])*(max_-len(path)))])
            solution[agent] = padded
        return solution