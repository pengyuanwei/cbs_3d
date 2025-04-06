import os
import math

from utilities.utilities import *
from utilities.visualizor import Simulator


if __name__ == '__main__':
    n_agents = 8
    delta_time = (0.002 * math.sqrt(3)) / 0.1

    global_model_dir_1 = 'D:/PythonProjects/AcousticLevitationEnvironment/examples/experiments/experiment_20/20_19_98_99'
    save_dir = os.path.join(global_model_dir_1, 'cbs')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name_0 = 'planner_v2/path_'
    file_name_1 = 'solution_'

    num_file = 30
    for n in range(num_file, 100):
        print(f'\n-----------------------The paths {n}-----------------------')

        csv_file = os.path.join(global_model_dir_1, f'{file_name_0}{str(n)}.csv')
        #csv_file = 'F:\Desktop\Projects\AcousticLevitationGym\examples\experiments\S2M2_8_experiments\data0.csv'
        csv_data = read_csv_file(csv_file)
        if csv_data == None:
            print(f"Skipping file due to read failure: {csv_file}")
            continue

        processed_data = read_paths(csv_data)
        # numpy: (n_agents, xyz)
        sources, targets = read_sources_targets(processed_data)
        transformed_sources = transform_coordinates(sources)
        transformed_targets = transform_coordinates(targets)

        # 输入读取的起点和终点
        Simulator.given_scenario(transformed_sources, transformed_targets)

        r = Simulator(n_agents, three_dimensional=True)
        if r.path.size > 0:  # 判断是否为空
            solution = inverse_transform(sources, targets, r.path)
            save_path = os.path.join(save_dir, f'{file_name_1}{str(n)}.csv')
            save_solution(solution, save_path, n_agents, delta_time)
        else:
            print("Warning: r.path is empty!")