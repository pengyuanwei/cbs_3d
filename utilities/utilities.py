import os
import csv
import numpy as np

def read_csv_file(file_path):
    '''
    读取.csv格式的路径
    '''
    if not os.path.exists(file_path):
        print(f"错误：文件路径无效 -> {file_path}")
        return None
    
    data_list = []
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            # 过滤掉空字符串
            filtered_row = [item for item in row if item.strip()]
            data_list.append(filtered_row)
    return data_list


def read_paths(csv_data):
    '''
    读取等长路径并转换为浮点型数组，同时检测是否包含 NaN 值。
    参数:
        csv_data (list of list): 输入的 CSV 数据，每行应该包含 5 个值。
    返回:
        tuple: 包含两部分 (numpy array of floats, bool indicating if NaN is present)
    '''
    try:
        # 转换为浮点数组，并过滤长度不是5的行
        csv_data_float = [
            [float(element) for element in row] 
            for row in csv_data if len(row) == 5
        ]
        
        # 转换为 numpy 数组
        csv_array = np.array(csv_data_float)
        # 检查是否存在 NaN 值
        if np.isnan(csv_array).any():
            return np.array([])
        
        # 处理轨迹
        # 每个粒子的轨迹长度相同
        paths_length = int(csv_data[1][1])
        # split_data_numpy的形状为(n_particles, n_keypoints, 5)
        # When axis=2: keypoints_id, time, x, y, z
        split_data = csv_array.reshape(-1, paths_length, 5)
        
        return split_data

    except ValueError:
        # 捕获不能转换为浮点数的异常
        return np.array([])
    
def read_sources_targets(processed_data):
    '''
    读取起点和终点
    '''
    return processed_data[:, 0, 2:], processed_data[:, -1, 2:]

def transform_coordinates(points):
    processed_points = (10000 * points + 600 + 75).astype(int)
    # 将二维numpy数组转换为List[Tuple[int, ...]]
    return [tuple(row) for row in processed_points.tolist()]

def inverse_transform(sources, targets, solution):
    transformed_solution = (solution - 675).astype(float) / 10000
    final_solution = np.concatenate((sources[:, np.newaxis, :], transformed_solution, targets[:, np.newaxis, :]), axis=1)
    return final_solution

def save_solution(solution, save_path, n_agents, delta_time):
    file_instance = open(save_path, "w", encoding="UTF8", newline='')
    csv_writer = csv.writer(file_instance)

    for i in range(n_agents):
        header = ['Agent ID', i]
        row_1 = ['Number of', len(solution[i])]

        csv_writer.writerow(header)
        csv_writer.writerow(row_1)

        rows = []
        path_time = 0.0
        for j in range(len(solution[i])):
            rows = [j, path_time, solution[i][j][0], solution[i][j][1], solution[i][j][2]]
            path_time += delta_time
            csv_writer.writerow(rows)

    file_instance.close()