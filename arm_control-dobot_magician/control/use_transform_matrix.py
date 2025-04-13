import numpy as np
import os

def load_transform_matrix(file_path):
    """
    从 .npy 文件中加载转换矩阵
    
    Args:
        file_path: .npy 文件的路径
        
    Returns:
        4x4 转换矩阵
    """
    try:
        matrix = np.load(file_path)
        print("成功加载转换矩阵:")
        print(matrix)
        return matrix
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return None
    except Exception as e:
        print(f"错误: 加载矩阵时出错 - {str(e)}")
        return None

def transform_position(matrix, point1, point2):
    """
    使用转换矩阵转换一个点
    
    Args:
        matrix: 4x4 转换矩阵
        point: 要转换的点 [x, y, z, 1]
        
    Returns:
        转换后的点 [x', y', z', 1]
    """
    point =( np.array(point1) + np.array(point2))/2
    # 确保点是齐次坐标 (4x1)
    if len(point) == 3:
        point = np.append(point, 1)
    
    # 转换为列向量
    point = np.array(point).reshape(4, 1)
    matrix = np.array(matrix)
    # 应用转换
    transformed_point = matrix @ point
    
    # 返回转换后的点
    return transformed_point.flatten()[:3]

def transform_angle(matrix, point1, point2):
    object_vector = np.array(point2) - np.array(point1)
    # 确保 matrix 是 NumPy 数组
    matrix = np.array(matrix)
    # 获取旋转矩阵（3x3）
    rotation_matrix = matrix[:3, :3]
    
    # 应用旋转
    transformed_vector = rotation_matrix @ object_vector[:3]
    
    # 计算角度（使用反正切函数）
    angle = np.arctan2(transformed_vector[1], transformed_vector[0])
    
    return angle * (180 / np.pi)

def transform_input(matrix, point1, point2):
    transformed_point = transform_position(matrix, point1, point2)
    transformed_angle = transform_angle(matrix, point1, point2)
    return transformed_point[0], transformed_point[1], transformed_point[2], transformed_angle


def main():
    matrix_file = "base_camera_result.npy"
    
    # 加载矩阵
    matrix = load_transform_matrix(matrix_file)
    
    if matrix is not None:
        # 示例: 转换一个点 [x, y, z, 1]
        # 这里使用示例坐标，您可以根据需要修改
        point = [100, 200, 300, 1]  # 示例点
        
        # 转换点
        transformed_point = transform_point(matrix, point)
        
        print("\n原始点:", point)
        print("转换后的点:", transformed_point)
        
        # 如果您有多个点需要转换
        points = [
            [100, 200, 300, 1],
            [150, 250, 350, 1],
            [200, 300, 400, 1]
        ]
        
        print("\n批量转换点:")
        for i, pt in enumerate(points):
            transformed = transform_point(matrix, pt)
            print(f"点 {i+1}: {pt} -> {transformed}")

if __name__ == "__main__":
    main() 