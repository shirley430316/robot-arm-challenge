import os
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from ultralytics import YOLO
import cv2
import numpy as np
import glob

# Load the model
model = YOLO("C:/Users/Administrator/Desktop/HKU/roboarm/yolo_ws/model/mennys_model2.pt")  # Must be an OBB model

# 指定输入图像文件夹路径
input_image_dir = "C:/Users/Administrator/Desktop/HKU/roboarm/yolo_ws/dataset/dataset8/test/images"

# 指定输出图像保存路径
output_dir = "C:/Users/Administrator/Desktop/HKU/roboarm/yolo_ws/output"
os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

# 获取所有图片文件
image_files = glob.glob(os.path.join(input_image_dir, "*.jpg"))
print(f"\n找到 {len(image_files)} 张测试图片")

# 处理每张图片
for image_file in image_files:
    print(f"\n处理图片: {os.path.basename(image_file)}")
    
    # 设置输出文件名
    output_image_path = os.path.join(output_dir, f"detection_{os.path.basename(image_file)}")
    
    results = model.predict(
        image_file,
        conf=0.35,    # 进一步降低置信度阈值，以检测更多small和big类别
        iou=0.2,      # 降低IOU阈值，减少目标合并
        max_det=30,   # 增加最大检测数量
        save=False,   # 不使用默认保存
        show=False,   # 不显示结果，我们将自定义显示
        verbose=False # 减少输出信息
    )

    # 处理结果
    for result in results:
        # 获取原始图像
        img = cv2.imread(image_file)
        
        # 自定义绘制检测框和标签
        for i in range(len(result.obb.cls)):
            # 获取检测信息
            class_name = result.names[int(result.obb.cls[i])]
            conf = result.obb.conf[i]
            box = result.obb.xyxyxyxy[i].cpu().numpy()  # 转换为numpy数组
            
            # 绘制旋转框
            points = box.reshape((-1, 2))
            points = points.astype(np.int32)
            cv2.polylines(img, [points], True, (0, 255, 0), 1)  # 减小线条粗细
            
            # 计算标签位置（使用旋转框的左上角）
            label_x = int(min(points[:, 0]))
            label_y = int(min(points[:, 1])) - 2  # 减小与框的距离
            
            # 创建标签文本（包含置信度）
            label = f"{class_name} ({conf:.2f})"
            
            # 设置字体
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.3  # 进一步减小字体大小
            thickness = 1     # 保持最小粗细
            
            # 获取文本大小
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # 绘制标签背景
            cv2.rectangle(img, 
                         (label_x, label_y - text_height - 2),  # 减小背景框的padding
                         (label_x + text_width, label_y),
                         (255, 255, 255),
                         -1)
            
            # 绘制文本
            cv2.putText(img, label,
                        (label_x, label_y - 2),  # 调整文本位置
                        font,
                        font_scale,
                        (0, 0, 0),
                        thickness)
        
        # 显示结果
        cv2.imshow('Detection Result', img)
        cv2.waitKey(1)  # 短暂显示图像
        
        # 保存图像到指定路径
        cv2.imwrite(output_image_path, img)
        print(f"检测结果已保存到: {output_image_path}")
        
        # 打印旋转框信息
        if hasattr(result, 'obb') and result.obb is not None:
            print("\n旋转框检测结果:")
            # 按类别统计检测数量
            class_counts = {}
            # 遍历所有检测结果
            for i in range(len(result.obb.cls)):
                class_name = result.names[int(result.obb.cls[i])]
                conf = result.obb.conf[i]
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
            
            # 打印每个类别的检测数量
            print("\n检测统计:")
            for class_name, count in class_counts.items():
                print(f"{class_name}: {count}个")
        else:
            print("未检测到旋转框")

print("\n所有图片处理完成。按任意键退出...")
cv2.waitKey(0)
cv2.destroyAllWindows()