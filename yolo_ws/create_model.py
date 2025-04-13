import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Add this FIRST

from ultralytics import YOLO

# Load current model as pretrained model
model = YOLO(os.path.join(os.path.dirname(__file__), "model", "mennys_model2.pt"))

# Train with small-dataset optimizations
results = model.train(
    data=os.path.join(os.path.dirname(__file__), "dataset", "dataset9", "data.yaml"),
    epochs=40,  # 小数据可以增加epochs
    batch=8,     # 根据GPU内存调整
    imgsz=640,
    patience=20,  # 早停耐心值
    device='cpu',    # 使用CPU而不是GPU
    workers=2,   # 小数据减少workers
    optimizer='AdamW',  # 小数据推荐AdamW
    lr0=0.001,   # 初始学习率
    weight_decay=0.05,
    hsv_h=0.015,  # 颜色增强
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=45.0,  # 旋转增强
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,    # 马赛克增强保持开启
    mixup=0.1,     # 小量MixUp
    copy_paste=0.1, # 小量复制粘贴增强
    erasing=0.2,   # 随机擦除
    cache=True     # 启用缓存加速小数据训练
)

# Save the new model with a different name to avoid overwriting
model.save(os.path.join(os.path.dirname(__file__), "model", "mennys_model3.pt"))



