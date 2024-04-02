import os
import torch
import cv2
import numpy as np
from mmaction.apis import inference_recognizer, init_recognizer
import config


def extract_features(model_config, model_checkpoint, video_path, output_file):
    # 加载TSN模型和配置
    config_file = 'path/to/TSN/config.py'
    checkpoint_file = 'path/to/TSN/checkpoint.pth'
    model = init_recognizer(config_file, checkpoint_file, device='cuda:0')

    # 设置数据处理流程
    pipeline = config.pipeline

    # 读取视频文件
    video_path = 'path/to/video.mp4'
    cap = cv2.VideoCapture(video_path)

    # 初始化特征列表
    features_list = []

    # 循环读取视频帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video ended or error.")
            break

        # 将帧转换为模型可接受的格式
        data = dict(img=frame, label=-1)
        data = pipeline(data)

        # 提取特征
        with torch.no_grad():
            feature = model.extract_feat(data['img'][None, :])
        features_list.append(feature.cpu().numpy())

    # 释放资源
    cap.release()

    # 将特征列表转换为numpy数组
    features_array = np.concatenate(features_list, axis=0)

    # 保存特征到文件
    output_file = 'path/to/features.npy'
    np.save(output_file, features_array)
    print(f"Feautres saved to {output_file}")
