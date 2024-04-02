import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import InputPadder
from RAFT.core.utils import flow_viz
from config import optical_flow_raft_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(model_path, args):
    model = RAFT(args)
    model.load_state_dict(torch.load(model_path))
    model = model.to(DEVICE)
    model.eval()
    return model


def run_optical_flow(model, video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    if not ret:
        print("Error: Could not read first video frame")
        cap.release()
        return

    prev_frame = torch.from_numpy(frame1).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    padder = InputPadder(prev_frame.shape)
    prev_frame = padder.pad(prev_frame)[0]

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        curr_frame = torch.from_numpy(frame2).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        curr_frame = padder.pad(curr_frame)[0]

        with torch.no_grad():
            flow_low, flow_up = model(prev_frame, curr_frame, iters=20, test_mode=True)

        # Visualize or process the flow here
        flow_up_np = flow_up[0].permute(1, 2, 0).cpu().numpy()
        flow_img = flow_viz.flow_to_image(flow_up_np)
        cv2.imshow('Optical Flow', flow_img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        prev_frame = curr_frame

    cap.release()
    cv2.destroyAllWindows()


def extract_and_save_optical_flow(video_path, output_dir):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first video frame")
        cap.release()
        return

    # 将第一帧转换为灰度
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    pbar = tqdm(total=total_frames - 1, desc="Extracting Optical Flow", unit="frame")
    idx = 0

    # 遍历视频中的每一帧
    while True:
        # 读取下一帧
        ret, next_frame = cap.read()
        if not ret:
            break

        # 将下一帧转换为灰度
        next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, next_frame_gray, None, 0.5, 3,
                                            15, 3, 5, 1.2, 0)
        # 将光流转换为可视化的颜色映射
        flow_vis = cv2.calcOpticalFlowFarneback(prev_frame_gray, next_frame_gray, None, 0.5, 3,
                                                15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mask = np.zeros_like(prev_frame)
        mask[..., 1] = 255
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

        # 保存光流图像
        mask_path = os.path.join(output_dir, f"flow_{idx:05d}.png")
        cv2.imwrite(mask_path, rgb)

        # 更新前一个帧和前一个灰度帧
        prev_frame_gray = next_frame_gray
        idx += 1
        pbar.update(1)
    cap.release()
    pbar.close()


if __name__ == '__main__':
    # 使用示例：
    video_path = 'input/test.mp4'
    output_dir = 'output/directory'
    # extract_and_save_optical_flow(video_path, output_dir)

    args = optical_flow_raft_model.get('args')
    model_path = 'model/raft-things.pth'
    model = load_model(model_path, args)
    run_optical_flow(model, video_path)
