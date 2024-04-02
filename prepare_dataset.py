import os
import shutil
import glob


def prepare_videos(scr_dir, dst_dir, target_width=320):
    # 定义视频文件扩展名
    video_suffix = '.mp4'
    # video_suffix = ['.mp4', '.avi', '.flv', '.mkv', '.rmvb', '.wmv']

    # 创建目标目录
    os.makedirs(dst_dir, exist_ok=True)

    video_files = glob.glob(os.path.join(src_dir, video_suffix))

    for video_file in video_files:
        # 获取视频文件名
        video_name = os.path.basename(video_file)

        # 创建目标视频文件路径
        video_dir = os.path.join(dst_dir, video_name.split('.')[0])
        os.makedirs(video_dir, exist_ok=True)

        # 复制视频文件到目标目录
        shutil.copy(video_file, video_dir)

        # 复制标注文件
        shutil.copy2(os.path.join(src_dir, video_name.replace(video_suffix, '.txt')), video_dir)

    print(f"Dataset prepared at {dst_dir}")


if __name__ == '__main__':
    prepare_videos('input_videos', 'prepared_videos')
