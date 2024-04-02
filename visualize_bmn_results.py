import cv2
import mmcv
import numpy as np


def draw_predictions(frame, detections, frame_idx, fps, thickness=2, font_scale=0.5, color=(0, 255, 0)):
    for start, end, score in detections:
        start_frame = int(start * fps)
        end_frame = int(end * fps)

        if start_frame <= frame_idx <= end_frame:
            cv2.putText(frame, f'Score: {score:.2f}', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, thickness)


def visualize_detections(video_path, detection_results, output_path):
    # Load the detection results
    detections = mmcv.load(detection_results)

    # Read the video
    video = mmcv.VideoReader(video_path)
    fps = video.fps
    frames = [frame for frame in video]

    # Prepare output video writer
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Draw detections on each frame
    for idx, frame in enumerate(frames):
        draw_predictions(frame, detections, idx, fps)
        out_video.write(frame)
    out_video.release()


if __name__ == '__main__':
    video_file_path = 'path/to/your/video.mp4'
    detection_results_path = 'path/to/your/detection_results.pkl'
    output_video_path = 'path/to/save/visualized_video.mp4'

    visualize_detections(video_file_path, detection_results_path, output_video_path)
    print(f"Visualization completed. Output saved to {output_video_path}")
