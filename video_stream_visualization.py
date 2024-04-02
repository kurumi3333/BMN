import cv2
import numpy as np


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    frame_idx = 0
    while True:
        # Read a new frame
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no more frames
        # Convert the frame to grayscale
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = frame

        # Display the frame number on the current frame
        cv2.putText(gray_frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, )

        # resize the window
        window_name = 'Processed Video'
        cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window_name, 768, 432)
        gray_frame = process_frame(gray_frame)
        # Show the processed frame
        cv2.imshow(window_name, gray_frame)

        # Wait for 25ms and check if the user wants to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        frame_idx += 1

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def process_frame(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 30, 150)

    # Apply Hough transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    # Draw lines on the frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame


if __name__ == '__main__':
    video_path = 'input/test.mp4'
    process_video(video_path)
