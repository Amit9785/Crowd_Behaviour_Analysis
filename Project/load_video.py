import cv2
import os

def load_video(video_path):
    """Loads a video file and returns a list of frames."""
    if not os.path.exists(video_path):
        print(f"Error: File '{video_path}' not found!")
        return None

    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop when the video ends
        frames.append(frame)

    cap.release()
    print(f"Loaded {len(frames)} frames from {video_path}")
    return frames

# Example usage
if __name__ == "__main__":
    video_path = r"C:\project\Minor\Project\data\1\1.mp4"  # Use absolute path

    frames = load_video(video_path)

    if frames and len(frames) > 0:
        cv2.imshow("First Frame", frames[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: No frames loaded.")