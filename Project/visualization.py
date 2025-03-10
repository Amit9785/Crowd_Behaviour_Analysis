import numpy as np
import cv2
import time

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx**2 + fy**2)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Load video
video_path = r"C:\project\Minor\Project\data\1\1.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

suc, prev = cap.read()
if not suc:
    print("Error: Could not read first frame.")
    cap.release()
    exit()

# Reduce the resolution of the frames
scale_percent = 150  # Adjust this value to change the resolution
width = int(prev.shape[1] * scale_percent / 100)
height = int(prev.shape[0] * scale_percent / 100)
dim = (width, height)

prev = cv2.resize(prev, dim, interpolation=cv2.INTER_AREA)
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
prevgray = cv2.GaussianBlur(prevgray, (5, 5), 0)  # Apply Gaussian blur
start = time.time()

while True:
    suc, img = cap.read()
    if not suc:
        break

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur
    
    # Adjust the parameters of the optical flow algorithm
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray.copy()

    end = time.time()
    fps = 1 / (end - start)
    start = end
    print(f"FPS: {fps:.2f}")

    cv2.imshow('Optical Flow', draw_flow(gray, flow))
    cv2.imshow('Flow HSV', draw_hsv(flow))

    if cv2.waitKey(30) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()