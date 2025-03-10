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

cap = cv2.VideoCapture(0)

suc, prev = cap.read()
if not suc:
    print("Failed to open camera.")
    cap.release()
    exit()

prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
start = time.time()

while True:
    suc, img = cap.read()
    if not suc:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray.copy()

    end = time.time()
    fps = 1 / (end - start)
    start = end
    print(f"FPS: {fps:.2f}")

    cv2.imshow('Optical Flow', draw_flow(gray, flow))
    cv2.imshow('Flow HSV', draw_hsv(flow))

    if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
