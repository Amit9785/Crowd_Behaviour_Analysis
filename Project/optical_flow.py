import numpy as np
import cv2

def draw_flow(img, flow, step=16):
    """Draws the optical flow vectors on the image."""
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
    """Visualizes the optical flow field using HSV color encoding."""
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx**2 + fy**2)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def compute_curl_divergence(flow):
    """Computes curl and divergence of the optical flow field."""
    fx, fy = flow[:, :, 0], flow[:, :, 1]

    # Compute derivatives
    dfx_dx = cv2.Sobel(fx, cv2.CV_64F, 1, 0, ksize=5)
    dfx_dy = cv2.Sobel(fx, cv2.CV_64F, 0, 1, ksize=5)
    dfy_dx = cv2.Sobel(fy, cv2.CV_64F, 1, 0, ksize=5)
    dfy_dy = cv2.Sobel(fy, cv2.CV_64F, 0, 1, ksize=5)

    # Compute divergence (dfx/dx + dfy/dy)
    divergence = dfx_dx + dfy_dy

    # Compute curl (dfy/dx - dfx/dy)
    curl = dfy_dx - dfx_dy

    return divergence, curl

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read video.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Parameters for tracking
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Initialize feature tracking
    prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    mask = np.zeros_like(prev_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Compute curl and divergence
        divergence, curl = compute_curl_divergence(flow)

        # Normalize and visualize
        divergence_vis = cv2.normalize(divergence, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        curl_vis = cv2.normalize(curl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Compute trajectory tracking using Lucas-Kanade method
        if prev_points is not None:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)
            
            # Select good points
            good_new = next_points[status == 1]
            good_old = prev_points[status == 1]
            
            # Draw the trajectories
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

            frame = cv2.add(frame, mask)
            prev_points = good_new.reshape(-1, 1, 2)

        prev_gray = gray

        # Display results
        cv2.imshow('Optical Flow', draw_flow(gray, flow))
        cv2.imshow('HSV Flow', draw_hsv(flow))
        cv2.imshow('Divergence', divergence_vis)
        cv2.imshow('Curl', curl_vis)
        cv2.imshow('Trajectories', frame)

        key = cv2.waitKey(30)
        if key == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Run with a sample video file
video_path = r"C:\project\Minor\Project\data\1\1.mp4"  # Change this to the path of your uploaded video
process_video(video_path)
