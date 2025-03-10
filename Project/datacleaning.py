import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr

# Load the video
video_path = r"C:\project\Minor\Project\data\1\1.mp4"
cap = cv2.VideoCapture(video_path)

# Read first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error: Could not read first frame.")
    cap.release()
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
h, w = prev_gray.shape

# Kernels for numerical differentiation (for curl and divergence)
kx = np.array([[-1, 1], [-1, 1]])  # Partial derivative w.r.t x
ky = np.array([[-1, -1], [1, 1]])  # Partial derivative w.r.t y

# Function to remove outliers using IQR method
def remove_outliers(flow):
    """ Remove outliers using the Interquartile Range (IQR) method. """
    flow_x, flow_y = flow[..., 0].flatten(), flow[..., 1].flatten()
    
    # Compute IQR thresholds
    q1_x, q3_x = np.percentile(flow_x, [25, 75])
    q1_y, q3_y = np.percentile(flow_y, [25, 75])
    iqr_x, iqr_y = iqr(flow_x), iqr(flow_y)
    
    lower_x, upper_x = q1_x - 1.5 * iqr_x, q3_x + 1.5 * iqr_x
    lower_y, upper_y = q1_y - 1.5 * iqr_y, q3_y + 1.5 * iqr_y

    # Mask for valid flow vectors
    mask_x = (flow_x >= lower_x) & (flow_x <= upper_x)
    mask_y = (flow_y >= lower_y) & (flow_y <= upper_y)
    mask = mask_x & mask_y

    # Set outliers to zero
    flow[..., 0].flat[~mask] = 0
    flow[..., 1].flat[~mask] = 0
    return flow

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Compute dense optical flow (Farneback)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # ** 1. Thresholding: Remove small motion vectors **
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    flow[mag < 1.0] = 0  # Remove small motion
    
    # ** 2. Outlier Removal using IQR **
    flow = remove_outliers(flow)

    # ** 3. Smoothing (Gaussian Filtering) **
    flow[..., 0] = cv2.GaussianBlur(flow[..., 0], (5, 5), 0)
    flow[..., 1] = cv2.GaussianBlur(flow[..., 1], (5, 5), 0)

    # ** 4. Interpolation (Bilateral Filtering) **
    flow[..., 0] = cv2.bilateralFilter(flow[..., 0], 9, 75, 75)
    flow[..., 1] = cv2.bilateralFilter(flow[..., 1], 9, 75, 75)

    # Compute Curl (Vorticity)
    dVy_dx = cv2.filter2D(flow[..., 1], -1, kx)  # dVy/dx
    dVx_dy = cv2.filter2D(flow[..., 0], -1, ky)  # dVx/dy
    curl = dVy_dx - dVx_dy  # Curl = dVy/dx - dVx/dy

    # Compute Divergence
    dVx_dx = cv2.filter2D(flow[..., 0], -1, kx)  # dVx/dx
    dVy_dy = cv2.filter2D(flow[..., 1], -1, ky)  # dVy/dy
    divergence = dVx_dx + dVy_dy  # Divergence = dVx/dx + dVy/dy

    # Normalize for display
    curl_norm = (curl - np.min(curl)) / (np.max(curl) - np.min(curl) + 1e-6)
    divergence_norm = (divergence - np.min(divergence)) / (np.max(divergence) - np.min(divergence) + 1e-6)

    # Display results
    cv2.imshow("Optical Flow", flow[..., 0])
    cv2.imshow("Curl", curl_norm)
    cv2.imshow("Divergence", divergence_norm)  # Divergence

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_gray = gray.copy()

cap.release()
cv2.destroyAllWindows()