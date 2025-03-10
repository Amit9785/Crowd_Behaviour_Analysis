# Code Citations

## License: unknown
https://github.com/eduardoZapata/Computer-Vision-Projects-CS378/tree/23d5e0ad85b2d5bc88f9a722a819c265415fe9ce/project_4/code/recognize_emotion/op_flow.py

```
np

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1
```


## License: GPL_3_0
https://github.com/burakbayramli/classnotes/tree/75d72566f32cf0e624a54bf84711f3e6930bc7c6/pde/pde_lk/pde_lk.tex

```
flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
```


## License: Apache_2_0
https://github.com/JuicyMango123/DeepST/tree/90226205fef7dd11959ef9d4e7d7732422a2a9f5/cv/samples/python/dis_opt_flow.py

```
w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x
```


## License: BSD_3_Clause
https://github.com/bassem-mostafa/Self-Localization/tree/0f3005d25dc1041e124a4ecf93a7973d1acb4fb7/script%28s%29/OpticalFlowDense.py

```
fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0
```


## License: unknown
https://github.com/PsorTheDoctor/artificial-intelligence/tree/97da50d4a25a2733a338b5c2cf5c979e1dae0eb8/computer_vision/optical_flow/dense_optical_flow.py

```
, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))
    for (x1, y1), (
```

