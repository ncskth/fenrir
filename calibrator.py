
import numpy as np
import cv2
import pdb
import numpy as np
import math
import time

off_frame = np.zeros((1200, 2000), dtype=np.uint8)
on_frame = off_frame.copy()
for px in range(5):
    for py in range(5):
        cx = 250 + 200*px
        cy = 550 + 200*py
        for i in range(cx - 10, cx + 11):
            for j in range(cy - 10, cy + 11):
                on_frame[i, j] = 255


win_name = "Calibration"
cv2.namedWindow(win_name)        # Create a named window
cv2.moveWindow(win_name, 640, 200)  # Move it to (40,30)

last_time = time.time()

frame_parity = False
while True:
    current_time = time.time()
    elapsed = current_time - last_time
    if elapsed < 1.0/120.0:
        time.sleep(1.0/120.0 - elapsed)
    if frame_parity:
        cv2.imshow(win_name, on_frame)
        frame_parity = False
    else:
        cv2.imshow(win_name, off_frame)
        frame_parity = True
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    last_time = time.time()
    #cv2.imshow(win_name, on_frame)
    #cv2.waitKey(0)