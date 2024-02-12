#import torch
import torch
import torch.nn as nn
import random
import numpy as np
import math
import pdb
import aestream
import time
import cv2
import pdb
import numpy as np
import math
import argparse
import csv
import os

# pdb.set_trace()


parser = argparse.ArgumentParser(description='Visualizer')
parser.add_argument('-p1', '--port1', type= int, help="Port for events", default=4001)
parser.add_argument('-p2', '--port2', type= int, help="Port for events", default=4002)
parser.add_argument('-a', '--accumulation', type= int, help="Accumulation", default=1)
parser.add_argument('-s', '--scale', type= float, help="Image scale", default=1)
parser.add_argument('-r', '--record', action= 'store_true', help="records fenrir's output")
e_port_1 = parser.parse_args().port1
e_port_2 = parser.parse_args().port2
accumulation = parser.parse_args().accumulation
scale = parser.parse_args().scale
record = parser.parse_args().record

win_name = "Fenrir"
cv2.namedWindow(win_name)        # Create a named window
cv2.moveWindow(win_name, 640, 200)  # Move it to (40,30)

# Stream events from UDP port 3333 (default)
frame = np.zeros((640*2,480*1,3))

fail1 = False
fail2 = False

it = time.time()
st = time.time()

fps = 60
images = []
slowest = 1/fps

with aestream.UDPInput((640, 480), device = 'cpu', port=e_port_1) as stream1:
    with aestream.UDPInput((640, 480), device = 'cpu', port=e_port_2) as stream2:
        count = 0
        try:    
            while True:

                st = time.time()
                frame[0:640,0:480,1] +=  stream1.read()# Provides a (640, 480)
                #frame[0:640,0:480,0] = frame[0:640,0:480,1]
                frame[640:640*2,0:480,1] += stream2.read() # Provides a (640, 480) 
                #frame[640:640*2,0:480,0] = frame[640:640*2,0:480,2]
                count += 1

                # pdb.set_trace()

                # image = cv2.resize(np.transpose(frame), (math.ceil(640*2*scale),math.ceil(480*1*scale)), interpolation = cv2.INTER_AREA)

                if count >= accumulation:
                    count = 0
                    frame[0:640*2,0:4,:] =  np.ones((640*2,4,3))
                    frame[0:640*2,-4:,:] = np.ones((640*2,4,3))
                    frame[0:4,0:480,:] =  np.ones((4,480,3))
                    frame[-4:,0:480,:] =  np.ones((4,480,3))
                    frame[638:642,0:480,:] =  np.ones((4,480,3))
                    image = cv2.resize(frame.transpose(1,0,2), (math.ceil(640*2*scale),math.ceil(480*1*scale)), interpolation = cv2.INTER_AREA)
                    fontScale = 2
                    thickness = 3
                    # image = cv2.putText(image, 'L', (1220,940), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255,255,255), thickness, cv2.LINE_AA)
                    # image = cv2.putText(image, 'R', (1300,940), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255,255,255), thickness, cv2.LINE_AA)
                    cv2.imshow(win_name, image)
                    
                    images.append(cv2.convertScaleAbs(image*256))

                    slowest = max(slowest, time.time()-st)

                    cv2.waitKey(1)
                    frame = np.zeros((640*2,480*1,3))
                
        except:
            if record:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = int(1/slowest)
                out = cv2.VideoWriter('./fenrir_view.mp4', fourcc, fps, (math.ceil(640*2*scale),math.ceil(480*1*scale)))
                for image in images:
                    out.write(image)
                out.release()
                print("\n\nVideo saved\n")