#!/usr/bin/env python3
import zmq
import numpy as np
import cv2
import struct
import argparse
from dv_processing import camera

from typing import *


# Map OpenCV type codes → (numpy dtype, channels)
CV_DEPTH = {
    0: np.uint8,
    1: np.int8,
    2: np.uint16,
    3: np.int16,
    4: np.int32,
    5: np.float32,
    6: np.float64,
}


def decode_cv_type(type_code):
    depth = type_code & 7
    channels = (type_code >> 3) + 1
    return CV_DEPTH[depth], channels


def frame_to_mat(meta_bytes, data_bytes):
    rows, cols, type_code, nbytes = struct.unpack('<iiiq', meta_bytes)
    dtype, channels = decode_cv_type(type_code)
    arr = np.frombuffer(data_bytes, dtype=dtype)
    if channels == 1:
        return arr.reshape((rows, cols))
    return arr.reshape((rows, cols, channels))


def hsv_to_rgb(h, s, v):
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = (i % 6).astype(np.int32)
    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def draw_depth_frame(depth_measurements: List[Tuple[int, int, int]], search_bound: int) -> np.ndarray:
    result = np.zeros((480, 640, 3), dtype=np.uint8)
    for x, y, d in depth_measurements:
        result[y, x] = hsv_to_rgb(120*d/(180*search_bound), 1, 1)
    return result


def build_camera_matrix(calib):
    fx, fy = calib.focalLength
    cx, cy = calib.principalPoint
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)


def draw_depth_color_key(calib_json: str, sbmSearchBound: int) -> cv2.Mat:
    calibration = camera.CalibrationSet.LoadFromFile(calib_json)

    left_calib = calibration.getCameraCalibration("C0")
    right_calib = calibration.getCameraCalibration("C1")
    resolution = left_calib.resolution

    camMatL = build_camera_matrix(left_calib)
    camMatR = build_camera_matrix(right_calib)

    distCoeffsL_mat = np.array(left_calib.distortion, dtype=np.float64)
    distCoeffsR_mat = np.array(right_calib.distortion, dtype=np.float64)

    R = np.array(right_calib.transformationToC0.getRotationMatrix(), dtype=np.float64)
    T = np.array(right_calib.transformationToC0.getTranslation(), dtype=np.float64)

    print(R.shape, T.shape, camMatL.shape, distCoeffsL_mat.shape)
    _, _, _, _, Q, _, _ = cv2.stereoRectify(
        camMatL, distCoeffsL_mat, camMatR, distCoeffsR_mat,
        resolution, R, T[:, np.newaxis]
    )

    f = Q[2, 3]
    Tx = -1.0 / Q[3, 2]

    height, width = 480, 200
    key = np.zeros((height, width, 3), np.uint8)

    for y in range(height - 1, -1, -1):
        disp = 1.0 - (height - y - 1) / height
        hue = int(120.0 * disp)
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        key[y, :width//2] = bgr

    numTicks = 5
    for i in range(numTicks - 1, -1, -1):
        disp = 10.0 + i / (numTicks - 1) * (sbmSearchBound - 20)
        depth = f * Tx / disp
        y = int((disp / sbmSearchBound) * height)
        y = max(0, min(height - 1, y))
        cv2.line(key, (width - 10, y), (width, y), (255, 255, 255), 2)
        label = "%.2f m" % depth
        cv2.putText(key, label, (100, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.putText(key, "Near", (20, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(key, "Far", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return key


def recv_message(sock):
    """Receive one logical multipart message."""
    TAG_LEFT = b'\x00'
    TAG_RIGHT = b'\x01'
    TAG_TRIPLES = b'\x02'

    tag = sock.recv()
    #print(tag_bytes)
    #tag = tag_bytes[0]

    if tag == TAG_LEFT:
        meta = sock.recv()
        data = sock.recv()
        return 'left', frame_to_mat(meta, data)

    if tag == TAG_RIGHT:
        meta = sock.recv()
        data = sock.recv()
        return 'right', frame_to_mat(meta, data)

    elif tag == TAG_TRIPLES:
        count_bytes = sock.recv()
        (count,) = struct.unpack('<Q', count_bytes)
        triples = []
        for _ in range(count):
            a = struct.unpack('<i', sock.recv())[0]
            b = struct.unpack('<i', sock.recv())[0]
            c = struct.unpack('<i', sock.recv())[0]
            triples.append((a, b, c))
        return 'triples', triples

    raise RuntimeError(f"unknown tag {tag}")


def receive_and_update_full_frame(sock, image: np.ndarray, search_bound: int) -> np.ndarray:
    for _ in range(3):
        kind, val = recv_message(sock)
        if kind == 'left':
            three_channels = np.stack([val, val, val], axis=-1)
            image[480:, :640] = three_channels
        elif kind == 'right':
            three_channels = np.stack([val, val, val], axis=-1)
            image[480:, 640:] = three_channels
        else:
            depth_render = draw_depth_frame(val, search_bound)
            image[:480, :640] = depth_render
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib_json", type=str, required=True, help="Path to the JSON file holding calibration information about the robot's cameras.")
    parser.add_argument("--search_bound", type=int, default=100,
                        help="Maximum pixel disparity run for the SBM algorithm on the robot (must match with the flag value on the robot's side!)")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_name", type=str, default="recording.mp4")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.bind("tcp://*:5555")

    cv2.namedWindow("Demo", cv2.WINDOW_NORMAL)

    depth_color_key = draw_depth_color_key(args.calib_json, args.search_bound)
    image = np.zeros((960, 1280, 3), dtype=np.uint8)
    image[:480, 640:840] = depth_color_key

    if args.record_video:
        video_frames = []

    while True:
        try:
            """
            payload = []
            for _ in range(3):
                meta = sock.recv()
                data = sock.recv()
                payload.append(frame_to_mat(meta, data))

            left, right, depth = payload
            img1 = np.concatenate((depth, np.zeros((480, 440, 3), dtype=np.uint8)), axis=1)
            img2 = np.concatenate((left, right), axis=1)
            img2 = np.stack([img2, img2, img2], axis=-1)
            image = np.concatenate((img1, img2), axis=0)
            """
            image = receive_and_update_full_frame(sock, image, args.search_bound)

            if args.record_video:
                video_frames.append(image)

            cv2.imshow("Demo",  image)
            cv2.waitKey(2)

        except KeyboardInterrupt:
            if args.record_video:
                print("Saving video . . .")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(args.video_name, fourcc, args.fps, (1280, 960))
                if not writer.isOpened():
                    # fall back to mp4v
                    writer = cv2.VideoWriter(
                        args.video_name, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1280, 960)
                    )
                if not writer.isOpened():
                    raise RuntimeError("no usable VideoWriter backend")

                for f in video_frames:
                    writer.write(f)
                writer.release()

            cv2.destroyAllWindows()
            sock.close()
            ctx.term()

            break

if __name__ == "__main__":
    main()