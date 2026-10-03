#!/usr/bin/env python3
import zmq
import numpy as np
import cv2
import struct
import argparse

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_name", type=str, default="recording.mp4")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)          # or SUB if the C++ side is PUB
    sock.bind("tcp://*:5555")  # or bind() if this is the server

    cv2.namedWindow("Demo", cv2.WINDOW_NORMAL)

    if args.record_video:
        video_frames = []

    while True:
        try:
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