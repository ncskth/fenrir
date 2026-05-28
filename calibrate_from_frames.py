import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import gaussian
from skimage.feature import peak_local_max

from typing import *

frame_files = ["calibration_frames/frame_1.npy",
               # "calibration_frames/new_frame_3.npy",
               "calibration_frames/newer_frame_1.npy",
               "calibration_frames/newest_frame_1.npy",
               "calibration_frames/most_newly_frame_1.npy"]


def adaptive_peak_finder(img: np.ndarray, n_expected_peaks: int):
    peaks = peak_local_max(img, threshold_rel=0.3)

    if len(peaks) == n_expected_peaks:
        return peaks
    elif len(peaks) > n_expected_peaks:
        small = 0.3
        curr = 0.35
        big = curr
        peaks = peak_local_max(img, threshold_rel=curr)
    else:
        big = 0.3
        curr = 0.25
        small = curr
        peaks = peak_local_max(img, threshold_rel=curr)

    count = 0
    while len(peaks) != n_expected_peaks:
        count += 1
        if count > 5:
            raise ValueError("Search did not converge!!")

        if len(peaks) > n_expected_peaks:
            small = curr
            curr = 0.5*(curr + big)
            peaks = peak_local_max(img, threshold_rel=curr)
        else:
            big = curr
            curr = 0.5*(curr + small)
            peaks = peak_local_max(img, threshold_rel=curr)

    return peaks


def detect_peaks_from_frames(frame_name: str, grid_size: int, show: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    frame = np.load(frame_name)

    # expect only green to be active
    frame_l = frame[:640, :, 1]
    frame_r = frame[640:, :, 1]

    # blur to avoid too many peaks detected
    frame_blurred_l = gaussian(frame_l, sigma=3)
    frame_blurred_r = gaussian(frame_r, sigma=3)

    # expect peaks to be greater than 30% of maximum value
    peaks_l = adaptive_peak_finder(frame_blurred_l, grid_size)
    peaks_r = adaptive_peak_finder(frame_blurred_r, grid_size)

    # should probably be 25 peaks each (unless calibrator display was changed)
    print(f"Left peaks: {len(peaks_l)}")
    print(f"Right peaks: {len(peaks_r)}")

    # need to sort them so that it is easy to locate each one in world space
    # this is a bit hacky but i dont see a better way to do it at the moment
    peaks_l_flat_coords = peaks_l[:, 0] + 0.1*peaks_l[:, 1]
    peaks_l_perm = np.argsort(peaks_l_flat_coords)
    peaks_l = peaks_l[peaks_l_perm]
    peaks_r_flat_coords = peaks_r[:, 0] + 0.1*peaks_r[:, 1]
    peaks_r_perm = np.argsort(peaks_r_flat_coords)
    peaks_r = peaks_r[peaks_r_perm]

    if show:
        fig = plt.figure()
        ax_l = fig.add_subplot(1, 2, 1)
        ax_r = fig.add_subplot(1, 2, 2)

        ax_l.imshow(frame_blurred_l.T)
        ax_r.imshow(frame_blurred_r.T)

        for i, (xi, yi) in enumerate(peaks_l):
            ax_l.annotate(str(i), (xi, yi),
                          xytext=(5, 5),  # Offset label slightly from point
                          textcoords='offset points',
                          fontsize=9,
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        for i, (xi, yi) in enumerate(peaks_r):
            ax_r.annotate(str(i), (xi, yi),
                          xytext=(5, 5),  # Offset label slightly from point
                          textcoords='offset points',
                          fontsize=9,
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        # annotations should appear directly to the above right of the peaks
        # and the numbers should have a good ordering to them
        plt.show()

    return peaks_l, peaks_r


def calibrate_from_pixel_peaks(peak_files_l: List[str],
                               peak_files_r: List[str],
                               grid_spacing_meters: float,
                               grid_shape: Tuple[int, int],
                               save_to_folder: str,
                               show: bool = True):

    peaks_l = [np.load(f).astype(np.float32) for f in peak_files_l]
    peaks_r = [np.load(f).astype(np.float32) for f in peak_files_r]

    world_points = grid_spacing_meters * \
        np.mgrid[:grid_shape[0], :grid_shape[1]
                 ].T.reshape(-1, 2)[:, ::-1].astype(np.float32)
    world_points = np.concat((world_points, np.zeros_like(
        world_points[:, 0][:, np.newaxis])), axis=-1)
    world_points_copied = [world_points.copy() for _ in peak_files_l]

    if show:
        fig = plt.figure()
        ax_l = fig.add_subplot(1, 2, 1)
        ax_r = fig.add_subplot(1, 2, 2)

        ax_l.scatter(peaks_l[0][:, 0], peaks_l[0][:, 1])
        ax_r.scatter(peaks_r[0][:, 0], peaks_r[0][:, 1])

        for i, (xi, yi) in enumerate(peaks_l[0]):
            ax_l.annotate(str(world_points[i, :2]), (xi, yi),
                          xytext=(5, 5),  # Offset label slightly from point
                          textcoords='offset points',
                          fontsize=9,
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        for i, (xi, yi) in enumerate(peaks_r[0]):
            ax_r.annotate(str(world_points[i, :2]), (xi, yi),
                          xytext=(5, 5),  # Offset label slightly from point
                          textcoords='offset points',
                          fontsize=9,
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        plt.show()

    _, camera_matrix_l, distortion_coeffs_l, _, _ = cv2.calibrateCamera(
        world_points_copied, peaks_l, (640, 480), None, None)
    _, camera_matrix_r, distortion_coeffs_r, _, _ = cv2.calibrateCamera(
        world_points_copied, peaks_r, (640, 480), None, None)

    _, camera_matrix_l, distortion_coeffs_l, \
        camera_matrix_r, distortion_coeffs_r, \
        rotation, translation, \
        essential, fundamental = cv2.stereoCalibrate(world_points_copied,
                                                     peaks_l,
                                                     peaks_r,
                                                     camera_matrix_l,
                                                     distortion_coeffs_l,
                                                     camera_matrix_r,
                                                     distortion_coeffs_r,
                                                     (640, 480))

    np.save(os.path.join(save_to_folder, "cam_mat_l.npy"), camera_matrix_l)
    np.save(os.path.join(save_to_folder, "cam_mat_r.npy"), camera_matrix_r)
    np.save(os.path.join(save_to_folder, "dist_coeffs_l.npy"),
            distortion_coeffs_l)
    np.save(os.path.join(save_to_folder, "dist_coeffs_r.npy"),
            distortion_coeffs_r)
    np.save(os.path.join(save_to_folder, "rotation.npy"), rotation)
    np.save(os.path.join(save_to_folder, "translation.npy"), translation)
    np.save(os.path.join(save_to_folder, "essential.npy"), essential)
    np.save(os.path.join(save_to_folder, "fundamental.npy"), fundamental)


def test_stereo_calibration(frame_path: str,
                            calib_data_folder: str):
    frame = np.load(frame_path)
    frame_l = gaussian(frame[:640, :, 1], sigma=3)
    #frame_l = (255*frame_l/frame_l.max()).astype(np.uint8)
    frame_r = gaussian(frame[640:, :, 1], sigma=3)
    #frame_r = (255*frame_r/frame_r.max()).astype(np.uint8)

    cam_mat_l = np.load(os.path.join(calib_data_folder, "cam_mat_l.npy"))
    cam_mat_r = np.load(os.path.join(calib_data_folder, "cam_mat_r.npy"))
    dist_coeffs_l = np.load(os.path.join(calib_data_folder, "dist_coeffs_l.npy"))
    dist_coeffs_r = np.load(os.path.join(calib_data_folder, "dist_coeffs_r.npy"))
    rotation = np.load(os.path.join(calib_data_folder, "rotation.npy"))
    translation = np.load(os.path.join(calib_data_folder, "translation.npy"))
    essential = np.load(os.path.join(calib_data_folder, "essential.npy"))
    fundamental = np.load(os.path.join(calib_data_folder, "fundamental.npy"))

    print(translation)
    print(cam_mat_l)
    print(cam_mat_r)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cam_mat_l, dist_coeffs_l,
        cam_mat_r, dist_coeffs_r,
        (640, 480), rotation, translation,
        alpha=0
    )
    print(P1)
    print(P2)
    #R1 = R1.dot(np.linalg.inv(cam_mat_l))
    #R2 = R2.dot(np.linalg.inv(cam_mat_r))

    map1_left, map2_left = cv2.initUndistortRectifyMap(
        cam_mat_l, dist_coeffs_l, R1, P1, (640, 480), cv2.CV_32FC1
    )
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        cam_mat_r, dist_coeffs_r, R2, P2, (640, 480), cv2.CV_32FC1
    )
    rectified_l = cv2.remap(frame_l, map1_left, map2_left, cv2.INTER_LINEAR)
    rectified_r = cv2.remap(frame_r, map1_right, map2_right, cv2.INTER_LINEAR)
    print(rectified_l.max())
    print(rectified_r.max())

    fig = plt.figure()
    ax_l = fig.add_subplot(2, 2, 1)
    ax_r = fig.add_subplot(2, 2, 2)
    ax_l_rect = fig.add_subplot(2, 2, 3)
    ax_r_rect = fig.add_subplot(2, 2, 4)

    ax_l.imshow(frame_l.T)
    ax_l.set_title("Left unrectified")
    ax_r.imshow(frame_r.T)
    ax_r.set_title("Right unrectified")
    ax_l_rect.imshow(rectified_l.T)
    ax_l_rect.set_title("Left rectified")
    ax_r_rect.imshow(rectified_r.T)
    ax_r_rect.set_title("Right rectified")

    plt.show()


if __name__ == "__main__":

    peak_files_l = []
    peak_files_r = []
    for ff in frame_files:
        peaks_l, peaks_r = detect_peaks_from_frames(ff, 15, show=False)
        head, tail = os.path.split(ff)
        pathl = os.path.join(head, "left_peaks_" + tail)
        pathr = os.path.join(head, "right_peaks_" + tail)
        np.save(pathl, peaks_l)
        np.save(pathr, peaks_r)
        peak_files_l.append(pathl)
        peak_files_r.append(pathr)
    calibrate_from_pixel_peaks(
        peak_files_l, peak_files_r, 0.038, (3, 5), "calibration_data", show=False)

    for ff in frame_files:
        test_stereo_calibration(ff, "calibration_data")
