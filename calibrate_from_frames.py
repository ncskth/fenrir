import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import gaussian
from skimage.feature import peak_local_max

from typing import *

frame_files = ["calibration_frames/frame_1.npy",
               #"calibration_frames/new_frame_3.npy",
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
                               cam_mat_l: str,
                               cam_mat_r: str,
                               dist_coeffs_l: str,
                               dist_coeffs_r: str,
                               show: bool = True):

    peaks_l = [np.load(f).astype(np.float32) for f in peak_files_l]
    peaks_r = [np.load(f).astype(np.float32) for f in peak_files_r]

    world_points = grid_spacing_meters*np.mgrid[:grid_shape[0], :grid_shape[1]].T.reshape(-1, 2)[:, ::-1].astype(np.float32)
    world_points = np.concat((world_points, np.zeros_like(world_points[:, 0][:, np.newaxis])), axis=-1)
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

    _, camera_matrix_l, distortion_coeffs_l, _, _ = cv2.calibrateCamera(world_points_copied, peaks_l, (640, 480), None, None)
    _, camera_matrix_r, distortion_coeffs_r, _, _ = cv2.calibrateCamera(world_points_copied, peaks_r, (640, 480), None, None)

    np.save(cam_mat_l, camera_matrix_l)
    np.save(cam_mat_r, camera_matrix_r)
    np.save(dist_coeffs_l, distortion_coeffs_l)
    np.save(dist_coeffs_r, distortion_coeffs_r)


def test_mono_calibration(frame_path: str,
                          left_matrix_path: str,
                          right_matrix_path: str,
                          left_distortion_path: str,
                          right_distortion_path: str):
    frame = np.load(frame_path)
    frame_l = frame[:640, :, 1]
    frame_r = frame[640:, :, 1]

    mat_l = np.load(left_matrix_path)
    mat_r = np.load(right_matrix_path)
    coeffs_l = np.load(left_distortion_path)
    coeffs_r = np.load(right_distortion_path)

    undistort_l = cv2.undistort(frame_l, mat_l, coeffs_l, None, None)
    undistort_r = cv2.undistort(frame_r, mat_r, coeffs_r, None, None)

    fig = plt.figure()
    ax_l = fig.add_subplot(1, 2, 1)
    ax_r = fig.add_subplot(1, 2, 2)

    ax_l.imshow(undistort_l.T)
    ax_r.imshow(undistort_r.T)

    plt.show()


if __name__ == "__main__":
    """
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
    calibrate_from_pixel_peaks(peak_files_l, peak_files_r, 0.038, (3, 5),
                               "calibration_data/cam_mat_l.npy", "calibration_data/cam_mat_r.npy",
                               "calibration_data/dist_coeffs_l.npy", "calibration_data/dist_coeffs_r.npy")
    """
    for ff in frame_files:
        test_mono_calibration(ff,
                              "calibration_data/cam_mat_l.npy",
                              "calibration_data/cam_mat_r.npy",
                              "calibration_data/dist_coeffs_l.npy",
                              "calibration_data/dist_coeffs_r.npy")