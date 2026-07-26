import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import gaussian
from skimage.feature import peak_local_max
from scipy.ndimage import center_of_mass

from typing import *

frames = [30]
frame_files = [(f"mimir_jr_calibration_frames/frame_left_{i}.npy", f"mimir_jr_calibration_frames/frame_right_{i}.npy") for i in frames]
frame_files = [ff for ff in frame_files if os.path.exists(ff[0]) and os.path.exists(ff[1])]
print(len(frame_files))


def adaptive_peak_finder(img: np.ndarray, n_expected_peaks: int):
    peaks = peak_local_max(img, min_distance=3, threshold_rel=0.5)

    if len(peaks) == n_expected_peaks:
        return peaks
    elif len(peaks) > n_expected_peaks:
        small = 0.5
        curr = 0.99
        big = curr
        peaks = peak_local_max(img, min_distance=3, threshold_rel=curr)
    else:
        big = 0.5
        curr = 0.01
        small = curr
        peaks = peak_local_max(img, min_distance=3, threshold_rel=curr)

    count = 0
    while len(peaks) != n_expected_peaks:
        count += 1
        if count > 20:
            raise ValueError("Search did not converge!!")

        if len(peaks) > n_expected_peaks:
            small = curr
            curr = 0.5*(curr + big)
            peaks = peak_local_max(img, min_distance=3, threshold_rel=curr)
        else:
            big = curr
            curr = 0.5*(curr + small)
            peaks = peak_local_max(img, min_distance=3, threshold_rel=curr)

    return peaks


def locate_peaks(event_frame: np.ndarray, grid_size: int, show: bool = False):
    frame_clipped = np.where(event_frame > 20, event_frame, np.zeros_like(event_frame))
    frame_blurred = gaussian(frame_clipped, sigma=3)
    if show:
        plt.imshow(frame_blurred.T)
        plt.show()
    peaks = adaptive_peak_finder(frame_blurred, grid_size)

    recentered_peaks = []
    for peak in peaks:
        x = peak[0] - 3
        y = peak[1] - 3
        center = center_of_mass(frame_blurred[x : x + 7, y : y + 7])
        recentered_peaks.append(np.array([x + center[0], y + center[1]]))

    return np.array(recentered_peaks)


# i am losing my mind here
def get_skewing_matrix(peaks: np.ndarray):
    center = np.mean(peaks, axis=0)
    origin = peaks[np.argmin([np.linalg.norm(p - center) for p in peaks])]

    greater_x = [p for p in peaks if p[0] > origin[0]]
    greater_y = [p for p in peaks if p[1] > origin[1]]

    i1 = np.argmin([(x[0] - origin[0])**2 + 2*(x[1] - origin[1])**2 for x in greater_x])
    vec1 = greater_x[i1] - origin

    i2 = np.argmin([2*(x[0] - origin[0])**2 + (x[1] - origin[1])**2 for x in greater_y])
    vec2 = greater_y[i2] - origin

    matrix = np.stack([vec1, vec2], axis=1)
    if np.linalg.det(matrix) < 0:
        matrix= matrix[:, ::-1]

    return matrix


# this is really annoying
def sort_peaks(peaks: np.ndarray, show: bool = True):
    skewer = get_skewing_matrix(peaks)
    peaks_unskewed = peaks@np.linalg.inv(skewer).T
    peak_coords_flattened = peaks_unskewed[:, 0] - 0.1*peaks_unskewed[:, 1]
    #permutation = np.lexsort((peaks_unskewed[:, 1], peaks_unskewed[:, 0]))
    permutation = np.argsort(peak_coords_flattened)
    if show:
        peaks_unskewed = peaks_unskewed[permutation]
        plt.scatter(peaks_unskewed[:, 0], peaks_unskewed[:, 1])
        for i, (xi, yi) in enumerate(peaks_unskewed):
            plt.annotate(str(i), (xi, yi),
                          xytext=(5, 5),  # Offset label slightly from point
                          textcoords='offset points',
                          fontsize=9,
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        plt.show()
    return peaks[permutation]


def detect_peaks_from_frames(frame_names: Tuple[str, str], grid_size: int, show: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    frame_l = np.load(frame_names[0])
    frame_r = np.load(frame_names[1])

    peaks_l = locate_peaks(frame_l, grid_size, show=show)
    peaks_r = locate_peaks(frame_r, grid_size, show=show)

    # should probably be 25 peaks each (unless calibrator display was changed)
    #print(f"Left peaks: {len(peaks_l)}")
    #print(f"Right peaks: {len(peaks_r)}")
    peaks_l = sort_peaks(peaks_l, show=show)
    peaks_r = sort_peaks(peaks_r, show=show)

    if show:
        fig = plt.figure()
        ax_l = fig.add_subplot(1, 2, 1)
        ax_r = fig.add_subplot(1, 2, 2)

        ax_l.imshow(frame_l.T)
        ax_r.imshow(frame_r.T)

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


def verify_correspondences(world_points, image_peaks, title="Verification"):
    """Check if points appear in the expected grid pattern"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot world points (should form perfect grid)
    axes[0].scatter(world_points[:, 0], world_points[:, 1])
    axes[0].set_title("World Points (perfect grid)")
    axes[0].set_aspect('equal')

    # Plot image peaks (should form grid, possibly rotated/sheared)
    axes[1].scatter(image_peaks[:, 0], image_peaks[:, 1])
    axes[1].set_title("Image Peaks (should look like grid)")
    axes[1].invert_yaxis()  # Image coordinates have y increasing downward

    # Connect points in order to visualize ordering
    for i in range(len(world_points) - 1):
        axes[0].plot([world_points[i, 0], world_points[i + 1, 0]],
                     [world_points[i, 1], world_points[i + 1, 1]], 'r-', alpha=0.3)
        axes[1].plot([image_peaks[i, 0], image_peaks[i+1, 0]],
                    [image_peaks[i, 1], image_peaks[i+1, 1]], 'r-', alpha=0.3)

    plt.suptitle(title)
    plt.show()


def create_world_points(grid_shape, grid_spacing_meters):
    """
    grid_shape: (rows, cols) e.g., (3, 5) for 3 rows, 5 columns
    Returns: (N, 3) array of world points in row-major order (x varies fastest)
    """
    rows, cols = grid_shape
    world_points = np.zeros((rows * cols, 3), dtype=np.float32)

    # Column-major order: iterate rows, then columns
    idx = 0
    for c in range(cols):  # y coordinate (row)
        for r in range(rows):  # x coordinate (column)
            world_points[idx, 0] = c * grid_spacing_meters  # x
            world_points[idx, 1] = r * grid_spacing_meters  # y
            world_points[idx, 2] = 0
            idx += 1

    return world_points


def calibrate_from_pixel_peaks(peak_files_l: List[str],
                               peak_files_r: List[str],
                               grid_spacing_meters: float,
                               grid_shape: Tuple[int, int],  # (rows, cols)
                               save_to_folder: str,
                               show: bool = True):

    # Load peaks and sort them properly
    peaks_l = []
    peaks_r = []
    for f_l, f_r in zip(peak_files_l, peak_files_r):
        p_l = np.load(f_l).astype(np.float32)
        p_r = np.load(f_r).astype(np.float32)

        # Sort peaks to match world point order
        #p_l_sorted = sort_peaks_grid_order(p_l, grid_shape)
        #p_r_sorted = sort_peaks_grid_order(p_r, grid_shape)

        peaks_l.append(p_l)
        peaks_r.append(p_r)

    # Create world points with correct ordering
    world_points = create_world_points(grid_shape, grid_spacing_meters)
    world_points_copied = [world_points.copy() for _ in peak_files_l]

    # Verify first set of correspondences
    if show:
        for i in range(len(peaks_l)):
            verify_correspondences(world_points, peaks_l[i], f"Left Camera {peak_files_l[i]}")
            verify_correspondences(world_points, peaks_r[i], f"Right Camera {peak_files_r[i]}")

    _, camera_matrix_l, distortion_coeffs_l, _, _ = cv2.calibrateCamera(
        world_points_copied, peaks_l, (640, 480), None, None)
    _, camera_matrix_r, distortion_coeffs_r, _, _ = cv2.calibrateCamera(
        world_points_copied, peaks_r, (640, 480), None, None)

    # Stereo calibration with proper flags
    flags = 0 # Use individual calibration results
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

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
                                                     (640, 480),
                                                     criteria=criteria,
                                                     flags=flags)
    #rotation = np.eye(3)
    #translation = np.array([0.11, 0, 0])

    np.save(os.path.join(save_to_folder, "cam_mat_l.npy"), camera_matrix_l)
    np.save(os.path.join(save_to_folder, "cam_mat_r.npy"), camera_matrix_r)
    np.save(os.path.join(save_to_folder, "dist_coeffs_l.npy"),
            distortion_coeffs_l)
    np.save(os.path.join(save_to_folder, "dist_coeffs_r.npy"),
            distortion_coeffs_r)
    np.save(os.path.join(save_to_folder, "rotation.npy"), rotation)
    np.save(os.path.join(save_to_folder, "translation.npy"), translation)
    #np.save(os.path.join(save_to_folder, "essential.npy"), essential)
    #np.save(os.path.join(save_to_folder, "fundamental.npy"), fundamental)


def verify_epipolar_geometry(frame_path: str, calib_data_folder: str, n_grid_points: int):
    """Check if epipolar constraint holds with your calibration"""
    frame = np.load(frame_path)
    frame_l = frame[:640]
    frame_r = frame[640:]

    # Load calibration
    cam_mat_l = np.load(os.path.join(calib_data_folder, "cam_mat_l.npy"))
    cam_mat_r = np.load(os.path.join(calib_data_folder, "cam_mat_r.npy"))
    dist_coeffs_l = np.load(os.path.join(calib_data_folder, "dist_coeffs_l.npy"))
    dist_coeffs_r = np.load(os.path.join(calib_data_folder, "dist_coeffs_r.npy"))
    fundamental = np.load(os.path.join(calib_data_folder, "fundamental.npy"))
    essential = np.load(os.path.join(calib_data_folder, "essential.npy"))

    # Detect a few points to test
    peaks_l = locate_peaks(frame_l, n_grid_points)  # Get some points
    peaks_r = locate_peaks(frame_r, n_grid_points)

    # Test epipolar constraint: x_r^T * F * x_l = 0
    errors = []
    for i in range(min(10, len(peaks_l), len(peaks_r))):
        # Convert to homogeneous coordinates
        x_l = np.array([peaks_l[i, 0], peaks_l[i, 1], 1.0])
        x_r = np.array([peaks_r[i, 0], peaks_r[i, 1], 1.0])

        # Epipolar constraint
        error = abs(x_r.T @ fundamental @ x_l)
        errors.append(error)

        # Also compute using essential matrix (requires normalized coordinates)
        x_l_norm = np.linalg.inv(cam_mat_l) @ x_l
        x_r_norm = np.linalg.inv(cam_mat_r) @ x_r
        error_norm = abs(x_r_norm.T @ essential @ x_l_norm)

        print(f"Point {i}: F error={error:.3f}, E error={error_norm:.6f}")

    print(f"Mean F error: {np.mean(errors):.3f} (should be < 1.0)")


def test_stereo_calibration(frame_paths: Tuple[str, str], calib_data_folder: str):

    # Extract and process - KEEP TRANSPOSED (same as detection)
    frame_l_raw = np.load(frame_paths[0])
    frame_r_raw = np.load(frame_paths[1])

    # Apply Gaussian blur
    frame_l = gaussian(frame_l_raw, sigma=3)
    frame_r = gaussian(frame_r_raw, sigma=3)

    # Normalize to uint8
    frame_l = (frame_l / frame_l.max() * 255).astype(np.uint8)
    frame_r = (frame_r / frame_r.max() * 255).astype(np.uint8)

    w, h = frame_l.shape  # h=480, w=640

    # Load calibration
    cam_mat_l = np.load(os.path.join(calib_data_folder, "cam_mat_l.npy"))
    cam_mat_r = np.load(os.path.join(calib_data_folder, "cam_mat_r.npy"))
    dist_coeffs_l = np.load(os.path.join(calib_data_folder, "dist_coeffs_l.npy"))
    dist_coeffs_r = np.load(os.path.join(calib_data_folder, "dist_coeffs_r.npy"))
    rotation = np.load(os.path.join(calib_data_folder, "rotation.npy"))
    translation = np.load(os.path.join(calib_data_folder, "translation.npy"))
    """
    rotation = np.eye(3)
    translation = np.array([0.11, 0, 0])
    dist_coeffs_l = np.zeros_like(dist_coeffs_l)
    dist_coeffs_r = np.zeros_like(dist_coeffs_r)
    cam_mat_l[0, 2] = 320
    cam_mat_r[0, 2] = 320
    cam_mat_l[1, 2] = 240
    cam_mat_r[1, 2] = 240
    """

    print(f"Image shape: {h}x{w}")
    print(f"Camera matrix left:\n {cam_mat_l}")
    print(f"Camera matrix right:\n {cam_mat_r}")
    print(f"Translation:\n {translation}")
    print((f"Rotation:\n {rotation}"))

    # Stereo rectification
    #rectification_flags = cv2.CALIB_ZERO_DISPARITY
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cam_mat_l, dist_coeffs_l,
        cam_mat_r, dist_coeffs_r,
        (w, h), rotation, translation,  # (640, 480) for transposed images
        alpha=0
    )

    print(f"P1[0,2] (cx): {P1[0,2]:.1f} (should be ~320)")
    print(f"P1[1,2] (cy): {P1[1,2]:.1f} (should be ~240)")
    print(f"roi1: {roi1}")

    # Create rectification maps
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        cam_mat_l, dist_coeffs_l, R1, P1, (w, h), cv2.CV_32FC1
    )
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        cam_mat_r, dist_coeffs_r, R2, P2, (w, h), cv2.CV_32FC1
    )

    # Check if maps are valid
    print(f"Map1_left range: [{map1_left.min():.1f}, {map1_left.max():.1f}]")
    print(f"Map2_left range: [{map2_left.min():.1f}, {map2_left.max():.1f}]")

    # Apply rectification
    rectified_l = cv2.remap(frame_l.T, map1_left, map2_left, cv2.INTER_LINEAR)
    rectified_r = cv2.remap(frame_r.T, map1_right, map2_right, cv2.INTER_LINEAR)

    print(f"Rectified L - dtype: {rectified_l.dtype}, min: {rectified_l.min()}, max: {rectified_l.max()}")
    print(f"Rectified R - dtype: {rectified_r.dtype}, min: {rectified_r.min()}, max: {rectified_r.max()}")

    # Display WITHOUT additional transpose (images are already correct orientation)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0,0].imshow(frame_l.T, cmap='gray')
    axes[0,0].set_title("Left unrectified")
    axes[0,0].axis('off')

    axes[0,1].imshow(frame_r.T, cmap='gray')
    axes[0,1].set_title("Right unrectified")
    axes[0,1].axis('off')

    axes[1,0].imshow(rectified_l, cmap='gray')
    axes[1,0].set_title(f"Left rectified (roi: {roi1})")
    axes[1,0].axis('off')

    axes[1,1].imshow(rectified_r, cmap='gray')
    axes[1,1].set_title(f"Right rectified (roi: {roi2})")
    axes[1,1].axis('off')

    plt.tight_layout()
    plt.show()

    # If still blank, check if remap works with identity
    identity_map_x = np.arange(w, dtype=np.float32).reshape(1, -1).repeat(h, axis=0)
    identity_map_y = np.arange(h, dtype=np.float32).reshape(-1, 1).repeat(w, axis=1)
    test_remap = cv2.remap(frame_l, identity_map_x, identity_map_y, cv2.INTER_LINEAR)
    print(f"Identity remap test - min: {test_remap.min()}, max: {test_remap.max()}")
    if test_remap.max() > 0:
        print("Identity remap works - problem is in rectification maps")
    else:
        print("Identity remap failed - problem is in frame_l data")


def test_both_mono_calibrations(frame_paths: Tuple[str, str], calib_data_folder: str):
    """Test both left and right camera undistortion side by side"""
    #frame = np.load(frame_path)

    # Process left camera
    frame_l_raw = np.load(frame_paths[0])
    frame_l_blurred = gaussian(frame_l_raw, sigma=3)
    frame_l = (frame_l_blurred / frame_l_blurred.max() * 255).astype(np.uint8)

    cam_mat_l = np.load(os.path.join(calib_data_folder, "cam_mat_l.npy"))
    dist_coeffs_l = np.load(os.path.join(calib_data_folder, "dist_coeffs_l.npy"))
    undistorted_l = cv2.undistort(frame_l, cam_mat_l, dist_coeffs_l)

    # Process right camera
    frame_r_raw = np.load(frame_paths[1])
    frame_r_blurred = gaussian(frame_r_raw, sigma=3)
    frame_r = (frame_r_blurred / frame_r_blurred.max() * 255).astype(np.uint8)

    cam_mat_r = np.load(os.path.join(calib_data_folder, "cam_mat_r.npy"))
    dist_coeffs_r = np.load(os.path.join(calib_data_folder, "dist_coeffs_r.npy"))
    undistorted_r = cv2.undistort(frame_r, cam_mat_r, dist_coeffs_r)

    # Display all four images
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0,0].imshow(frame_l.T, cmap='gray')
    axes[0,0].set_title("Left - Original")
    axes[0,0].axis('off')

    axes[0,1].imshow(undistorted_l.T, cmap='gray')
    axes[0,1].set_title("Left - Undistorted")
    axes[0,1].axis('off')

    axes[1,0].imshow(frame_r.T, cmap='gray')
    axes[1,0].set_title("Right - Original")
    axes[1,0].axis('off')

    axes[1,1].imshow(undistorted_r.T, cmap='gray')
    axes[1,1].set_title("Right - Undistorted")
    axes[1,1].axis('off')

    plt.tight_layout()
    plt.show()



def debug_peak_ordering(frame_path: str, grid_shape: Tuple[int, int]):
    """Verify peak detection order matches world point order"""
    frame = np.load(frame_path)

    # Get peaks for both cameras
    frame_l = frame[:640]
    frame_r = frame[640:]

    peaks_l = locate_peaks(frame_l, grid_shape[0] * grid_shape[1])
    peaks_r = locate_peaks(frame_r, grid_shape[0] * grid_shape[1])
    peaks_l_flat_coords = peaks_l[:, 1] + 0.1*peaks_l[:, 0]
    peaks_l_perm = np.argsort(peaks_l_flat_coords)
    peaks_l = peaks_l[peaks_l_perm]
    peaks_r_flat_coords = peaks_r[:, 1] + 0.1*peaks_r[:, 0]
    peaks_r_perm = np.argsort(peaks_r_flat_coords)
    peaks_r = peaks_r[peaks_r_perm]

    # Create world points to see expected order
    world_points = create_world_points(grid_shape, 0.076)

    # Visualize left camera
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left camera with peak numbers
    axes[0].imshow(frame_l.T, cmap='gray')
    axes[0].set_title("Left Camera - Peak Ordering")
    for i, (x, y) in enumerate(peaks_l):
        axes[0].plot(x, y, 'ro', markersize=6)
        axes[0].annotate(str(i), (x+5, y+5), color='red', fontsize=8, weight='bold')

    # Draw lines connecting peaks in order (should form a grid pattern)
    for i in range(len(peaks_l)-1):
        axes[0].plot([peaks_l[i,0], peaks_l[i+1,0]],
                    [peaks_l[i,1], peaks_l[i+1,1]], 'b-', alpha=0.3)

    # Right camera with peak numbers
    axes[1].imshow(frame_r.T, cmap='gray')
    axes[1].set_title("Right Camera - Peak Ordering")
    for i, (x, y) in enumerate(peaks_r):
        axes[1].plot(x, y, 'bo', markersize=6)
        axes[1].annotate(str(i), (x+5, y+5), color='blue', fontsize=8, weight='bold')

    for i in range(len(peaks_r)-1):
        axes[1].plot([peaks_r[i,0], peaks_r[i+1,0]],
                    [peaks_r[i,1], peaks_r[i+1,1]], 'r-', alpha=0.3)

    plt.suptitle(f"Expected grid shape: {grid_shape[0]} rows, {grid_shape[1]} cols\n"
                 f"Peaks should connect in row-major order (left-to-right, top-to-bottom)")
    plt.show()

    print(f"Number of peaks detected: Left={len(peaks_l)}, Right={len(peaks_r)}")
    print(f"Expected: {grid_shape[0] * grid_shape[1]}")

    # Check if peaks form a grid (approximate)
    if len(peaks_l) == grid_shape[0] * grid_shape[1]:
        # Check X coordinates - should have grid_shape[1] distinct values
        x_unique = np.unique(np.round(peaks_l[:, 0]))
        y_unique = np.unique(np.round(peaks_l[:, 1]))
        print(f"Unique X positions (should be ~{grid_shape[1]}): {len(x_unique)}")
        print(f"Unique Y positions (should be ~{grid_shape[0]}): {len(y_unique)}")


if __name__ == "__main__":


    peak_files_l = []
    peak_files_r = []
    for ff in frame_files:
        print(ff)
        peaks_l, peaks_r = detect_peaks_from_frames(ff, 25, show=True)
        _, tail_left = os.path.split(ff[0])
        _, tail_right = os.path.split(ff[1])
        pathl = os.path.join("grid_points", "peaks_" + tail_left)
        pathr = os.path.join("grid_points", "peaks_" + tail_right)
        np.save(pathl, peaks_l)
        np.save(pathr, peaks_r)
        peak_files_l.append(pathl)
        peak_files_r.append(pathr)

    peak_files_l = [os.path.join("grid_points", f) for f in os.listdir("grid_points") if 'left' in f]
    peak_files_r = [os.path.join("grid_points", f) for f in os.listdir("grid_points") if 'right' in f]
    calibrate_from_pixel_peaks(
        peak_files_l, peak_files_r, 0.04, (5, 5), "mimir_jr_calibration_data", show=True)

    for ff in frame_files:
        #`debug_peak_ordering(ff, (4, 5))
        #verify_epipolar_geometry(ff, "calibration_data", 20)
        test_both_mono_calibrations(ff, "mimir_jr_calibration_data")
        test_stereo_calibration(ff, "mimir_jr_calibration_data")
