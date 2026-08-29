from itertools import combinations

import cv2
import cv2.aruco as ar
import numpy as np

import pgs_recon.utils.charuco as char


def unit_vec(v):
    """Return a unit length v"""
    return v / np.sqrt(np.sum(v ** 2))


def find_nearest(value, array):
    return np.abs(np.asarray(array) - value).argmin()


# Physical positions of known keypoints relative to top-left corner of sample
# square. See the Sample Square v1 Placement Guide in:
# https://gitlab.com/educelab/acquisition-workflow
_SAMPLE_SQUARE_V1_KP_POS_CM = [
    # M0,0 -> M0,3
    np.array((0.966666666666667, 0.3)),
    np.array((0.3, 0.966666666666667)),
    np.array((1.633333333333334, 0.966666666666667)),
    np.array((0.966666666666667, 1.633333333333334)),
    # C0,0 -> C0,3
    np.array((0.866623541666667, 0.866623541666667)),
    np.array((1.533290208333333, 0.866623541666667)),
    np.array((0.866623541666667, 1.533290208333333)),
    np.array((1.533290208333333, 1.533290208333333)),
    # M1,0 -> M1,3
    np.array((0.966666666667, 13.4)),
    np.array((0.3, 14.066666666666667)),
    np.array((1.633333333333334, 14.066666666666667)),
    np.array((0.966666666666667, 14.733333333333334)),
    # C1,0 -> C1,3
    np.array((0.866623541666667, 13.966623541666667)),
    np.array((1.533290208333333, 13.966623541666667)),
    np.array((0.866623541666667, 14.633290208333333)),
    np.array((1.533290208333333, 14.633290208333333))
]

# Pre-calculated distances between each keypoint in cm. This list does not
# store duplicates. For example, the value for D(0, 10) and D(10, 0) is only
# stored in list[0][10]. Use keypoint_distance(a, b) to easily get distances
# between arbitrary keys.
_SAMPLE_SQUARE_V1_KP_DIST_CM = [
    [np.linalg.norm(b - a) for b in _SAMPLE_SQUARE_V1_KP_POS_CM[idx + 1:]] for
    idx, a in enumerate(_SAMPLE_SQUARE_V1_KP_POS_CM[:-1])]


def kp_dist(a, b):
    """Get the distance from keypoint a to b"""
    if a == b:
        return 0.
    if b < a:
        a, b = b, a
    return _SAMPLE_SQUARE_V1_KP_DIST_CM[a][b - a - 1]


def kp_dir(a, b):
    """Get the direction from keypoint a to b"""
    return _SAMPLE_SQUARE_V1_KP_POS_CM[b] - _SAMPLE_SQUARE_V1_KP_POS_CM[a]


def rotate_kp(kp, dim, rot):
    if rot == 0:
        u = dim[1] - kp[1]
        v = kp[0]
    elif rot == 1:
        u = dim[0] - kp[0]
        v = dim[1] - kp[1]
    else:
        u = kp[1]
        v = dim[0] - kp[0]
    return np.array((u, v))


# Detect the EduceLab sample square in an image
def detect_sample_square(img):
    # Check all image orientations
    flip = None
    boards, kp_ids, kp_pos = _detect_educelab_boards(img)
    for axis in 1, 0:
        flipped = cv2.flip(img, axis)
        b_new, kp_ids_new, kp_pos_new = _detect_educelab_boards(flipped)
        if len(kp_ids_new) > len(kp_ids):
            boards = b_new
            kp_ids = kp_ids_new
            kp_pos = kp_pos_new
            flip = axis

    # Make sure we have at least two landmarks
    num_ldms = sum((b.marker_cnt + b.board_cnt for b in boards))
    detected = num_ldms > 1

    # Calculate pixels-per-cm
    ppcm = 0.
    rotate = None
    if num_ldms > 1:
        ppc_samples = []
        rot_samples = []
        for ids, pts in zip(combinations(kp_ids, r=2),
                            combinations(kp_pos, r=2)):
            # Skip keypoints which have matching IDs
            if ids[0] == ids[1]:
                continue

            # calculate ppcm for this kp pair
            dist_px = np.linalg.norm(pts[1] - pts[0])
            dist_cm = kp_dist(ids[0], ids[1])
            ppc_samples.append(dist_px / dist_cm)

            # get expected vector directions
            dir_px = unit_vec(pts[1] - pts[0])
            dir_cm = unit_vec(kp_dir(ids[0], ids[1]))

            # calculate rotation for this key-value pair
            y = dir_px[0] * dir_cm[1] - dir_px[1] * dir_cm[0]
            x = dir_px[0] * dir_cm[0] + dir_px[1] * dir_cm[1]
            theta = np.arctan2(y, x)
            # normalize to [0, 2pi]
            theta -= 2 * np.pi * (theta // (2*np.pi))
            if find_nearest(theta, [1.5 * np.pi, 2 * np.pi]) == 1:
                theta = 0.
            rot_samples.append(theta)

        ppcm = np.mean(ppc_samples)
        theta = np.mean(rot_samples)

        # Rotate detected key points
        rot = find_nearest(theta, [0., np.pi / 2, np.pi, 1.5 * np.pi, 2 * np.pi]) - 1
        if 0 <= rot < 3:
            rotate = rot
            max_x = img.shape[1] - 1
            max_y = img.shape[0] - 1
            dim = (max_x, max_y)
            for idx, kp in enumerate(kp_pos):
                kp_pos[idx] = rotate_kp(kp, dim, rot)
            for b in boards:
                for idx, c in enumerate(b.board_corners):
                    b.board_corners[idx, 0] = rotate_kp(c[0], dim, rot)
                for idx, marker in enumerate(b.marker_corners):
                    m = np.array([rotate_kp(c, dim, rot) for c in marker[0]])
                    b.marker_corners[idx][0] = m

    return detected, boards, ppcm, kp_ids, kp_pos, flip, rotate


# Low-level function for detecting the two EduceLab boards from the sample
# square
def _detect_educelab_boards(img):
    # Results
    boards = []
    kp_ids = []
    kp_pos = []
    # Try to detect both boards
    for idx in range(2):
        board = char.generate_board(offset=idx * 512)

        # Detect board
        b = char.detect_board(img, board)

        # Shift IDs to [0, 15]
        if b.marker_corners is not None and len(b.marker_corners) > 0:
            # can't update directly through += so update reference
            ids = b.marker_ids
            ids += idx * 8
            kp_ids.extend(ids.flatten().tolist())
            kp_pos.extend(c[:, 0, :].squeeze() for c in b.marker_corners)
        if b.board_corners is not None and len(b.board_corners) > 0:
            # can't update directly through += so update reference
            ids = b.board_ids
            ids += (idx * 8) + 4
            kp_ids.extend(ids.flatten().tolist())
            kp_pos.extend(c[0, :].squeeze() for c in b.board_corners)

        boards.append(b)
    return boards, kp_ids, kp_pos


def generate_tray_mask(img, open_iterations=4, save_debug=False):
    # Setup gray scale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold kernel size
    kt = np.min(gray.shape[:2])
    kt = max(kt // 100, 11)
    kt = kt + 1 if (kt % 2 == 0) else kt

    # gaussian kernel size, bilateral sigma
    kb = max(kt, 3)
    kb = kb + 1 if (kb % 2 == 0) else kb
    sigma = max(kb // 4, 9)
    gray = cv2.GaussianBlur(gray, (kb, kb), 0)
    gray = cv2.bilateralFilter(gray, -1, sigma, sigma)
    if save_debug:
        cv2.imwrite('debug-mask-1-blur.jpg', gray)

    # threshold image
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, kt, 2)
    if save_debug:
        cv2.imwrite('debug-mask-2-thresh.jpg', gray)

    # morphological filter
    km = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, km,
                            iterations=open_iterations)
    gray = cv2.bitwise_not(gray)
    if save_debug:
        cv2.imwrite('debug-mask-3-morph.jpg', gray)

    # detect contours then sort through to find the inner contours
    cnts, hierarchy = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    inner_contours = []
    for i, cnt in enumerate(cnts):
        if hierarchy[0, i, 3] != -1:
            inner_contours.append(cnt)

    # sort by size and get largest
    cnt = sorted(inner_contours, key=lambda c: cv2.contourArea(c), reverse=True)[0]
    if save_debug:
        contour_img = img.copy()
        cv2.drawContours(contour_img, [cnt], -1, [0, 255, 0], thickness=2)
        cv2.imwrite('debug-mask-4-contour.jpg', contour_img)

    # fit bounding box
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # generate mask image
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [box], 0, 255, thickness=cv2.FILLED)
    return mask

def main():
    # local imports
    import cv2
    import argparse

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-image', '-i', required=True,
                        help='Input image')
    parser.add_argument('--output-image', '-o',
                        help='Draw detected markers onto the input image and '
                             'save to the provided file path')
    args = parser.parse_args()

    # Load image
    img = cv2.imread(args.input_image)

    if img is None:
        print('Failed to load image/image empty')
        return
    else:
        print(f'Loaded image: {img.shape}')

    # Detect boards
    detected, boards, ppcm, _, _, flip, rotate = detect_sample_square(img)

    # Flip image
    if flip is not None:
        print(f'Flipping image along axis {flip}')
        img = cv2.flip(img, flip)

    # Rotate image
    if rotate is not None:
        msg = ['90°', '180°', '270°']
        print(f'Rotating image {msg[rotate]}')
        img = cv2.rotate(img, rotate)

    # Draw detected markers
    if detected:
        # Print results
        num_markers = sum((b.marker_cnt for b in boards))
        num_boards = sum((b.board_cnt for b in boards))
        print(f'Detected:\n'
              f' - Markers: {num_markers}\n'
              f' - Board corners: {num_boards}\n'
              f' - Texture resolution (pixels/cm): {ppcm}')

        # Draw each board
        if args.output_image is not None:
            for b in boards:
                # Draw markers and board corners
                if b.marker_corners is not None and len(b.marker_corners) > 0:
                    ar.drawDetectedMarkers(img, b.marker_corners, b.marker_ids)
                if b.board_corners is not None and len(b.board_corners) > 0:
                    ar.drawDetectedCornersCharuco(img, b.board_corners,
                                                  b.board_ids)
            cv2.imwrite(args.output_image, img)
    else:
        print('No markers detected')


if __name__ == '__main__':
    main()
