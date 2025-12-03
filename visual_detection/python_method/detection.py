import cv2
import numpy as np
import time
import os

# =============== 1. 在这里修改黑白子阈值 ===============
# 解释：
# 图像灰度范围是 0 (纯黑) - 255 (纯白)
#
# BINARY_THRESH_LOW:  小于这个值的像素被认为是【黑子】
#   - 如果黑子识别不到，把这个值调高 (例如 80, 90)
#   - 如果把阴影误识别为黑子，把这个值调低 (例如 50, 60)
BINARY_THRESH_LOW = 60

# BINARY_THRESH_HIGH: 大于这个值的像素被认为是【白子】
#   - 如果白子识别不到，把这个值调低 (例如 160, 150)
#   - 如果把反光的木板误识别为白子，把这个值调高 (例如 200)
BINARY_THRESH_HIGH = 200
# ======================================================

CANNY_LOW = 30
CANNY_HIGH = 100
SMOOTH_ALPHA = 0.7

EMPTY = 0
BLACK = 1
WHITE = 2
STATE_SEARCHING = 0
STATE_TRACKING = 1

g_camera_matrix = None
g_dist_coeffs = None
g_map1, g_map2 = None, None
g_has_calibration = False
g_board_data = np.zeros((15, 15), dtype=int)
g_last_xs = np.zeros(15, dtype=float)
g_last_ys = np.zeros(15, dtype=float)
g_grid_initialized = False


def init_system(img_size):
    global g_camera_matrix, g_dist_coeffs, g_map1, g_map2, g_has_calibration
    calib_file = "cam_params.xml"
    if os.path.exists(calib_file):
        try:
            fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
            g_camera_matrix = fs.getNode("camera_matrix").mat()
            g_dist_coeffs = fs.getNode("dist_coeffs").mat()
            fs.release()
            new_cam_mat, _ = cv2.getOptimalNewCameraMatrix(
                g_camera_matrix, g_dist_coeffs, img_size, 0, img_size
            )
            g_map1, g_map2 = cv2.initUndistortRectifyMap(
                g_camera_matrix,
                g_dist_coeffs,
                None,
                new_cam_mat,
                img_size,
                cv2.CV_16SC2,
            )
            g_has_calibration = True
            print("✅ 标定文件已加载。")
        except:
            pass
    else:
        print("⚠️ 未找到标定文件，使用原始画面。")


def sort_corners(pts):
    pts = pts.reshape(4, 2)
    sorted_y = pts[np.argsort(pts[:, 1])]
    top = sorted_y[:2]
    bottom = sorted_y[2:]
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


def is_geo_valid(pts):
    pts = pts.reshape(4, 2)
    d1 = np.linalg.norm(pts[0] - pts[1])
    d2 = np.linalg.norm(pts[2] - pts[3])
    if d2 == 0:
        return False
    if max(d1, d2) / min(d1, d2) > 2.0:
        return False
    return cv2.isContourConvex(pts.astype(np.int32))


def find_board_auto(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_cnt = None
    found = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40000:
            continue
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            if area > max_area:
                max_area = area
                best_cnt = approx
                found = True
    if found:
        corners = best_cnt.astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        return True, sort_corners(corners)
    return False, None


def solve_grid_lines(proj, max_len, last_data):
    global g_grid_initialized
    # 稍微加强平滑，防止断线
    temp = cv2.GaussianBlur(proj, (3, 3), 0).flatten()
    approx_step = max_len // 14
    peaks = []

    # 简单的峰值查找
    # 这里为了防止边缘丢失，降低一点阈值 800 -> 500
    search_temp = temp.copy()
    for _ in range(20):
        idx = np.argmax(search_temp)
        if search_temp[idx] < 500:
            break
        peaks.append(idx)
        start = max(0, idx - approx_step // 3)
        end = min(len(search_temp), idx + approx_step // 3)
        search_temp[start:end] = 0
    peaks.sort()

    coords = []
    if len(peaks) >= 2:
        gaps = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
        median_step = int(np.median(gaps))
        if median_step < 10:
            median_step = approx_step

        # 找基准线
        center_idx = 0
        min_dist = max_len
        for i, p in enumerate(peaks):
            if abs(p - max_len / 2) < min_dist:
                min_dist = abs(p - max_len / 2)
                center_idx = i
        anchor = peaks[center_idx]

        estimated_idx = round((anchor / max_len) * 14.0)
        estimated_idx = max(0, min(14, estimated_idx))

        for i in range(15):
            pos = anchor + (i - estimated_idx) * median_step
            coords.append(pos)
    else:
        for i in range(15):
            coords.append(20 + i * (max_len - 40) / 14.0)

    coords = np.array(coords, dtype=float)
    if not g_grid_initialized:
        np.copyto(last_data, coords)
    else:
        for i in range(15):
            if abs(coords[i] - last_data[i]) > 20:
                last_data[i] = coords[i]
            else:
                last_data[i] = last_data[i] * SMOOTH_ALPHA + coords[i] * (
                    1.0 - SMOOTH_ALPHA
                )
            coords[i] = round(last_data[i])
    return coords.astype(int)


def find_dynamic_grid_lines(warped, debug_disp):
    global g_grid_initialized
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5
    )

    rows, cols = warped.shape[:2]
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // 10))
    v_lines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel_v)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // 10, 1))
    h_lines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel_h)

    col_proj = np.sum(v_lines, axis=0).astype(np.float32)
    row_proj = np.sum(h_lines, axis=1).astype(np.float32)  # 注意这里是 axis=1

    xs = solve_grid_lines(col_proj, cols, g_last_xs)
    ys = solve_grid_lines(row_proj, rows, g_last_ys)
    g_grid_initialized = True

    for x in xs:
        cv2.line(debug_disp, (x, 0), (x, rows), (0, 0, 255), 1)
    for y in ys:
        cv2.line(debug_disp, (0, y), (cols, y), (0, 0, 255), 1)
    return xs, ys


def scan_pieces(warped, xs, ys, debug_disp):
    global g_board_data
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, bin_black = cv2.threshold(gray, BINARY_THRESH_LOW, 255, cv2.THRESH_BINARY_INV)
    _, bin_white = cv2.threshold(gray, BINARY_THRESH_HIGH, 255, cv2.THRESH_BINARY)

    roi_s = 10
    h, w = warped.shape[:2]
    for r in range(15):
        for c in range(15):
            cx = xs[c] if c < len(xs) else c * 30
            cy = ys[r] if r < len(ys) else r * 30
            x1, y1 = max(0, cx - roi_s // 2), max(0, cy - roi_s // 2)
            x2, y2 = min(w, cx + roi_s // 2), min(h, cy + roi_s // 2)

            roi_b = bin_black[y1:y2, x1:x2]
            roi_w = bin_white[y1:y2, x1:x2]
            if roi_b.size == 0:
                continue

            area = (x2 - x1) * (y2 - y1)
            if cv2.countNonZero(roi_b) > area * 0.4:
                g_board_data[r][c] = BLACK
                cv2.circle(debug_disp, (cx, cy), 4, (0, 0, 255), -1)
            elif cv2.countNonZero(roi_w) > area * 0.4:
                g_board_data[r][c] = WHITE
                cv2.circle(debug_disp, (cx, cy), 4, (255, 0, 0), -1)
            else:
                g_board_data[r][c] = EMPTY


def draw_virtual_board(s):
    board = np.full((s, s, 3), (160, 200, 230), dtype=np.uint8)
    m = 30
    step = (s - 2 * m) / 14.0
    for i in range(15):
        p = int(round(m + i * step))
        cv2.line(board, (m, p), (s - m, p), (0, 0, 0), 1)
        cv2.line(board, (p, m), (p, s - m), (0, 0, 0), 1)
    for r in range(15):
        for c in range(15):
            if g_board_data[r][c] == EMPTY:
                continue
            cx = int(round(m + c * step))
            cy = int(round(m + r * step))
            if g_board_data[r][c] == BLACK:
                cv2.circle(board, (cx, cy), 13, (10, 10, 10), -1)
            else:
                cv2.circle(board, (cx, cy), 13, (240, 240, 240), -1)
                cv2.circle(board, (cx, cy), 13, (100, 100, 100), 1)
    return board


def main():
    init_system((640, 480))
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    WARP_SIZE = 450
    PADDING = 20
    dst_pts = np.array(
        [
            [PADDING, PADDING],
            [WARP_SIZE - PADDING, PADDING],
            [WARP_SIZE - PADDING, WARP_SIZE - PADDING],
            [PADDING, WARP_SIZE - PADDING],
        ],
        dtype=np.float32,
    )

    cv2.namedWindow("Main View")
    cv2.namedWindow("Analysis")
    cv2.namedWindow("Virtual Board")

    state = STATE_SEARCHING
    prev_gray = None
    points = None
    lost_cnt = 0

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        if g_has_calibration:
            frame = cv2.remap(raw_frame, g_map1, g_map2, cv2.INTER_LINEAR)
        else:
            frame = raw_frame.copy()  # 这里获得的是干净的画面

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if state == STATE_SEARCHING:
            found, detected_pts = find_board_auto(frame)
            if found:
                points = detected_pts
                prev_gray = curr_gray.copy()
                state = STATE_TRACKING
                lost_cnt = 0
            cv2.putText(
                frame,
                "SEARCHING...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        else:  # TRACKING
            p0 = points.reshape(-1, 1, 2)
            p1, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, p0, None, winSize=(31, 31), maxLevel=3
            )
            good_pts = []
            if status is not None:
                good_pts = p1[status == 1]

            if len(good_pts) < 4 or not is_geo_valid(good_pts):
                lost_cnt += 1
                if lost_cnt > 10:
                    state = STATE_SEARCHING
            else:
                points = good_pts.reshape(4, 2)
                prev_gray = curr_gray.copy()
                lost_cnt = 0

                # ================= 核心修正 =================
                # 1. 先进行分析 (使用还未被画线的 frame)
                H = cv2.getPerspectiveTransform(points.astype(np.float32), dst_pts)
                warped = cv2.warpPerspective(frame, H, (WARP_SIZE, WARP_SIZE))

                debug_disp = warped.copy()
                xs, ys = find_dynamic_grid_lines(warped, debug_disp)
                scan_pieces(warped, xs, ys, debug_disp)
                cv2.imshow("Analysis", debug_disp)

                # 2. 分析完了，再在 frame 上画框给用户看
                for i in range(4):
                    cv2.line(
                        frame,
                        tuple(points[i].astype(int)),
                        tuple(points[(i + 1) % 4].astype(int)),
                        (0, 255, 0),
                        2,
                    )
                # ==========================================

            cv2.putText(
                frame, "TRACKING", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

        cv2.imshow("Main View", frame)
        cv2.imshow("Virtual Board", draw_virtual_board(500))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            state = STATE_SEARCHING
            g_grid_initialized = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
