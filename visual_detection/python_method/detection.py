import cv2
import numpy as np
import time
import os
import threading
import copy

# ================= 配置区域 =================
# 稍微调严一点白子阈值，防止反光误判
BINARY_THRESH_LOW = 100    # 黑子 (越低越严)
BINARY_THRESH_HIGH = 200  # 白子 (越低越容易识别，太低会把地板当白子，建议 160-200 之间微调)
# ===========================================

# 状态常量
EMPTY = 0
BLACK = 1
WHITE = 2

class GobangVision:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # 最终输出的稳定棋盘数据
        self.board_data = np.zeros((15, 15), dtype=int)
        
        # 【新增】防抖投票箱 (-10 ~ 10)
        # > 5 确认为黑子
        # < -5 确认为白子
        # 0 附近为空
        self.vote_matrix = np.zeros((15, 15), dtype=int)
        
        # 内部变量
        self.g_has_calibration = False
        self.g_map1, self.g_map2 = None, None
        self.g_last_xs = np.zeros(15, dtype=float)
        self.g_last_ys = np.zeros(15, dtype=float)
        self.g_grid_initialized = False
        
        self._init_calibration((640, 480))

    # ==================== 外部接口 ====================
    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()
        print("[Vision] 视觉防抖模式已启动...")

    def stop(self):
        self.running = False
        if self.thread: self.thread.join()

    def get_current_board(self):
        with self.lock:
            return copy.deepcopy(self.board_data)

    def get_black_coordinates(self):
        with self.lock:
            # 使用 .tolist() 将 np.int64 转换为标准 python int
            coords = np.argwhere(self.board_data == BLACK)
            return [tuple(c.tolist()) for c in coords]

    def get_white_coordinates(self):
        with self.lock:
            coords = np.argwhere(self.board_data == WHITE)
            return [tuple(c.tolist()) for c in coords]

    # ==================== 内部逻辑 ====================
    def _processing_loop(self):
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        WARP_SIZE = 450
        PADDING = 20
        dst_pts = np.array([
            [PADDING, PADDING], [WARP_SIZE - PADDING, PADDING],
            [WARP_SIZE - PADDING, WARP_SIZE - PADDING], [PADDING, WARP_SIZE - PADDING],
        ], dtype=np.float32)

        state = 0 # SEARCHING
        points = None
        prev_gray = None
        lost_cnt = 0

        while self.running:
            ret, raw_frame = cap.read()
            if not ret: time.sleep(0.01); continue

            if self.g_has_calibration: frame = cv2.remap(raw_frame, self.g_map1, self.g_map2, cv2.INTER_LINEAR)
            else: frame = raw_frame.copy()

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if state == 0: # SEARCHING
                found, detected_pts = self._find_board_auto(frame)
                if found:
                    points = detected_pts
                    prev_gray = curr_gray.copy()
                    state = 1
                    lost_cnt = 0
                cv2.putText(frame, "SEARCHING...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            else: # TRACKING
                p0 = points.reshape(-1, 1, 2)
                p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, winSize=(31, 31), maxLevel=3)
                good_pts = []
                if status is not None: good_pts = p1[status == 1]

                if len(good_pts) < 4 or not self._is_geo_valid(good_pts):
                    lost_cnt += 1
                    if lost_cnt > 10: state = 0
                else:
                    points = good_pts.reshape(4, 2)
                    prev_gray = curr_gray.copy()
                    lost_cnt = 0

                    H = cv2.getPerspectiveTransform(points.astype(np.float32), dst_pts)
                    warped = cv2.warpPerspective(frame, H, (WARP_SIZE, WARP_SIZE))
                    
                    debug_disp = warped.copy()
                    xs, ys = self._find_dynamic_grid_lines(warped, debug_disp)
                    
                    # 1. 获取当前帧的原始识别结果 (不直接写入全局)
                    raw_board = self._scan_pieces_raw(warped, xs, ys, debug_disp)
                    
                    # 2. 进行投票防抖处理 (核心修改)
                    self._update_votes(raw_board)

                    # 3. 绘图 (用稳定后的结果画图，看起来更稳)
                    self._draw_debug_overlay(debug_disp, xs, ys)
                    
                    cv2.imshow("Analysis", debug_disp)
                    for i in range(4): cv2.line(frame, tuple(points[i].astype(int)), tuple(points[(i + 1) % 4].astype(int)), (0, 255, 0), 2)

            cv2.imshow("Main View", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): self.running = False

        cap.release()
        cv2.destroyAllWindows()

    # ==================== 核心防抖算法 ====================
    def _update_votes(self, raw_board):
        """
        根据当前帧的识别结果 raw_board，更新 vote_matrix。
        机制：
        - 看到黑子: +1 分
        - 看到白子: -1 分
        - 看到空位: 向 0 靠拢
        - 积分范围限制在 -10 到 10 之间
        - 只有积分超过阈值 (5 或 -5) 才更新最终结果
        """
        # 1. 更新投票箱
        for r in range(15):
            for c in range(15):
                val = raw_board[r][c]
                
                if val == BLACK:
                    self.vote_matrix[r][c] += 2 # 加分快一点 (灵敏度)
                elif val == WHITE:
                    self.vote_matrix[r][c] -= 2
                else:
                    # 如果当前帧是空的，让分数慢慢归零
                    if self.vote_matrix[r][c] > 0:
                        self.vote_matrix[r][c] -= 1
                    elif self.vote_matrix[r][c] < 0:
                        self.vote_matrix[r][c] += 1

        # 2. 限制范围 (-10 到 10)
        self.vote_matrix = np.clip(self.vote_matrix, -10, 10)

        # 3. 根据分数决定最终结果
        with self.lock:
            for r in range(15):
                for c in range(15):
                    score = self.vote_matrix[r][c]
                    # 连续几帧确认为黑子
                    if score > 6: 
                        self.board_data[r][c] = BLACK
                    # 连续几帧确认为白子
                    elif score < -6:
                        self.board_data[r][c] = WHITE
                    # 分数不高不低，认为是空
                    elif -3 < score < 3:
                        self.board_data[r][c] = EMPTY

    # ==================== 辅助函数 ====================
    def _scan_pieces_raw(self, warped, xs, ys, debug_disp):
        """扫描单帧图像，返回一个临时的 board 矩阵"""
        raw_board = np.zeros((15, 15), dtype=int)
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
                if roi_b.size == 0: continue
                
                area = (x2 - x1) * (y2 - y1)
                # 这里稍微改严格了一点点 0.4 -> 0.45
                if cv2.countNonZero(roi_b) > area * 0.45:
                    raw_board[r][c] = BLACK
                elif cv2.countNonZero(roi_w) > area * 0.45:
                    raw_board[r][c] = WHITE
        return raw_board

    def _draw_debug_overlay(self, img, xs, ys):
        """根据最终稳定的 board_data 画圈圈"""
        with self.lock:
            for r in range(15):
                for c in range(15):
                    if self.board_data[r][c] == EMPTY: continue
                    
                    cx = xs[c] if c < len(xs) else c * 30
                    cy = ys[r] if r < len(ys) else r * 30
                    
                    if self.board_data[r][c] == BLACK:
                        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1) # 红点表示检测到黑子
                    elif self.board_data[r][c] == WHITE:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1) # 蓝点表示检测到白子

    def _init_calibration(self, img_size):
        # (保持原有的标定代码不变)
        if os.path.exists("cam_params.xml"):
            try:
                fs = cv2.FileStorage("cam_params.xml", cv2.FILE_STORAGE_READ)
                cam = fs.getNode("camera_matrix").mat()
                dist = fs.getNode("dist_coeffs").mat()
                fs.release()
                n_cam, _ = cv2.getOptimalNewCameraMatrix(cam, dist, img_size, 0, img_size)
                self.g_map1, self.g_map2 = cv2.initUndistortRectifyMap(cam, dist, None, n_cam, img_size, cv2.CV_16SC2)
                self.g_has_calibration = True
            except: pass

    def _find_board_auto(self, src):
        # (保持原有的自动寻线代码不变)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        edges = cv2.dilate(edges, np.ones((3,3)), iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = None; max_area = 0; found = False
        for cnt in contours:
            if cv2.contourArea(cnt) < 40000: continue
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt,True), True)
            if len(approx)==4 and cv2.isContourConvex(approx):
                if cv2.contourArea(cnt) > max_area:
                    max_area = cv2.contourArea(cnt); best_cnt = approx; found = True
        if found:
            corners = cv2.cornerSubPix(gray, best_cnt.astype(np.float32), (5,5), (-1,-1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT, 30, 0.1))
            return True, self._sort_corners(corners)
        return False, None

    def _sort_corners(self, pts):
        # (保持不变)
        pts = pts.reshape(4, 2)
        sorted_y = pts[np.argsort(pts[:, 1])]
        top, bottom = sorted_y[:2], sorted_y[2:]
        return np.array([top[np.argsort(top[:, 0])][0], top[np.argsort(top[:, 0])][1], bottom[np.argsort(bottom[:, 0])][1], bottom[np.argsort(bottom[:, 0])][0]], dtype=np.float32)

    def _is_geo_valid(self, pts):
        # (保持不变)
        pts = pts.reshape(4, 2)
        d1 = np.linalg.norm(pts[0]-pts[1]); d2 = np.linalg.norm(pts[2]-pts[3])
        return False if d2==0 or max(d1,d2)/min(d1,d2)>2.0 else cv2.isContourConvex(pts.astype(np.int32))

    def _find_dynamic_grid_lines(self, warped, debug_disp):
        # (保持不变，但去掉了画图部分，因为移到了专门的 draw 函数)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5)
        h, w = warped.shape[:2]
        v_lines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, h//15)))
        h_lines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (w//15, 1)))
        
        xs = self._solve_grid_lines(np.sum(v_lines, axis=0).astype(np.float32), w, self.g_last_xs)
        ys = self._solve_grid_lines(np.sum(h_lines, axis=1).astype(np.float32), h, self.g_last_ys)
        self.g_grid_initialized = True
        # 网格线绘制
        if debug_disp is not None:
            for x in xs: cv2.line(debug_disp, (x, 0), (x, h), (0, 0, 255), 1)
            for y in ys: cv2.line(debug_disp, (0, y), (w, y), (0, 0, 255), 1)
        return xs, ys

    def _solve_grid_lines(self, proj, max_len, last_data):
        # (保持不变)
        temp = cv2.GaussianBlur(proj, (3, 3), 0).flatten()
        peaks = []
        for _ in range(20):
            idx = np.argmax(temp)
            if temp[idx] < np.max(temp)*0.15: break
            peaks.append(idx)
            temp[max(0, idx-max_len//42):min(len(temp), idx+max_len//42)] = 0
        peaks.sort()
        coords = []
        if len(peaks) >= 2:
            gaps = [peaks[i+1]-peaks[i] for i in range(len(peaks)-1)]
            median = int(np.median(gaps)) if len(gaps)>0 else max_len//14
            anchor = peaks[np.argmin([abs(p-max_len/2) for p in peaks])]
            for i in range(15): coords.append(anchor + (i - 7) * (median if median > 10 else max_len//14))
        else:
             for i in range(15): coords.append(20 + i * (max_len - 40) / 14.0)
        coords = np.array(coords, dtype=float)
        if not self.g_grid_initialized: np.copyto(last_data, coords)
        else: 
            for i in range(15):
                if abs(coords[i]-last_data[i])>20: last_data[i]=coords[i]
                else: last_data[i]=last_data[i]*0.7+coords[i]*0.3
                coords[i]=round(last_data[i])
        return coords.astype(int)
