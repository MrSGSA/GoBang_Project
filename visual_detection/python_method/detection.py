import cv2
import numpy as np
import time
import threading
import copy
import os

# =================参数设置=================
HOUGH_PARAM2 = 20       
MIN_RADIUS = 12         
MAX_RADIUS = 23         
COLOR_THRESH = 140      
BOARD_SIZE = 19
EMPTY, BLACK, WHITE = 0, 1, 2
STATE_SEARCHING = 0
STATE_LOCKED = 1

# 电脑端使用的分辨率 (大多数USB摄像头支持此分辨率)
CAM_WIDTH = 1280
CAM_HEIGHT = 720

class GobangVision:
    def __init__(self, camera_id=0, rotate_image=0):
        self.camera_id = camera_id
        self.rotate_image = rotate_image 
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        self.board_data = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.vote_matrix = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        
        self.calib_mtx = None
        self.calib_dist = None
        self.new_cam_mtx = None
        self.has_calib = False
        
        # 加载校准文件，注意尺寸改为 USB 摄像头的尺寸
        self._load_calibration((CAM_WIDTH, CAM_HEIGHT))
        
        self.last_xs = np.zeros(BOARD_SIZE, dtype=float)
        self.last_ys = np.zeros(BOARD_SIZE, dtype=float)
        self.grid_initialized = False
        
        self.locked_pts = None     
        self.stable_counter = 0    
        self.M_locked = None       
        
        # 用于存储 AI 建议的坐标 (row, col)
        self.ai_point = None 

    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"[Vision] Started on PC/USB. Rotation: {self.rotate_image}")

    def stop(self):
        self.running = False
        if self.thread: self.thread.join()

    def get_current_board(self):
        with self.lock: return copy.deepcopy(self.board_data)
    
    def set_ai_hint(self, move):
        """ move: (row, col) """
        self.ai_point = move

    def _loop(self):
        # === 修改：纯 USB 摄像头初始化 ===
        cap = cv2.VideoCapture(self.camera_id)
        # 尝试设置高分辨率，如果摄像头不支持会自动降级
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        
        # 检查摄像头是否成功打开
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            self.running = False
            return

        WARP_S = 800
        dst_pts = np.float32([[15, 15], [WARP_S-15, 15], [WARP_S-15, WARP_S-15], [15, WARP_S-15]])
        state = STATE_SEARCHING
        
        while self.running:
            # === 修改：纯 OpenCV 读取 ===
            ret, raw_frame = cap.read()
            if not ret: 
                time.sleep(0.01)
                continue
            
            # 畸变矫正
            if self.has_calib:
                try: frame = cv2.undistort(raw_frame, self.calib_mtx, self.calib_dist, None, self.new_cam_mtx)
                except: frame = raw_frame
            else:
                frame = raw_frame

            # 旋转处理
            if self.rotate_image == 1: frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.rotate_image == 2: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotate_image == 3: frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            debug_view = frame.copy()
            warped_view = None
            virtual_board = None

            if state == STATE_SEARCHING:
                found, pts = self._find_board_robust(frame)
                if found:
                    cv2.polylines(debug_view, [pts.astype(int)], True, (0, 255, 0), 2)
                    if self.locked_pts is not None and self._check_similarity(self.locked_pts, pts):
                        self.stable_counter += 1
                    else:
                        self.stable_counter = 0; self.locked_pts = pts 
                    cv2.putText(debug_view, f"Locking: {self.stable_counter}/20", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    if self.stable_counter > 20:
                        state = STATE_LOCKED
                        self.M_locked = cv2.getPerspectiveTransform(self.locked_pts, dst_pts)
                else:
                    self.stable_counter = 0

            elif state == STATE_LOCKED:
                cv2.polylines(debug_view, [self.locked_pts.astype(int)], True, (0, 0, 255), 3)
                warped = cv2.warpPerspective(frame, self.M_locked, (WARP_S, WARP_S))
                xs, ys = self._find_grid_morphology(warped)
                self.last_xs, self.last_ys = xs, ys
                raw_board, circles = self._scan_pieces_hough(warped, xs, ys)
                self._update_votes(raw_board)
                
                warped_view = warped.copy()
                self._draw_analysis_view(warped_view, xs, ys, raw_board, circles)
                # 绘制 AR 叠加 (包含 AI 提示)
                self._draw_overlay_ar(debug_view, xs, ys, self.M_locked)
                virtual_board = self._create_virtual_board_image()

            # 显示窗口
            display_main = cv2.resize(debug_view, (960, 720)) 
            cv2.imshow("Main View", display_main)
            if warped_view is not None: cv2.imshow("Analysis", cv2.resize(warped_view, (400, 400)))
            if virtual_board is not None: cv2.imshow("Virtual Board", virtual_board)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'): self.running = False
            if key & 0xFF == ord('r'): state = STATE_SEARCHING; self.stable_counter = 0; self.board_data[:] = 0; self.vote_matrix[:] = 0; self.ai_point = None

        cap.release()
        cv2.destroyAllWindows()

    def _draw_overlay_ar(self, img, xs, ys, M):
        _, inv_M = cv2.invert(M)
        pts_b, pts_w = [], []
        
        # 1. 绘制已识别的棋子
        with self.lock:
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if self.board_data[r, c] == BLACK: pts_b.append([xs[c], ys[r]])
                    elif self.board_data[r, c] == WHITE: pts_w.append([xs[c], ys[r]])
        
        if pts_b:
            dst = cv2.perspectiveTransform(np.array([pts_b], dtype='float32'), inv_M)[0]
            for pt in dst: cv2.circle(img, (int(pt[0]), int(pt[1])), 12, (0, 0, 255), -1)
        if pts_w:
            dst = cv2.perspectiveTransform(np.array([pts_w], dtype='float32'), inv_M)[0]
            for pt in dst: cv2.circle(img, (int(pt[0]), int(pt[1])), 12, (255, 0, 0), -1)
            
        # 2. 绘制 AI 提示点
        if self.ai_point is not None:
            ar, ac = self.ai_point
            if 0 <= ar < BOARD_SIZE and 0 <= ac < BOARD_SIZE:
                target_x = xs[ac]
                target_y = ys[ar]
                src_pt = np.array([[[target_x, target_y]]], dtype='float32')
                dst_pt = cv2.perspectiveTransform(src_pt, inv_M)[0][0]
                cx, cy = int(dst_pt[0]), int(dst_pt[1])
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), -1)
                cv2.circle(img, (cx, cy), 20, (0, 255, 0), 2)
                cv2.putText(img, "AI", (cx-10, cy-25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    def _check_similarity(self, pts1, pts2):
        dist = np.mean(np.linalg.norm(pts1 - pts2, axis=1))
        return dist < 10.0
    
    def _create_virtual_board_image(self):
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        img[:] = (130, 205, 238)
        step = 500 // (BOARD_SIZE + 1)
        for i in range(BOARD_SIZE):
            pos = step * (i + 1)
            cv2.line(img, (pos, step), (pos, 500-step), (0, 0, 0), 1)
            cv2.line(img, (step, pos), (500-step, pos), (0, 0, 0), 1)
        stars = [3, 9, 15]
        for r in stars:
            for c in stars:
                cv2.circle(img, (step*(c+1), step*(r+1)), 3, (0,0,0), -1)
        with self.lock:
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    state = self.board_data[r, c]
                    cx = step * (c + 1)
                    cy = step * (r + 1)
                    if state == BLACK: cv2.circle(img, (cx, cy), 11, (10, 10, 10), -1)
                    elif state == WHITE: cv2.circle(img, (cx, cy), 11, (240, 240, 240), -1); cv2.circle(img, (cx, cy), 11, (100, 100, 100), 1)
        return img

    def _scan_pieces_hough(self, warped, xs, ys):
        raw = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)
        blur = cv2.GaussianBlur(enhanced_gray, (9, 9), 2)
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=100, param2=HOUGH_PARAM2, minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cx, cy, r = i[0], i[1], i[2]
                c_idx = (np.abs(xs - cx)).argmin()
                r_idx = (np.abs(ys - cy)).argmin()
                grid_x, grid_y = xs[c_idx], ys[r_idx]
                if np.sqrt((cx - grid_x)**2 + (cy - grid_y)**2) > 15: continue
                mask = np.zeros_like(gray)
                cv2.circle(mask, (cx, cy), int(r*0.6), 255, -1)
                mean_val = cv2.mean(gray, mask=mask)[0]
                if mean_val < COLOR_THRESH: raw[r_idx, c_idx] = BLACK
                else: raw[r_idx, c_idx] = WHITE
        return raw, []

    def _find_grid_morphology(self, warped):
        h, w = warped.shape[:2]
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)
        v_lines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20)))
        h_lines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1)))
        xs = self._snap_to_peaks(np.sum(v_lines, axis=0), w, self.last_xs)
        ys = self._snap_to_peaks(np.sum(h_lines, axis=1), h, self.last_ys)
        self.grid_initialized = True
        return xs, ys

    def _snap_to_peaks(self, proj, length, last_val):
        theoretical = np.linspace(15, length-15, BOARD_SIZE)
        result = np.zeros(BOARD_SIZE)
        for i in range(BOARD_SIZE):
            anchor = theoretical[i]
            start, end = max(0, int(anchor - 12)), min(length, int(anchor + 12))
            region = proj[start:end]
            if np.max(region) > 500: result[i] = start + np.argmax(region)
            else: result[i] = last_val[i] if self.grid_initialized else anchor
        if self.grid_initialized: result = last_val * 0.6 + result * 0.4
        return result.astype(int)

    def _update_votes(self, raw):
        self.vote_matrix[raw == BLACK] += 2
        self.vote_matrix[raw == WHITE] -= 2
        empty = (raw == EMPTY)
        self.vote_matrix[empty & (self.vote_matrix > 0)] -= 1
        self.vote_matrix[empty & (self.vote_matrix < 0)] += 1
        np.clip(self.vote_matrix, -20, 20, out=self.vote_matrix)
        with self.lock:
            self.board_data[:] = EMPTY
            self.board_data[self.vote_matrix > 10] = BLACK
            self.board_data[self.vote_matrix < -10] = WHITE

    def _draw_analysis_view(self, img, xs, ys, raw_board, circles):
        for x in xs: cv2.line(img, (x, 0), (x, img.shape[0]), (0, 0, 100), 1)
        for y in ys: cv2.line(img, (0, y), (img.shape[1], y), (0, 0, 100), 1)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cx, cy = xs[c], ys[r]
                st = raw_board[r, c]
                if st == BLACK: cv2.circle(img, (cx, cy), 12, (0, 0, 255), 2)
                elif st == WHITE: cv2.circle(img, (cx, cy), 12, (255, 0, 0), 2)
    
    def _find_board_robust(self, img):
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = None
        max_score = -1e9
        center_x, center_y = w // 2, h // 2
        for c in cnts:
            area = cv2.contourArea(c)
            if area < (w * h * 0.1): continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                    score = area - (dist * 50)
                    if score > max_score: max_score = score; best_cnt = approx
        if best_cnt is not None: return True, self._sort_pts(best_cnt.reshape(4, 2))
        return False, None
    
    def _load_calibration(self, size):
        if os.path.exists("cam_params.xml"):
            try:
                fs = cv2.FileStorage("cam_params.xml", cv2.FILE_STORAGE_READ)
                self.calib_mtx = fs.getNode("camera_matrix").mat()
                self.calib_dist = fs.getNode("dist_coeffs").mat()
                fs.release()
                self.new_cam_mtx, self.roi = cv2.getOptimalNewCameraMatrix(self.calib_mtx, self.calib_dist, size, 0, size)
                self.has_calib = True
                print("Calibration loaded.")
            except: 
                print("Failed to load calibration.")
                pass

    def _sort_pts(self, pts):
        s = pts[np.argsort(pts[:, 1])]
        top = s[:2][np.argsort(s[:2, 0])]
        bot = s[2:][np.argsort(s[2:, 0])]
        return np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)

if __name__ == "__main__":
    # 如果你的摄像头画面是反的，修改 rotate_image=0, 1, 2, 3
    # 0: 不旋转, 1: 180度, 2: 顺时针90度, 3: 逆时针90度
    vision = GobangVision(camera_id=0, rotate_image=1)
    vision.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        vision.stop()