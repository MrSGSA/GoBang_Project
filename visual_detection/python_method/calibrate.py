import cv2
import numpy as np
import time
import os

# 尝试导入 Picamera2
try:
    from picamera2 import Picamera2
except ImportError:
    print("错误: 未找到 Picamera2 库。")
    exit(1)

# ================= 配置区域 =================
# 1. 设置棋盘格的【内部角点】数量 (列数, 行数)
# 注意：是数黑白交界的点，不是数格子！
# 例如：如果你的棋盘横向有10个方块，纵向有7个方块，那角点数通常是 (9, 6)
CHECKERBOARD = (9, 6) 

# 2. 分辨率 (必须与 detection.py 中的一致)
RESOLUTION = (1640, 1232)

# 3. 拍摄数量要求
MIN_SAMPLES = 15
# ===========================================

def run_calibration():
    # 准备对象点: (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # 用于存储所有图像的对象点和图像点
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # 初始化相机
    print("[Calibration] 初始化相机...")
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": RESOLUTION, "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    
    # 预热
    time.sleep(2)
    print(f"[Calibration] 相机已启动 ({RESOLUTION[0]}x{RESOLUTION[1]})")
    print("="*50)
    print(f"操作指南:")
    print(f"1. 将棋盘格 ({CHECKERBOARD[0]}x{CHECKERBOARD[1]}) 放入画面")
    print(f"2. 按键盘 【C】 键拍摄一张照片")
    print(f"3. 请改变角度、远近、旋转，至少拍摄 {MIN_SAMPLES} 张")
    print(f"4. 按 【Q】 键结束拍摄并开始计算")
    print("="*50)

    count = 0
    
    while True:
        # 获取图像
        frame_rgb = picam2.capture_array()
        if frame_rgb is None: continue
        
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        display = frame.copy()
        
        # 寻找角点
        # 广角镜头边缘畸变大，可能很难找，CALIB_CB_ADAPTIVE_THRESH 有助于提高成功率
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
        
        # 绘制反馈
        if ret:
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners, ret)
            msg = "Ready to Capture (Press C)"
            color = (0, 255, 0)
        else:
            msg = "Searching Checkerboard..."
            color = (0, 0, 255)

        cv2.putText(display, f"{msg} | Count: {count}/{MIN_SAMPLES}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # 缩小一点显示，防止屏幕放不下
        display_small = cv2.resize(display, (1024, 768))
        cv2.imshow("Calibration", display_small)
        
        key = cv2.waitKey(1) & 0xFF
        
        # 按 'C' 拍摄
        if key == ord('c'):
            if ret:
                # 亚像素级精确化角点位置
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                objpoints.append(objp)
                imgpoints.append(corners2)
                
                count += 1
                print(f"[Capture] 已捕获第 {count} 张图像")
                
                # 闪烁一下屏幕表示拍摄成功
                cv2.rectangle(display, (0,0), (RESOLUTION[0], RESOLUTION[1]), (255,255,255), -1)
                cv2.imshow("Calibration", cv2.resize(display, (1024, 768)))
                cv2.waitKey(100)
            else:
                print("[Error] 未检测到完整的棋盘格，无法拍摄！请调整角度。")

        # 按 'Q' 退出
        if key == ord('q'):
            if count < MIN_SAMPLES:
                print(f"[Warning] 样本数量不足 ({count}/{MIN_SAMPLES})，建议继续拍摄。按 'y' 确认退出，其他键继续。")
                if cv2.waitKey(0) & 0xFF == ord('y'):
                    break
            else:
                break

    picam2.stop()
    cv2.destroyAllWindows()

    if count > 0:
        print("\n正在计算畸变参数，这可能需要几秒钟...")
        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            
            print(f"标定完成！重投影误差 (RMS): {ret:.4f}")
            print(f"相机矩阵:\n{mtx}")
            print(f"畸变系数:\n{dist}")
            
            # 保存到 XML
            filename = "cam_params.xml"
            cv_file = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
            cv_file.write("camera_matrix", mtx)
            cv_file.write("dist_coeffs", dist)
            cv_file.write("image_width", RESOLUTION[0])
            cv_file.write("image_height", RESOLUTION[1])
            cv_file.release()
            
            print(f"\n[Success] 参数已保存至 {os.path.abspath(filename)}")
            print("现在你可以运行 detection.py 了。")
            
        except Exception as e:
            print(f"[Error] 标定计算失败: {e}")
    else:
        print("[End] 未拍摄任何照片，程序结束。")

if __name__ == "__main__":
    run_calibration()