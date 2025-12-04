from detection import GobangVision
import time

if __name__ == "__main__":
    # 模拟 AI 团队的代码
    print("初始化中")
    vision = GobangVision(camera_id=0)
    vision.start()
    print("读取中")
    try:
        while True:
            # === 这里就是大模型获取坐标的地方 ===
            black_stones = vision.get_black_coordinates()
            white_stones = vision.get_white_coordinates()
            board = vision.get_current_board()
            print("\033[H\033[J", end="")
            print("-" * 30)
            print(f"黑子数量: {len(black_stones)}")
            print(f"黑子坐标: \n{black_stones}")
            print(f"白子数量: {len(white_stones)}")
            print(f"白子坐标: \n{white_stones}")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("用户终止程序")
        vision.stop()
