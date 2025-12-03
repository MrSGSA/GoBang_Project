import cv2
import numpy as np
import random


def create_gobang_board(img_size=800, margin=50):
    """
    绘制一个标准的 15x15 五子棋棋盘
    :param img_size: 图片分辨率 (正方形)
    :param margin: 棋盘边缘留白大小
    :return: 绘制好的棋盘图像 (numpy array), 网格间距 step
    """
    # 1. 创建背景 (木纹色 RGB: 230, 200, 160 -> BGR: 160, 200, 230)
    board_color = (160, 200, 230)
    image = np.full((img_size, img_size, 3), board_color, dtype=np.uint8)

    # 计算网格间距
    # 15条线，意味着有14个间隔
    step = (img_size - 2 * margin) / 14.0

    # 2. 绘制网格线
    for i in range(15):
        # 坐标计算
        pos = int(round(margin + i * step))

        # 横线
        cv2.line(image, (margin, pos), (img_size - margin, pos), (0, 0, 0), 2)
        # 竖线
        cv2.line(image, (pos, margin), (pos, img_size - margin), (0, 0, 0), 2)

    # 3. 绘制“星位” (天元和四个角的星)
    # 标准15路棋盘星位通常在：(3,3), (11,3), (7,7), (3,11), (11,11)  <- 索引从0开始
    star_points = [(3, 3), (11, 3), (7, 7), (3, 11), (11, 11)]

    for r, c in star_points:
        cx = int(round(margin + c * step))
        cy = int(round(margin + r * step))
        cv2.circle(image, (cx, cy), 6, (0, 0, 0), -1)  # 实心小黑点

    return image, step


def draw_pieces(image, step, margin, count=20):
    """
    在棋盘上随机画一些黑白子
    """
    # 复制一份图像，不影响原图
    img_with_pieces = image.copy()

    # 棋子半径 (稍微比格子的间距一半小一点，留出空隙)
    radius = int(step / 2 * 0.9)

    # 记录已下的位置，防止重叠
    occupied = set()

    # 随机生成黑白子
    # 假设黑先，交替落子
    is_black = True

    for _ in range(count):
        while True:
            r = random.randint(0, 14)
            c = random.randint(0, 14)
            if (r, c) not in occupied:
                occupied.add((r, c))
                break

        cx = int(round(margin + c * step))
        cy = int(round(margin + r * step))

        if is_black:
            # 画黑子 (纯黑)
            # 阴影 (可选，增加立体感)
            cv2.circle(img_with_pieces, (cx + 2, cy + 2), radius, (100, 100, 100), -1)
            # 实体
            cv2.circle(img_with_pieces, (cx, cy), radius, (20, 20, 20), -1)
        else:
            # 画白子 (纯白 + 灰色边框)
            # 阴影
            cv2.circle(img_with_pieces, (cx + 2, cy + 2), radius, (100, 100, 100), -1)
            # 实体
            cv2.circle(img_with_pieces, (cx, cy), radius, (245, 245, 245), -1)
            # 边框 (让白子在浅色背景更明显)
            cv2.circle(img_with_pieces, (cx, cy), radius, (180, 180, 180), 1)

        is_black = not is_black  # 切换颜色

    return img_with_pieces


if __name__ == "__main__":
    # 1. 设置参数
    SIZE = 800  # 图片大小 800x800
    MARGIN = 50  # 边距

    # 2. 生成空棋盘
    empty_board, step = create_gobang_board(SIZE, MARGIN)
    cv2.imwrite("gobang_empty.jpg", empty_board)
    print("✅ 已生成空棋盘: gobang_empty.jpg")

    # 3. 生成带棋子的棋盘 (随机下 40 步)
    game_board = draw_pieces(empty_board, step, MARGIN, count=40)
    cv2.imwrite("gobang_game.jpg", game_board)
    print("✅ 已生成对弈棋盘: gobang_game.jpg")

    # 4. 显示一下
    cv2.imshow("Empty", cv2.resize(empty_board, (400, 400)))
    cv2.imshow("Game", cv2.resize(game_board, (400, 400)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
