import random
import numpy as np
import gradio as gd

BOARD_SIZE = 15
board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype = int )
current_player = 1 #黑色为1，白色为0

def board_init():
    global board,current_player
    board = ((BOARD_SIZE, BOARD_SIZE), dtype=int)
    current_player = 1
    return draw_board()

