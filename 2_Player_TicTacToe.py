import numpy as np
import keyboard
from tkinter import *


game_input_keys = np.array(['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c'])

coord_top_left_corner_of_field = np.array([[10, 10], [210, 10], [410, 10],
                                          [10, 210], [210, 210], [410, 210],
                                          [10, 410], [210, 410], [410, 410]])

tkinter = Tk()
tic_tac_toe_map = Canvas(tkinter, width=605, height=605)
tkinter.title("Tic Tac Toe")
tic_tac_toe_map.create_line(0, 200, 604, 200, fill='black')
tic_tac_toe_map.create_line(0, 400, 604, 400, fill='black')
tic_tac_toe_map.create_line(200, 0, 200, 604, fill='black')
tic_tac_toe_map.create_line(400, 0, 400, 604, fill='black')
tic_tac_toe_map.pack()
tkinter.update()


class TicTacToe:
    def __init__(self, player_0_wins=False, player_1_wins=False):
        self.board = np.array([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])
        self.player_0_wins = player_0_wins
        self.player_1_wins = player_1_wins

    def game_over(self):
        for location_on_board in range(3):
            if self.board[3*location_on_board] == self.board[3*location_on_board + 1] == self.board[3*location_on_board + 2] == 'x':
                self.player_0_wins = True
                break

        for location_on_board in range(3):
            if self.board[location_on_board] == self.board[location_on_board + 3] == self.board[location_on_board + 6] == 'x':
                self.player_0_wins = True
                break

        for location_on_board in range(2):
            if self.board[2*location_on_board] == self.board[4] == self.board[-2*location_on_board + 8] == 'x':
                self.player_0_wins = True
                break

        for location_on_board in range(3):
            if self.board[3*location_on_board] == self.board[3*location_on_board + 1] == self.board[3*location_on_board + 2] == 'o':
                self.player_1_wins = True
                break

        for location_on_board in range(3):
            if self.board[location_on_board] == self.board[location_on_board + 3] == self.board[location_on_board + 6] == 'o':
                self.player_1_wins = True
                break

        for location_on_board in range(2):
            if self.board[2*location_on_board] == self.board[4] == self.board[-2*location_on_board + 8] == 'o':
                self.player_1_wins = True
                break


def draw_map(player_0_turn, field_on_board):
    if player_0_turn:
        for location_on_board in range(9):
            if field_on_board == location_on_board:
                tic_tac_toe_map.create_line(coord_top_left_corner_of_field[location_on_board][0], coord_top_left_corner_of_field[location_on_board][1] + 179, coord_top_left_corner_of_field[location_on_board][0] + 179, coord_top_left_corner_of_field[location_on_board][1], fill='black')
                tic_tac_toe_map.create_line(coord_top_left_corner_of_field[location_on_board][0], coord_top_left_corner_of_field[location_on_board][1], coord_top_left_corner_of_field[location_on_board][0] + 179, coord_top_left_corner_of_field[location_on_board][1] + 179, fill='black')

    else:
        for location_on_board in range(9):
            if field_on_board == location_on_board:
                tic_tac_toe_map.create_oval(coord_top_left_corner_of_field[location_on_board][0], coord_top_left_corner_of_field[location_on_board][1], coord_top_left_corner_of_field[location_on_board][0] + 179, coord_top_left_corner_of_field[location_on_board][1] + 179)

    tkinter.update()


def player_0():
    player_0s_turn = True
    is_end_turn = False

    while is_end_turn == False:
        for index_for_game_board in range(9):
            if(keyboard.is_pressed(game_input_keys[index_for_game_board]) and tic_tac_toe.board[index_for_game_board] == ' '):
                tic_tac_toe.board[index_for_game_board] = 'x'
                draw_map(player_0s_turn, index_for_game_board)
                is_end_turn = True

                break


def player_1():
    player_0s_turn = False
    is_end_turn = False

    while is_end_turn == False:
        for index_for_game_board in range(9):
            if(keyboard.is_pressed(game_input_keys[index_for_game_board]) and tic_tac_toe.board[index_for_game_board] == ' '):
                tic_tac_toe.board[index_for_game_board] = 'o'
                draw_map(player_0s_turn, index_for_game_board)
                is_end_turn = True

                break


if __name__ == "__main__":
    tic_tac_toe = TicTacToe()

    player_0()

    for game_round in range(0, 4):
        player_1()
        tic_tac_toe.game_over()

        if tic_tac_toe.player_1_wins:
            print('Player 1 wins!')
            break


        player_0()
        tic_tac_toe.game_over()

        if tic_tac_toe.player_0_wins:
            print('Player 0 wins!')
            break


    if tic_tac_toe.player_0_wins == tic_tac_toe.player_1_wins:
        print('Draw!')
