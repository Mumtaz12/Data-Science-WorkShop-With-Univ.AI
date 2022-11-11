data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import random
import numpy as np
random.seed(1)
def create_board():
    board = np.zeros((3,3), dtype=int)
    return board
def place(board, player, position):
    if board[position] == 0:
        board[position] = player
        return board
def possibilities(board):
    return list(zip(*np.where(board == 0)))
def random_place(board, player):
    selections = possibilities(board)
    if len(selections) > 0:
        selection = random.choice(selections)
        place(board, player, selection)
    return board
def row_win(board, player):
    winner = False
    if np.any(np.all(board==player,axis=1)):
        return True
    else:
        return False
def col_win(board, player):
    if np.any(np.all(board==player,axis=0)):
        return True
    else:
        return False
def diag_win(board, player):
    if np.all(np.diag(board)==player) or np.all(np.diag(np.fliplr(board))==player):
        return True
    else:
        return False
def evaluate(board):
    winner = 0
    for player in [1, 2]:
        if row_win(board, player) or col_win(board, player) or diag_win(board, player):
            winner = player
    if np.all(board != 0) and winner == 0:
        winner = -1
    return winner
def play_game():
    board, winner = create_board(), 0
    while winner == 0:
        for player in [1, 2]:
            random_place(board, player)
            winner = evaluate(board)
            if winner != 0:
                break
    return winner

def play_strategic_game():
    board, winner = create_board(), 0
    board[1,1] = 1
    while winner == 0:
        for player in [2,1]:
            # use `random_place` to play a game, and store as `board`.
            # use `evaluate(board)`, and store as `winner`.
            if winner != 0:
                break
    return winner

play_strategic_game()  

def play_strategic_game():
    board, winner = create_board(), 0
    board[1,1] = 1
    while winner == 0:
        for player in [2,1]:
            random_place(board, player)
            winner = evaluate(board)
            if winner != 0:
                break
    return winner

play_strategic_game()









    test_function("time.time",
              not_called_msg = "Make sure to call `time.time`!",
              incorrect_msg = "Check your definition of `create_board` again.")
test_student_typed("plt.show",
              pattern=False,
              not_typed_msg="Did you use `plt.show`?")
test_student_typed("plt.hist",
              pattern=False,
              not_typed_msg="Did you use `plt.hist`?")              
test_student_typed("play_strategic_game()",
              pattern=False,
              not_typed_msg="Did you use `play_strategic_game()`?")
success_msg("Great work!  Yes, starting in the middle square is a large advantage when play is otherwise random.  Also, each game takes less time to play, because each victory is decided earlier.  Player 1 wins much more than Player 2, and draws are less common. This concludes this week's homework.  You can return to the course through this link:  https://courses.edx.org/courses/course-v1:HarvardX+PH526x+1T2018")