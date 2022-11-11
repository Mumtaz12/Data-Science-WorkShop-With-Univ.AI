---
title: 'Homework 2'
description: 'Exercises for Homework (Week 2).  Tic-tac-toe (or noughts and crosses) is a simple strategy game in which two players take turns placing a mark on a 3x3 board, attempting to make a row, column, or diagonal of three with their mark.  In this homework, we will use the tools we''ve covered in the past two weeks to create a tic-tac-toe simulator and evaluate basic winning strategies.'
---

## Exercise 1

```yaml
type: NormalExercise
key: cfd2bb78d3
lang: python
xp: 100
skills: 2
```

Tic-tac-toe (or noughts and crosses) is a simple strategy game in which two players take turns placing a mark on a 3x3 board, attempting to make a row, column, or diagonal of three with their mark.  In this homework, we will use the tools we've covered in the past two weeks to create a tic-tac-toe simulator and evaluate basic winning strategies.

In the following 13 exercises, we will learn to create a tic-tac-toe board, place markers on the board, evaluate if either player has won, and use this to simulate two basic strategies.

`@instructions`
- For our tic-tac-toe board, we will use a numpy array with dimension 3 by 3.  Make a function `create_board()` that creates such a board, with values of integers `0`.
- Call `create_board()`, and store this as `board`.

`@hint`
- The function`zeros` in the `numpy` library could do the trick!

`@pre_exercise_code`
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import random
import numpy as np

```

`@sample_code`
```{python}
# write your code here!







```

`@solution`
```{python}
import random
import numpy as np

def create_board():
    board = np.zeros((3,3), dtype=int)
    return board

board = create_board() 
```

`@sct`
```{python}
test_function("create_board",
              not_called_msg = "Make sure to call `create_board`!",
              incorrect_msg = "Check your definition of `create_board` again.")
test_object("board",
            undefined_msg = "Did you define `board`?",
            incorrect_msg = "It looks like `board` wasn't defined correctly.")
success_msg("Great work!")
```

---

## Exercise 2

```yaml
type: NormalExercise
key: 12006fdb5a
lang: python
xp: 100
skills: 2
```

Tic-tac-toe (or noughts and crosses) is a simple strategy game in which two players take turns placing a mark on a 3x3 board, attempting to make a row, column, or diagonal of three with their mark.  In this homework, we will use the tools we've covered in the past two weeks to create a tic-tac-toe simulator and evaluate basic winning strategies.

Players 1 and 2 will take turns changing values of this array from a 0 to a 1 or 2, indicating the number of the player who places there.

`@instructions`
- Create a function `place(board, player, position)`, where:
    - `player` is the current player (an integer 1 or 2)
    - `position` a tuple of length 2 specifying a desired location to place their marker.
    - Your function should only allow the current player to place a marker on the board (change the board position to their number) if that position is empty (zero).
- Use `create_board()` to store a board as `board`, and use `place` to have Player 1 place a marker on location `(0, 0)`.

`@hint`
- Because `board` is a `numpy.array` object, you can assign a value to its positions using bracket indices, just like a dictionary.
- Keep in mind that the positions in this array are tuples!  For example, position `(0, 0)` can be reassigned using `board[(0,0)] == 0`.

`@pre_exercise_code`
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import random
import numpy as np
def create_board():
    board = np.zeros((3,3), dtype=int)
    return board
```

`@sample_code`
```{python}
# write your code here!







```

`@solution`
```{python}
def place(board, player, position):
    if board[position] == 0:
        board[position] = player
        return board

board = create_board()
place(board, 1, (0, 0))
```

`@sct`
```{python}
test_function("place",
              not_called_msg = "Make sure to call `place`!",
              incorrect_msg = "Check your definition of `place` again.")
test_object("board",
            undefined_msg = "Did you define `board`?",
            incorrect_msg = "It looks like `board` wasn't defined correctly.")
success_msg("Great work!")
```

---

## Exercise 3

```yaml
type: NormalExercise
key: a336ef36ee
lang: python
xp: 100
skills: 2
```

Tic-tac-toe (or noughts and crosses) is a simple strategy game in which two players take turns placing a mark on a 3x3 board, attempting to make a row, column, or diagonal of three with their mark.  In this homework, we will use the tools we've covered in the past two weeks to create a tic-tac-toe simulator and evaluate basic winning strategies.

In this exercise, we will determine which positions are available to either player for placing their marker.

`@instructions`
- Create a function `possibilities(board)` that returns a list of all positions (`tuples`) on the board that are not occupied (`0`).  (Hint: `numpy.where` is a handy function that returns a list of indices that meet a condition.)
- `board` is already defined from previous exercises.  Call `possibilities(board)` to see what it returns!

`@hint`
- Try using `numpy.where(board == 0)`.
- You can also use a `for` loop for each position in the array, and check if `board[position] == 0`.

`@pre_exercise_code`
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import random
import numpy as np
def create_board():
    board = np.zeros((3,3), dtype=int)
    return board
board = create_board()
def place(board, player, position):
    if board[position] == 0:
        board[position] = player
        return board
place(board, 1, (0, 0))
```

`@sample_code`
```{python}
# write your code here!





```

`@solution`
```{python}
def possibilities(board):
    return list(zip(*np.where(board == 0)))

possibilities(board)
```

`@sct`
```{python}
test_function("possibilities",
              not_called_msg = "Make sure to call `possibilities`!",
              incorrect_msg = "Check your definition of `possibilities` again.")
test_student_typed("==",
              pattern=False,
              not_typed_msg="Do you test which board positions are equal to 0?")              
success_msg("Great work!")
```

---

## Exercise 4

```yaml
type: NormalExercise
key: e511ea8d2b
lang: python
xp: 100
skills: 2
```

Tic-tac-toe (or noughts and crosses) is a simple strategy game in which two players take turns placing a mark on a 3x3 board, attempting to make a row, column, or diagonal of three with their mark.  In this homework, we will use the tools we've covered in the past two weeks to create a tic-tac-toe simulator and evaluate basic winning strategies.

The next step is for the current player to place a marker among the available positions. In this exercise, we will select an available board position at random and place a marker there.

`@instructions`
- Write a function `random_place(board, player)` that places a marker for the current player at random among all the available positions (those currently set to 0).
	- Find possible placements with `possibilities(board)`.
	- Select one possible placement at random using `random.choice(selection)`.
- `board` is already defined from previous exercises.  Call `random_place(board, player)` to place a random marker for Player 2, and store this as `board` to update its value.

`@hint`
- Note that the `choice` function in the `random` library takes a single parameter, an iterable. `random.choice()` will randomly select one item from that iterable.

`@pre_exercise_code`
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import random
import numpy as np
random.seed(1)
def create_board():
    board = np.zeros((3,3), dtype=int)
    return board
board = create_board()
def place(board, player, position):
    if board[position] == 0:
        board[position] = player
        return board
place(board, 1, (0, 0))
def possibilities(board):
    return list(zip(*np.where(board == 0)))
```

`@sample_code`
```{python}
# write your code here!







```

`@solution`
```{python}
def random_place(board, player):
    selections = possibilities(board)
    if len(selections) > 0:
        selection = random.choice(selections)
        place(board, player, selection)
    return board

random_place(board, 2)
```

`@sct`
```{python}
test_function("random_place",
              not_called_msg = "Make sure to call `random_place`!",
              incorrect_msg = "Check your definition of `random_place` again.")
test_object("board",
            undefined_msg = "Did you define `board`?",
            incorrect_msg = "It looks like `board` wasn't defined correctly.")
success_msg("Great work!")
```

---

## Exercise 5

```yaml
type: NormalExercise
key: 436b7ed3e4
lang: python
xp: 100
skills: 2
```

Tic-tac-toe (or noughts and crosses) is a simple strategy game in which two players take turns placing a mark on a 3x3 board, attempting to make a row, column, or diagonal of three with their mark.  In this homework, we will use the tools we've covered in the past two weeks to create a tic-tac-toe simulator and evaluate basic winning strategies.

We will now have both players place three markers each.

`@instructions`
- `board` is already given.  Call `random_place(board, player)` to place three pieces each on `board` for players 1 and 2.
- Print `board` to see your result.

`@hint`
- Can you use `for` loops to alternate a `random_place` for each player three times?

`@pre_exercise_code`
```{python}
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
```

`@sample_code`
```{python}
board = create_board()





```

`@solution`
```{python}
board = create_board()
for i in range(3):
    for player in [1, 2]:
        random_place(board, player)

print(board)

```

`@sct`
```{python}
test_function("print",
              not_called_msg = "Make sure to call `print`!",
              incorrect_msg = "Check your definition of `print` again.")
test_object("board",
            undefined_msg = "Did you define `board`?",
            incorrect_msg = "It looks like `board` wasn't defined correctly.")
success_msg("Great work!")
```

---

## Exercise 6

```yaml
type: NormalExercise
key: 2d47bf75c5
lang: python
xp: 100
skills: 2
```

Tic-tac-toe (or noughts and crosses) is a simple strategy game in which two players take turns placing a mark on a 3x3 board, attempting to make a row, column, or diagonal of three with their mark.  In this homework, we will use the tools we've covered in the past two weeks to create a tic-tac-toe simulator and evaluate basic winning strategies.

In the next few exercises, we will make functions that verify if either player has won the game.

`@instructions`
- Make a function `row_win(board, player)` that takes the player (integer), and determines if any row consists of only their marker.  Have it return `True` of this condition is met, and `False` otherwise.
- `board` is already defined from previous exercises.  Call `row_win` to check if Player 1 has a complete row.

`@hint`
- Using `if` and `else` would work elegantly.
- `numpy` contains function `any` and `all`, which might be helpful For more information, use `help()`!

`@pre_exercise_code`
```{python}
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
board = create_board()
for i in range(3):
    for player in [1, 2]:
        random_place(board, player)
```

`@sample_code`
```{python}
# write your code here!







```

`@solution`
```{python}
def row_win(board, player):
    if np.any(np.all(board==player,axis=1)): # this checks if any row contains all positions equal to player.
        return True
    else:
        return False

row_win(board, 1)
```

`@sct`
```{python}
test_function("row_win",
              not_called_msg = "Make sure to call `row_win`!",
              incorrect_msg = "Check your definition of `row_win` again.")
success_msg("Great work!")
```

---

## Exercise 7

```yaml
type: NormalExercise
key: 1b692c47fe
lang: python
xp: 100
skills: 2
```

Tic-tac-toe (or noughts and crosses) is a simple strategy game in which two players take turns placing a mark on a 3x3 board, attempting to make a row, column, or diagonal of three with their mark.  In this homework, we will use the tools we've covered in the past two weeks to create a tic-tac-toe simulator and evaluate basic winning strategies.

In the next few exercises, we will make functions that verify if either player has won the game.

`@instructions`
- Create a similar function `col_win(board, player)` that takes the player (integer), and determines if any column consists of only their marker.  Have it return `True` if this condition is met, and `False` otherwise.
- `board` is already defined from previous exercises.  Call `col_win` to check if Player 1 has a complete column.

`@hint`
- This exercise should be nearly identical to `row_win`!

`@pre_exercise_code`
```{python}
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
board = create_board()
for i in range(3):
    for player in [1, 2]:
        random_place(board, player)
```

`@sample_code`
```{python}
# write your code here!







```

`@solution`
```{python}
def col_win(board, player):
    if np.any(np.all(board==player,axis=0)):
        return True
    else:
        return False

col_win(board, 1)
```

`@sct`
```{python}
test_function("col_win",
              not_called_msg = "Make sure to call `col_win`!",
              incorrect_msg = "Check your definition of `col_win` again.")
success_msg("Great work!")
```

---

## Exercise 8

```yaml
type: NormalExercise
key: c059adbd6b
lang: python
xp: 100
skills: 2
```

Tic-tac-toe (or noughts and crosses) is a simple strategy game in which two players take turns placing a mark on a 3x3 board, attempting to make a row, column, or diagonal of three with their mark.  In this homework, we will use the tools we've covered in the past two weeks to create a tic-tac-toe simulator and evaluate basic winning strategies.

In the next few exercises, we will make functions that verify if either player has won the game.

`@instructions`
- Finally, create a function `diag_win(board, player)` that tests if either diagonal of the board consists of only their marker. Have it return `True` if this condition is met, and `False` otherwise.
- `board` is already defined from previous exercises.  Call `diag_win` to check if Player 1 has a complete diagonal.

`@hint`
-  This should be very similar to the previous two exercises.  However, in this case, there are only two diagonals to check!

`@pre_exercise_code`
```{python}
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
board = create_board()
for i in range(3):
    for player in [1, 2]:
        random_place(board, player)
```

`@sample_code`
```{python}
# write your code here!







```

`@solution`
```{python}
def diag_win(board, player):
    if np.all(np.diag(board)==player) or np.all(np.diag(np.fliplr(board))==player):
        # np.diag returns the diagonal of the array
        # np.fliplr rearranges columns in reverse order
        return True
    else:
        return False

diag_win(board, 1)
```

`@sct`
```{python}
test_function("diag_win",
              not_called_msg = "Make sure to call `diag_win`!",
              incorrect_msg = "Check your definition of `diag_win` again.")
success_msg("Great work!")
```

---

## Exercise 9

```yaml
type: NormalExercise
key: 5fc2ecbc43
lang: python
xp: 100
skills: 2
```

Tic-tac-toe (or noughts and crosses) is a simple strategy game in which two players take turns placing a mark on a 3x3 board, attempting to make a row, column, or diagonal of three with their mark.  In this homework, we will use the tools we've covered in the past two weeks to create a tic-tac-toe simulator and evaluate basic winning strategies.

`@instructions`
- Create a function `evaluate(board)` that uses `row_win`, `col_win`, and `diag_win` functions for both players.  If one of them has won, return that player's number.  If the board is full but no one has won, return `-1`.  Otherwise, return `0`.
- `board` is already defined from previous exercises.  Call `evaluate` to see if either player has won the game yet.

`@hint`
- This function will require two parts.  First, checking to see if either player meets the winning condition.  Second, check if any possibilities to place pieces remain for either player!

`@pre_exercise_code`
```{python}
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
board = create_board()
for i in range(3):
    for player in [1, 2]:
        random_place(board, player)
```

`@sample_code`
```{python}
def evaluate(board):
    winner = 0
    for player in [1, 2]:
        # Check if `row_win`, `col_win`, or `diag_win` apply. 
		# If so, store `player` as `winner`.
    if np.all(board != 0) and winner == 0:
        winner = -1
    return winner

# add your code here.


```

`@solution`
```{python}
def evaluate(board):
    winner = 0
    for player in [1, 2]:
        if row_win(board, player) or col_win(board, player) or diag_win(board, player):
            winner = player
    if np.all(board != 0) and winner == 0:
        winner = -1
    return winner

evaluate(board)


```

`@sct`
```{python}
test_function("evaluate",
              not_called_msg = "Make sure to call `evaluate`!",
              incorrect_msg = "Check your definition of `evaluate` again.")
success_msg("Great work!")
```

---

## Exercise 10

```yaml
type: NormalExercise
key: c55bf90a13
lang: python
xp: 100
skills: 2
```

Tic-tac-toe (or noughts and crosses) is a simple strategy game in which two players take turns placing a mark on a 3x3 board, attempting to make a row, column, or diagonal of three with their mark.  In this homework, we will use the tools we've covered in the past two weeks to create a tic-tac-toe simulator and evaluate basic winning strategies.

In this exercise, we will use all the functions we have made to simulate an entire game.

`@instructions`
- `create_board()`, `random_place(board, player)`, and `evaluate(board)` have been created from previous exercises.  Create a function `play_game()` that:
	- Creates a board.
	- Alternates taking turns between two players (beginning with Player 1), placing a marker during each turn.
	- Evaluates the board for a winner after each placement.
	- Continues the game until one player wins (returning `1` or `2` to reflect the winning player), or the game is a draw (returning `-1`).
- Call `play_game` once.

`@hint`
- Use a `while` loop to check if anyone has won, or the game is a draw.  While no one has, keep alternating between Players 1 and 2 using a `for` loop and `random_place`!
- Once either player has won or the game is a draw, you could use `break` to quit the `while` loop.

`@pre_exercise_code`
```{python}
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
```

`@sample_code`
```{python}
# write your code here!







```

`@solution`
```{python}
def play_game():
    board = create_board()
    winner = 0
    while winner == 0:
        for player in [1, 2]:
            random_place(board, player)
            winner = evaluate(board)
            if winner != 0:
                break
    return winner

play_game()
```

`@sct`
```{python}
test_function("play_game",
              not_called_msg = "Make sure to call `play_game`!",
              incorrect_msg = "Check your definition of `play_game` again.")
success_msg("Great work!")
```

---

## Exercise 11

```yaml
type: NormalExercise
key: a22dfc2ad8
lang: python
xp: 100
skills: 2
```

Tic-tac-toe (or noughts and crosses) is a simple strategy game in which two players take turns placing a mark on a 3x3 board, attempting to make a row, column, or diagonal of three with their mark.  In this homework, we will use the tools we've covered in the past two weeks to create a tic-tac-toe simulator and evaluate basic winning strategies.

We will now play several games, and visualize how often either player wins, or how often the game is a draw.

`@instructions`
- Use the `play_game()` function to play 1,000 random games, where Player 1 always goes first.
- Import and use the `time` library to call the `time()` function both before and after playing all 1,000 games.
	- Store these times as `start` and `stop`, respectively.
	- Subtract them to evaluate how long it takes to play 1,000 games, and print your answer.
- The library `matplotlib.pyplot` has already been stored as `plt`.  Use `plt.hist()` and `plt.show()` to plot a histogram of the results.
	- Does Player 1 win more than Player 2?
	- Does either player win more than each player draws?

`@hint`
- You can call and store the results of `play_game` very quickly using a list comprehension!
- You can subtract two `time.time` objects to obtain the number of seconds between them.

`@pre_exercise_code`
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import random
import numpy as np
import matplotlib.pyplot as plt
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
```

`@sample_code`
```{python}
# write your code here!







```

`@solution`
```{python}
import time
start = time.time()
games = [play_game() for i in range(1000)]
stop = time.time()
print(stop - start)
plt.hist(games)
plt.show()
```

`@sct`
```{python}
test_function("time.time",
              not_called_msg = "Make sure to call `time.time`!",
              incorrect_msg = "Check your definition of `create_board` again.")
test_student_typed("print",
              pattern=False,
              not_typed_msg="Did you make sure to print the time difference?")
test_student_typed("plt.show",
              pattern=False,
              not_typed_msg="Did you use `plt.show`?")
test_student_typed("plt.hist",
              pattern=False,
              not_typed_msg="Did you use `plt.hist`?")              
test_student_typed("play_game()",
              pattern=False,
              not_typed_msg="Did you use `play_game()`?")              
success_msg("Great work!  We see that Player 1 wins more than Player 2, and the game sometimes ends in draws.  The total amount of time taken is about a few seconds, but will vary from machine to machine.")
```

---

## Exercise 12

```yaml
type: NormalExercise
key: b02cc12320
lang: python
xp: 100
skills: 2
```

Tic-tac-toe (or noughts and crosses) is a simple strategy game in which two players take turns placing a mark on a 3x3 board, attempting to make a row, column, or diagonal of three with their mark.  In this homework, we will use the tools we've covered in the past two weeks to create a tic-tac-toe simulator and evaluate basic winning strategies.

In the previous exercise, we see that when guessing at random, it's better to go first, as expected.  Let's see if Player 1 can improve their strategy.

`@instructions`
- `create_board()`, `random_place(board, player)`, and `evaluate(board)` have been created from previous exercises.  Create a function `play_strategic_game()`, where Player 1 always starts with the middle square, and otherwise both players place their markers randomly.
- Call `play_strategic_game` once.

`@hint`
- First assign the middle position to Player 1 directly, then alternate between Players 2 and 1 using `for` loops to place randomly!

`@pre_exercise_code`
```{python}
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
```

`@sample_code`
```{python}
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


```

`@solution`
```{python}
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


```

`@sct`
```{python}
test_function("play_strategic_game",
              not_called_msg = "Make sure to call `play_strategic_game`!",
              incorrect_msg = "Check your definition of `play_strategic_game` again.")
success_msg("Great work!")
```

---

## Exercise 13

```yaml
type: NormalExercise
key: d99f988283
lang: python
xp: 100
skills: 2
```

Tic-tac-toe (or noughts and crosses) is a simple strategy game in which two players take turns placing a mark on a 3x3 board, attempting to make a row, column, or diagonal of three with their mark.  In this homework, we will use the tools we've covered in the past two weeks to create a tic-tac-toe simulator and evaluate basic winning strategies.

`@instructions`
- The results from Exercise 12 have been stored.  Use the `play_strategic_game()` function to play 1,000 random games.
- Use the `time` libary to evaluate how long all these games takes.
- The library `matplotlib.pyplot` has already been stored as `plt`.  Use `plt.hist` and `plt.show` to plot your results.  Did Player 1's performance improve?  Does either player win more than each player draws?

`@hint`
-  You can again use a list comprehension to repeatedly call and store `play_strategic_game`.

`@pre_exercise_code`
```{python}
data_filepath = "https://s3.amazonaws.com/assets.datacamp.com/production/course_974/datasets/"
import random
import numpy as np
import matplotlib.pyplot as plt
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
            random_place(board, player)
            winner = evaluate(board)
            if winner != 0:
                break
    return winner
```

`@sample_code`
```{python}
# write your code here!







```

`@solution`
```{python}
import time
start = time.time()
games = [play_strategic_game() for i in range(1000)]
stop = time.time()
print(stop - start)
plt.hist(games)
plt.show()


```

`@sct`
```{python}
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
```
