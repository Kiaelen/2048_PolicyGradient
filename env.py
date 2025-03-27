import numpy as np
import torch
import math

MOTIONLESS = -10
DEADGAME = -50
INVALID_THRESHOLD = 3

class Game:
    def __init__(self, instance=None, game_size=4, initial_cells=2):
        self.game_size = game_size
        if instance == None:
            self.game_board = np.zeros((game_size, game_size), dtype=int)
            self.score = 0
            self.moves = 0
            self.invalid_moves = 0
            self.end_game = False
            for _ in range(initial_cells):
                self.generate(self.game_board)
        else:
            self.game_board = instance.game_board
            self.score = instance.score
            self.moves = instance.moves
            self.invalid_moves = instance.invalid_moves
            self.end_game = instance.end_game
        
    def generate(self, board):
        '''Generate number in empty cell'''
        empty_cells = np.argwhere(board == 0)
        if len(empty_cells) == 0:
            return
        idx = np.random.choice(len(empty_cells))
        row, col = empty_cells[idx]
        board[row, col] = 2 if np.random.random() < 0.9 else 4
    
    def get_s(self):
        board = torch.log(torch.tensor(self.game_board, dtype=torch.float32) + 1)
        futile = torch.ones(board.shape) * self.invalid_moves
        return torch.stack((board, futile))
    
    def get_greedy_a(self):
        if np.random.random() < 0.5 and self.step(3, simulate=True) >= 0:
            return 3
        elif self.step(0, simulate=True) >= 0:
            return 0
        elif self.step(3, simulate=True) >= 0:
            return 3
        elif np.random.random() < 0.9 and self.step(2, simulate=True) >= 0:
            return 2
        return 1
            
    
    def print(self):
        '''Print game'''
        if self.end_game:
            print("Game over!!!")
            print(f"Your final score: {self.score}")
            print(f"Your final game board:")
        else:
            print(f"Your current score: {self.score}")
            print(f"Your current game board:")
            
        for i in range(self.game_size):
            for j in range(self.game_size):
                print(self.game_board[i, j], end="\t")
            print()
    
    def detect_movable(self, board):
        '''Detect whether game is lost'''
        # Check for empty cells
        if np.any(board == 0):
            return True
        # Check for possible row merges
        for i in range(self.game_size):
            for j in range(self.game_size - 1):
                if board[i, j] == board[i, j+1]:
                    return True
        # Check for possible column merges
        for j in range(self.game_size):
            for i in range(self.game_size - 1):
                if board[i, j] == board[i+1, j]:
                    return True
        return False
    
    def step(self, a, simulate=False): # 0 left, cw
        '''Take an action'''
        if self.end_game:
            return DEADGAME
        
        r = 0
        a_took = False # can move in this direction
        
        board = np.rot90(self.game_board.copy(), k=a)
        # Left action
        for i in range(self.game_size):
            left_index = 0 # left movable space
            zero_deteced = False # elements can move
            left_element = 0 # possible merge element
                  
            for j in range(self.game_size):
                if board[i, j] == 0: # nonempty cell
                    zero_deteced = True
                    continue
                if zero_deteced == True:
                    a_took = True
                
                if left_element == 0: # no mergable element
                    left_element = board[i, j]
                elif board[i, j] != left_element: # cannot merge
                    board[i, left_index] = left_element
                    left_index += 1
                    left_element = board[i, j]
                else: # mergable
                    a_took = True
                    board[i, left_index] = 2 * left_element
                    r += 2 * left_element
                    left_index += 1
                    left_element = 0
            if left_element != 0:
                board[i, left_index] = left_element
                left_index += 1
                
            for j in range(left_index, self.game_size): # erase the remaining numbers
                board[i, j] = 0
        
        if not a_took:
            if not simulate:
                self.invalid_moves += 1
            if self.invalid_moves >= INVALID_THRESHOLD:
                if not simulate:
                    self.end_game = True
                return DEADGAME
            return MOTIONLESS
        
        board = np.rot90(board, k=4-a)
        self.generate(board)
        
        if not simulate:
            self.invalid_moves = 0
            self.game_board = board
            self.moves += 1
        
        if not self.detect_movable(board):
            if not simulate:
                self.end_game = True
            return DEADGAME
        
        if not simulate:
            self.score += r
            
        return r
    
        