import numpy as np
import torch
import math

MOTIONLESS = -5
DEADGAME = -20
INVALID_THRESHOLD = 4

class Game:
    def __init__(self, game_size=4, initial_cells=2):
        self.game_size = game_size
        self.game_board = np.zeros((game_size, game_size), dtype=int)
        self.score = 0
        self.moves = 0
        self.invalid_moves = 0
        self.greedy_dir = 0
        self.end_game = False
        for i in range(initial_cells):
            self.generate()
        
    def generate(self):
        '''Generate number in empty cell'''
        empty_cells = np.argwhere(self.game_board == 0)
        if len(empty_cells) == 0:
            return
        idx = np.random.choice(len(empty_cells))
        row, col = empty_cells[idx]
        self.game_board[row, col] = 2 if np.random.random() < 0.9 else 4
    
    def get_s(self):
        return torch.tensor(self.game_board, dtype=torch.float32)
    
    def get_greedy_a(self):
        randir = self.greedy_dir
        if self.step(randir, simulate=True) >= 0:
            return randir
        elif self.step((randir+1)%4, simulate=True) >= 0:
            return (randir+1)%4
        elif self.step((randir+2)%4, simulate=True) >= 0:
            return (randir+2)%4
        return (randir+3)%4
            
    
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
    
    def detect_movable(self):
        '''Detect whether game is lost'''
        # Check for empty cells
        if np.any(self.game_board == 0):
            return True
        # Check for possible row merges
        for i in range(self.game_size):
            for j in range(self.game_size - 1):
                if self.game_board[i, j] == self.game_board[i, j+1]:
                    return True
        # Check for possible column merges
        for j in range(self.game_size):
            for i in range(self.game_size - 1):
                if self.game_board[i, j] == self.game_board[i+1, j]:
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
        
        self.invalid_moves = 0
        
        board = np.rot90(board, k=4-a)
        if not simulate:
            self.game_board = board
        self.generate()
        
        if not simulate:
            self.moves += 1
        
        if not self.detect_movable():
            if not simulate:
                self.end_game = True
            return DEADGAME
        
        if not simulate:
            self.score += r
            
        return r
    
        