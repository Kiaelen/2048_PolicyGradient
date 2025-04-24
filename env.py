import numpy as np
import torch
import math

MOTIONLESS = -100
DEADGAME = -1000
INVALID_THRESHOLD = 3
NUM_CHANNEL = 5

RAND_NUM = [0]
RAND_DIS = []
LOG_MAX = 12

for i in range(LOG_MAX+1):
    RAND_NUM.append(2 ** i)
for i in range(LOG_MAX+1):
    RAND_DIS.append(2 ** -(i+1))
RAND_DIS.append(2 ** -(LOG_MAX+1))

def old_get_s(self):
    board = torch.log(torch.tensor(self.game_board, dtype=torch.float32) + 1)
    futile = torch.ones(board.shape) * self.invalid_moves
    max_mask1 = torch.tensor(self.game_board >= (self.game_board.max()), dtype=torch.float32)
    max_mask2 = torch.tensor(self.game_board >= (self.game_board.max()/4), dtype=torch.float32)
    max_mask3 = torch.tensor(self.game_board >= (self.game_board.max()/16), dtype=torch.float32)
    return torch.stack((board, futile, max_mask1, max_mask2, max_mask3))

def new_get_s(self):
    t_board = torch.tensor(self.game_board, dtype=torch.float32)
        
    board = torch.log(t_board + 1)
    futile = torch.ones(board.shape) * self.invalid_moves
    
    max_mask1 = torch.tensor(self.game_board >= (self.game_board.max()), dtype=torch.float32)
    max_mask2 = torch.tensor(self.game_board >= (self.game_board.max()/4), dtype=torch.float32)
    max_mask3 = torch.tensor(self.game_board >= (self.game_board.max()/16), dtype=torch.float32)
    
    c_copy = torch.zeros(board.shape)
    c_copy[:, :self.game_size-1] = t_board[:, 1:]
    c_diff = t_board - c_copy
    
    r_copy = torch.zeros(board.shape)
    r_copy[:self.game_size-1, :] = t_board[1:, :]
    r_diff = t_board - r_copy
    
    return torch.stack((board, futile, c_diff, r_diff, max_mask1, max_mask2, max_mask3))

class Game:
    def __init__(self, instance=None, rand=False, game_size=4, initial_cells=2):
        self.game_size = game_size
        if rand:
            while True:
                self.game_board = np.random.choice(RAND_NUM, size=(game_size, game_size), p=RAND_DIS)
                if self.detect_movable(self.game_board):
                    break

            self.score = 0
            self.moves = 0
            self.invalid_moves = 0
            self.end_game = False
        elif instance != None:
            self.game_board = instance.game_board.copy()
            self.score = instance.score
            self.moves = instance.moves
            self.invalid_moves = instance.invalid_moves
            self.end_game = instance.end_game
        else:
            self.game_board = np.zeros((game_size, game_size), dtype=int)
            self.score = 0
            self.moves = 0
            self.invalid_moves = 0
            self.end_game = False
            for _ in range(initial_cells):
                self.generate(self.game_board)
        
    def generate(self, board):
        '''Generate number in empty cell'''
        empty_cells = np.argwhere(board == 0)
        if len(empty_cells) == 0:
            return
        idx = np.random.choice(len(empty_cells))
        row, col = empty_cells[idx]
        board[row, col] = 2 if np.random.random() < 0.9 else 4
    
    def get_s(self, old_ver=False):
        if old_ver:
            return old_get_s(self)
        return old_get_s(self)
        
    
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
    
    def get_complementary_score(self, board, empty_score=10):
        score = 0
        for i in range(self.game_size):
            for j in range(self.game_size):
                score += (board[i, j] == 0) * empty_score
                score += board[i, j] * (abs(self.game_size / 2 - 0.5 - i) + abs(self.game_size / 2 - 0.5 - j))
        return score
    
    def step(self, a, simulate=False, cplm_para=1): # 0 left, cw
        '''Take an action'''
        if self.end_game:
            return DEADGAME
        
        r = 0
        # ini_cplm = self.get_complementary_score(self.game_board)
        a_took = False # can move in this direction
        
        board = np.rot90(self.game_board.copy(), k=a)
        # Left action
        for i in range(self.game_size):
            left_index = 0 # left movable space
            zero_deteced = False # elements can move
            left_element = 0 # possible merge element
                  
            for j in range(self.game_size):
                if board[i, j] == 0: # empty cell
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
        
        # aft_cplm = self.get_complementary_score(board)
        r_cplm = 0
        
        return r + cplm_para * r_cplm
    
        