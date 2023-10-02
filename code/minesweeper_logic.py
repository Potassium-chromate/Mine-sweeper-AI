# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 16:10:19 2023

@author: Eason
"""
'''
for combine_matrix: 9 is unknown area. 
                  : 0~8 is number of mines surround the rect
                  : 10 is flag
'''
import numpy as np
# auto
def reveal_zeros(board, mask,flag, x, y):
    size=len(board)
    if x < 0 or x >= size or y < 0 or y >= size:
        return
    if mask[x][y]==False:
        return
    if board[x][y] != 0:
        return
  
    mask[x][y] = False
    flag[x][y] = False
    reveal_zeros(board, mask, flag, x - 1, y)
    reveal_zeros(board, mask, flag, x + 1, y)
    reveal_zeros(board, mask, flag, x, y - 1)
    reveal_zeros(board, mask, flag, x, y + 1)
   

def surround_zreos(board,mask,flag):
    size=len(board)
    for x in range(size):
        for y in range(size):
                if (board[x][y]==0)and(mask[x][y]==False):
                    if x>0:
                        mask[x-1][y]=False
                        flag[x-1][y] = False
                    if x<size-1:
                        mask[x+1][y]=False
                        flag[x+1][y] = False
                    if y>0:
                        mask[x][y-1]=False
                        flag[x][y-1] = False
                    if y<size-1:
                        mask[x][y+1]=False
                        flag[x][y+1] = False
            

def new_game(size):
    
    import random
    
    board = [[0 for x in range(size)] for y in range(size)]
    mines = [(random.randint(0, size-1), random.randint(0, size-1)) for i in range(int(size**2/100*15))]
    for mine in mines:
        x, y = mine
        board[x][y] = -1
        
    # Count number of mines around each cell
    for x in range(size):
        for y in range(size):
            if board[x][y] == -1:
                continue
            count = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    if x + dx >= 0 and x + dx < size and y + dy >= 0 and y + dy < size:
                        if board[x + dx][y + dy] == -1:
                            count += 1
            board[x][y] = count
            

    mask = [[True for i in range(size)] for j in range(size)]
    flag = [[False for i in range(size)] for j in range(size)]
    
    return(board,mines,mask,flag)

def check_win(board,mask,flag):
    size=len(board)
    board_cpy = np.array(board)
    mask_cpy = np.array(mask)
    index = np.argwhere(board_cpy.flatten() != -1)
    count = 0
    for i in index.flatten():
        if mask_cpy.flatten()[i] == True:
            count += 1
    if count == 0:  
        return True  
    for x in range(size):
        for y in range(size):
            if (flag[x][y]==False) and (board[x][y]==-1): #hasn't toggle all mine
                return False        
            elif (flag[x][y]==True) and (board[x][y]!=-1):#toggle on wrong site
                return False
    return True 

def pre_check_win(board,combine_matrix_,x,y):
    size=len(board)
    if board[x][y] == -1:
        return False
    for i in range(size):
        for j in range(size):
            if i==x and j== y:
                continue
            if combine_matrix_[i][j]==9 and board[i][j]!=-1: #hasn't toggle all mine
                return False
    return True

def check_lose(board,mask,flag,combine_matrix_):
    size=len(board)
    for x in range(size):
        for y in range(size):
            if mask[x][y]==False and board[x][y]==-1 and flag[x][y]==False:
                return True
    
    combine_matrix_cpy = np.array(combine_matrix_)
 
    if not np.any(combine_matrix_cpy == 9):
        for x in range(size):
            for y in range(size):
                if flag[x][y]==True and board[x][y]!=-1 :
                    return True
    return False
    
def combine_matrix(board, mask, flag):
    size = len(board)
    combined = [[0 for x in range(size)] for y in range(size)]
    for x in range(size):
        for y in range(size):
            if flag[x][y] :
                combined[x][y] = 10 #flag exist but no mask exist
            elif mask[x][y] and flag[x][y]==False:
                combined[x][y] = 9 #mask exist
            else:
                combined[x][y] = board[x][y]
    return combined
    

def run_the_game(board, mask, flag, x, y):
    if mask[x][y]==True and flag[x][y]==False:
        if board[x][y]!=-1:
            mask[x][y]=False
        elif board[x][y]==0:
            reveal_zeros(board, mask, x, y)
            surround_zreos(board,mask)
        else:
            mask[x][y]=False
    
    return board, mask, flag, combine_matrix(board, mask, flag) # Return the combined matrix

def toggle_flag(board, mask, flag,combine_matrix_, x, y):
    if combine_matrix_[x][y]!=9:
        return board, mask, flag,combine_matrix_
    else:
        flag[x][y] = True
        return board, mask, flag, combine_matrix(board, mask, flag)
