# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 00:46:24 2023

@author: Eason
"""
import minesweeper_logic
import pygame
import numpy as np
import tensorflow as tf
import random
from collections import deque
import datetime
import time
import threading
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense





#load model
model = tf.keras.models.load_model('C:/Users/88696/minesweeper/minesweeper_model_9.h5')
size=9


model.summary()
layer = model.layers[0]
print(layer.get_config())


#adjust training variable

screen = pygame.display.set_mode((50*size, 50*size+120))
pygame.init()

def draw_game(board, mask, combine_matrix,train_times,win_times,lose_times,remember_times,fit_times,set_weights_times,total_step,epsilon):
    screen.fill((255, 255, 255))
    #draw rect and the number inside the rect
    for x in range(size):
        for y in range(size):
            #draw rect line
            rect = pygame.Rect(x * 50, y * 50, 50, 50)
            pygame.draw.rect(screen,(0,0,0), rect, 1)
            #draw number 0~8
            if combine_matrix[x][y] in range(0,9):
                font = pygame.font.Font(None, 36)
                text = font.render(str(board[x][y]), True, (0,0,0))
                screen.blit(text, ( y* 50 + 15, x * 50 + 15))
            elif combine_matrix[x][y] == -1:
                fill_rect = pygame.Rect(y* 50, x * 50, 50, 50)
                screen.fill((255,0,0), fill_rect)

    #train_times
    font = pygame.font.Font(None, 24)
    text = font.render("train_times: "+str(train_times), True, (255,0,0))
    screen.blit(text, (5, 50*size+10))
    #win_times
    font = pygame.font.Font(None, 24)
    text = font.render("win_times: "+str(win_times), True, (255,0,0))
    screen.blit(text, (5, 50*size+39))
    #lose_times
    font = pygame.font.Font(None, 24)
    text = font.render("lose_times: "+str(lose_times), True, (255,0,0))
    screen.blit(text, (5, 50*size+68))
    pygame.display.update()
    #total_step
    font = pygame.font.Font(None, 24)
    text = font.render("total: "+str(total_step), True, (255,0,0))
    screen.blit(text, (5, 50*size+97))
    pygame.display.update()
    #remember_times
    font = pygame.font.Font(None, 24)
    text = font.render("remember: "+str(remember_times), True, (255,0,0))
    screen.blit(text, (50*size//2+5, 50*size+10))
    #win_times
    font = pygame.font.Font(None, 24)
    text = font.render("fit: "+str(fit_times), True, (255,0,0))
    screen.blit(text, (50*size//2+5, 50*size+39))
    #lose_times
    font = pygame.font.Font(None, 24)
    text = font.render("set_weights: "+str(set_weights_times), True, (255,0,0))
    screen.blit(text, (50*size//2+5, 50*size+68))
    pygame.display.update()
    #epsilon
    font = pygame.font.Font(None, 24)
    epsi =  round(epsilon, 3)
    text = font.render("epsilon: "+str(epsi), True, (255,0,0))
    screen.blit(text, (50*size//2+5, 50*size+97))
    pygame.display.update()

# Define the state space
state_space = np.zeros((1,size*size))
for i in range(size**2):
    state_space[0][i] = 9

def check_progress(combine_matrix,x,y):
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            if x + dx >= 0 and x + dx < size and y + dy >= 0 and y + dy < size:
                if combine_matrix[x + dx][y + dy] != 9:
                    return True
    return False
                
# Define the reward function
def reward_function(board,combine_matrix,x,y):
    if combine_matrix[x][y]!=9: #click a revealed rect
        return -0.3
    elif combine_matrix[x][y]==9 and board[x][y]==-1: #lose the game
        return -1
    elif minesweeper_logic.pre_check_win(board,combine_matrix,x,y):
        return 1
    elif check_progress(combine_matrix,x,y):
        return 0.6
    else:
        return -0.2 #YOLO
def future_reward(board,mask,location):
    y,x=int(location%size),int(location//size)
    mask[x][y]=False
    f_state = [[0 for x in range(size)] for y in range(size)]
    for x in range(size):
        for y in range(size):
            if mask[x][y] ==True:
                f_state[x][y] = 9 #mask exist
            else:
                f_state[x][y] = board[x][y]
    ret =[]
    for i in range(size):
        for j in range(size):
            ret.append(reward_function(board,f_state,i,j))
    ret = np.array(ret)
    return ret.max()


class DQNAgent_keep_train:
    def __init__(self, state_space, size, model, replay_memory_size=50000, batch_size=40):
        self.state_space = state_space
        self.size = size
        self.memory_over = deque(maxlen=replay_memory_size)
        self.memory_normal = deque(maxlen=replay_memory_size)
        self.batch_size = batch_size
        self.epsilon = .04
        self.epsilon_min = 0.001
        self.gamma = 0.5
        self.epsilon_decay = 1 #make epsilon smaller and smaller
        self.model = model        #update weight much more often then target_model
        self.target_model = model #use to predict and saved with old weight
    
    def act(self, state,combine_matrix):
        if self.epsilon > random.random():
            location = random.randint(0, 80)
            return location
        else:
            q_values = self.target_model.predict(state)
            q_values = np.array(q_values).flatten()
            location = np.argmax(q_values)
            return location

            
    
    def remember_over(self, state,location,board,mask,episode_state, reward): #remember data of loss or win
        self.memory_over.append((state,location,board,mask,episode_state, reward))
        
    def remember_normal(self, state,location,board,mask,episode_state, reward): #remember data of not loss or win yet
        self.memory_normal.append((state,location,board,mask,episode_state, reward))
    
    def train(self):
        if len(self.memory_normal) < self.batch_size or len(self.memory_over) < self.batch_size:
            return
        # choose 100 sample form database to train model
        minibatch_over = random.sample(self.memory_over, int(self.batch_size*0.3))
        minibatch_normal = random.sample(self.memory_normal, int(self.batch_size*0.7))
        minibatch =  minibatch_normal +minibatch_over
        for state,location,board,mask,episode_state, reward in minibatch: #state is 3D, mask: 2D, 
            q_table = self.model.predict(state)
            q_table_cpy = np.array(q_table).reshape(size,size) #state is 1D
            state_cpy = np.array(state).reshape(size,size)
            print("Before")
            for i in range(size**2):
                if i % size == 0:
                    print("\n")
                print("%.1f"%q_table[0][i],end=",")  
            print("\nAfter")
            
            for i in range(size):
                for j in range(size):
                    if board[i][j]!=-1:
                        q_table_cpy[i][j] = reward_function(board,state_cpy,i,j) #+self.gamma*future_reward(board,mask,location)
                    else:
                        q_table_cpy[i][j] = reward_function(board,state_cpy,i,j)
            q_table = q_table_cpy.reshape(1,size**2)
            for i in range(size**2):
                if i % size == 0:
                    print("\n")
                print(q_table[0][i],end=",")  
            print("\n################")
            self.model.fit(state, q_table, epochs=3, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def save_model(self, file_path):
        self.model.save(file_path)
  

#initialize the model
agent = DQNAgent_keep_train(state_space, size, model)

#variable to be shown in pygame window
play_times=10000
train_times=0
lose_times=0
win_times=0
remember_times=0
fit_times=0
set_weights_times=0
total_step=0
for i in range(play_times):
    
    #initialize the game
    (board,mines,mask,flag)=minesweeper_logic.new_game(size)
    combine_matrix = [[9 for x in range(size)] for y in range(size)]
    running=True
    draw_game(board, mask, combine_matrix,train_times,win_times,lose_times,remember_times,fit_times,set_weights_times,total_step,agent.epsilon)
    total_reward = 0
    #print("################\nnewgame")
    
    while running:
        
        episode_state=False #False = game is not over
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    pos = pygame.mouse.get_pos()
        
        #code for ai operation         
        # Get the current state
        state_cpy = np.array(combine_matrix).flatten()
        state = np.zeros(size**2,)
        for i in range(size**2):
            state[i] = state_cpy[i]/10

        state_cpy=state_cpy.reshape(1,size,size,1)
        previous_state = state_cpy
        
        # Choose an action
        location = agent.act(state_cpy,combine_matrix) ##has been change
        #location the model click
        y,x=int(location%size),int(location//size)
        
        #set the reward before combine_matrix change
        reward=reward_function(board,combine_matrix,x,y)
        total_reward += reward
        #print("%.1f"%total_reward)
        if total_step%1000 == 0:
            print('win: ',win_times,' ','fit: ',fit_times,' ',"total: ",total_step,' ',"train_times: ",train_times)
            now = datetime.datetime.now()
            print("Current time is: ", now.strftime("%m-%d %H:%M:%S"))

     
        #the logic and rule of the game

        #minesweeper_logic.reveal_zeros(board,mask,flag, x, y)
        #minesweeper_logic.surround_zreos(board,mask,flag)
        board, mask, flag, combine_matrix = minesweeper_logic.run_the_game(board, mask, flag, x, y)


        #if win
        if minesweeper_logic.check_win(board,mask,flag):
            #(board,mines,mask,flag)=minesweeper_logic.new_game(size)
            train_times+=1
            win_times+=1
            running=False
            total_step+=1
            remember_times+=1
            episode_state = not episode_state
            agent.remember_over(previous_state,location,board,mask,episode_state, reward) #if game is over, it must be documented
           

        #if lose   
        elif minesweeper_logic.check_lose(board,mask,flag,combine_matrix):
            #(board,mines,mask,flag)=minesweeper_logic.new_game(size)
            train_times+=1
            lose_times+=1
            running=False  
            total_step+=1
            episode_state = not episode_state
            agent.remember_over(previous_state,location,board,mask,episode_state, reward) #if game is over, it must be documented
            remember_times+=1
            
            
        #remember 50% probability     
        if np.random.rand()*10 <= 5: 
            if episode_state==False: #game is not over
                agent.remember_normal(previous_state,location,board,mask,episode_state, reward)
                remember_times+=1
        #train 0.1% probability
        if np.random.rand()*1000 <= 8: 
            agent.train()
            fit_times+=1
        #learn 0.05% probability
        if np.random.rand()*1000 <= 1: 
            agent.learn()   
            set_weights_times+=1
        total_step+=1
        
        '''
        if total_step>100:
            running=False
        ''' 
        
        draw_game(board, mask, combine_matrix,train_times,win_times,lose_times,remember_times,fit_times,set_weights_times,total_step,agent.epsilon)
        model.save("C:/Users/88696/minesweeper/minesweeper_model_9.h5")
        
        #time.sleep(.5)
  #Quit game
pygame.quit()
agent.learn()
model.save("C:/Users/88696/minesweeper/minesweeper_model_9.h5")