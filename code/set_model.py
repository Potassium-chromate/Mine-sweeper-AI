# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 10:33:16 2023

@author: Eason
"""
'''
for combine_matrix: 9 is unknown area. 
                  : 0~8 is number of mines surround the rect
                  : 10 is flag
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import save_model


size = 9

def build_model():
    model = Sequential(
        [
            Conv2D(filters=128, kernel_size=(5, 5), activation='relu', padding='same', input_shape=(size, size, 1)),
            #Dropout(0.2),
            #Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(size, size, 1)),
            Conv2D(filters=128, kernel_size=(1, 1), activation='relu', padding='same', input_shape=(size, size, 1)),
            Flatten(),
            Dense(512,kernel_initializer='normal', activation='relu'),
            Dense(1024,kernel_initializer='normal', activation='relu'),
            Dropout(0.2),
            Dense(1024,kernel_initializer='normal', activation='relu'),
            Dense(512,kernel_initializer='normal', activation='relu'),
            Dropout(0.2),
            Dense(512,kernel_initializer='normal', activation='relu'),
            Dense(size**2,kernel_initializer='normal', activation='linear'),
            
        ])
    adam = Adam(learning_rate=0.0025, clipnorm=0.001, epsilon=0.001)
    model.compile(optimizer=adam,loss='mse')
    return model
#binary_crossentropy

model = build_model()
# Save the model to a specific path
save_model(model,'C:/Users/88696/minesweeper/minesweeper_model_9.h5')
print('model has been saved')
model.summary()

