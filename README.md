# Mine-sweeper-AI
This project contains an implementation of Minesweeper game logic and an AI that can play the game. The AI reveals tiles and places flags on the board, aiming to reveal all non-mine tiles without triggering any mines. The project utilizes a graphical interface built with Pygame and a machine learning model built with TensorFlow and Keras.

Table of Contents
- [Dependencies](#Dependencies)
- [Graphical Interface](#Graphical_Interface)
- [Machine Learning Model](#Machine_Learning_Model)
- [Training](#Training)
- [Code Structure](#)
- License

# Dependencies
This project requires the following libraries:

- Pygame
- TensorFlow
- Keras
- Numpy

# Graphical_Interface
The graphical interface is built using Pygame. It displays the Minesweeper board, game statistics, and training variables. The interface updates in real-time as the AI plays the game.

# Machine_Learning_Model
The AI utilizes a Deep Q-Network (DQN) implemented with TensorFlow and Keras. The model is trained continuously as the AI plays the game, learning to make better decisions over time.

# Training
The model is trained using a combination of experiences from winning, losing, and ongoing games. The training process involves adjusting the Q-values of the chosen actions based on the received rewards and the maximum Q-value of the next state, following the Q-learning update rule.

# Code Structure
- `main.py`: The main file to run the Minesweeper AI.
- `minesweeper_logic.py`: Contains the logic for the Minesweeper game, including functions to reveal tiles,       place flags, and check win/lose conditions.
- `minesweeper_training.py`: Contains the logic for training the AI, including the DQN implementation and the     training loop.

# License
This project is licensed under the MIT License - see the LICENSE file for details.
