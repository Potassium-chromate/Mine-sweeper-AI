# Mine-sweeper-AI
This project contains an implementation of Minesweeper game logic and an AI that can play the game. The AI reveals tiles and places flags on the board, aiming to reveal all non-mine tiles without triggering any mines. The project utilizes a graphical interface built with Pygame and a machine learning model built with TensorFlow and Keras.

Table of Contents
- [Dependencies](#Dependencies)
- [Graphical Interface](#Graphical_Interface)
- [Machine Learning Model](#Machine_Learning_Model)
- [Training](#Training)
  - [Rewarding Policy](#Rewarding_Policy)
- [Code Structure](#Code_Structure)
- [License](#License)

# Dependencies
This project requires the following libraries:

- Pygame
- TensorFlow
- Keras
- Numpy

# Graphical_Interface
The graphical interface is built using Pygame. It displays the Minesweeper board, game statistics, and training variables. The interface updates in real-time as the AI plays the game. 

![](https://github.com/Potassium-chromate/Mine-sweeper-AI/blob/main/picture/interface.png)

# Machine_Learning_Model
The AI utilizes a Deep Q-Network (DQN) implemented with TensorFlow and Keras. The model is trained continuously as the AI plays the game, learning to make better decisions over time.

### Rewarding_Policy
In this Minesweeper AI, the agent can perform two types of actions: clicking a cell or toggling a flag on a cell. The reward function is designed to guide the agent to make optimal decisions based on the current state of the game. Below is the rewarding policy used in the model:

1. **Click Action (`action == 1`):**
  - **Clicking a Revealed Cell (`combine_matrix[x][y] != 9`:**
    - Reward: `-2`
    - This is considered a suboptimal action, as clicking an already revealed cell does not provide any new information.
  - **Clicking a Mine (`combine_matrix[x][y] == 9 and board[x][y] == -1`):**
    - Reward: `-1`
    - This action leads to losing the game, thus receiving a negative reward.
  - **Clicking an Unknown Cell (`combine_matrix[x][y] == 9 and board[x][y] != -1`):**
    - Reward: `10`
    - This is considered a good action, as it reveals new information without hitting a mine.
2. **Toggle Flag Action (`action == -1`):**
  - **Toggling Flag on the Last Mine (`minesweeper_logic.pre_check_win returns True`):**
    - Reward: `50`
    - This action leads to winning the game, thus receiving a high positive reward.
  - **Toggling Flag on a Mine (`combine_matrix[x][y] == 9 and board[x][y] == -1`):**
    - Reward: `1`
    - Correctly flagging a mine is a good action, receiving a positive reward.
  - **Toggling Flag on a Wrong Place (`combine_matrix[x][y] == 9 and board[x][y] != -1`):**
    - Reward: `-0.5`
    - Incorrectly flagging a cell is a suboptimal action, receiving a small negative reward.
  - **Toggling Flag on a Revealed Cell (`combine_matrix[x][y] != 9`):**
    - Reward: `-2`
    - This is considered a suboptimal action, as flagging an already revealed cell does not conform to the game rules.

### Model Structure:
The model is a Convolutional Neural Network (CNN) with the following layers:

1. **Input Layer:**
  - Conv2D Layer with 128 filters, a 5x5 kernel, 'relu' activation, and 'same' padding.
  - Input Shape: `(size, size, 1)`
2. **Hidden Layers:**
  - Conv2D Layer with 128 filters, a 1x1 kernel, 'relu' activation, and 'same' padding.
  - Flatten Layer to flatten the 2D matrix data to a vector.
  - Dense Layers with 'relu' activation and varying units: 512, 1024, 512.
  - Dropout Layers with a dropout rate of 0.2 to prevent overfitting.
3. **Output Layer:**
  - Dense Layer with `size**2` units and 'linear' activation.

# Training
The model is trained using a combination of experiences from winning, losing, and ongoing games. The training process involves adjusting the Q-values of the chosen actions based on the received rewards and the maximum Q-value of the next state, following the Q-learning update rule.

# Code_Structure
- `main.py`: The main file to run the Minesweeper AI.
- `minesweeper_logic.py`: Contains the logic for the Minesweeper game, including functions to reveal tiles,       place flags, and check win/lose conditions.
- `minesweeper_training.py`: Contains the logic for training the AI, including the DQN implementation and the     training loop.

# License
This project is licensed under the MIT License - see the LICENSE file for details.
