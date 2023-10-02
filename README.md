# Mine-sweeper-AI
This project contains an implementation of Minesweeper game logic and an AI that can play the game. The AI reveals tiles and places flags on the board, aiming to reveal all non-mine tiles without triggering any mines. The project utilizes a graphical interface built with Pygame and a machine learning model built with TensorFlow and Keras.

Table of Contents
- [Dependencies](#Dependencies)
- [Graphical Interface](#Graphical_Interface)
- [Machine Learning Model](#Machine_Learning_Model)
- [Training](#Training)
  - [Rewarding Policy](#Rewarding_Policy)
  - [Model Structure](#Model_Structure)
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
The AI agent receives rewards or penalties based on its actions during the game, which helps it learn the optimal strategy for playing Minesweeper. Here’s a breakdown of the rewarding policy:

1. **Clicking a Revealed Cell:** `-0.3`  
If the agent clicks on a cell that has already been revealed, it receives a penalty of `-0.3`.

3. **Hitting a Mine:** `-1`    
If the agent reveals a cell containing a mine, it receives a penalty of `-1`, representing the highest penalty, as it leads to losing the game.

4. **Successful Progress:** `0.6`   
If the agent reveals a cell that progresses the game (i.e., revealing a cell that is not a mine and has not been revealed yet), it receives a reward of `0.6`.

5. **YOLO Move:** `-0.2`   
If the agent makes a move that doesn’t progress the game or hit a mine, it receives a penalty of `-0.2`.

6. **Winning the Game:** `1`   
If the agent successfully reveals all cells without mines and flags all cells with mines, it receives the maximum reward of `1`.

### Model_Structure:
The model is a Convolutional Neural Network (CNN) with the following layers:

![](https://github.com/Potassium-chromate/Mine-sweeper-AI/blob/main/picture/structure%20for%20size%209.png)

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
