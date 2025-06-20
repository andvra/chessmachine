Make sure to download larger PGN files for training. For instance from the Lichess database:
https://database.lichess.org/standard/

## Architecture

Dataset --> Neural net (back prop) --> Training --> Model

Model --> Inference --> Move

## Idea

Chessmachine is a chess engine with training, evaluation and the possibility to play against the machine. 

### Training

As input to training we use recorded games from chess websites. We pick the games where the players have an ELO rating above a certain threshold and consider the moves in these games as optimal.
We want our input layer to give a lossless representation of the board at the given step. For this we use a design where each piece is given 8x8 nodes with either 0 (off) or 1 (on). We need two sets of these boards, on for each color. Since we have six different types of pieces we need 2x6x8x8=768 input nodes.
Output layer consisting of a single node where aim is to make it return 0 for losing position, 1/2 for draw and 1 for a winning position. 

### Training data - PGN files
Each [PGN](https://en.wikipedia.org/wiki/Portable_Game_Notation) files contain representations of a number of chess games. Each chess game includes metadata such as who is playing, ELO ratings and the result of the game.


### Playing

When we have a trained model we can play agains the computer, or see it play itself. When it's the computers turn it will find all allowed moves and rank them by win probability. The best possible move will we used for moving.

