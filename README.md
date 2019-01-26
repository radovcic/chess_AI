# chess_AI

Chess AI trained on human played games by supervised learning.

## Description

Chess AI is trained to distinguish strong human-played moves from randomly played moves. 

Data was extracted from [KingBase Lite 2019](http://kingbase-chess.net/download/762) database from [KingBase](http://www.kingbase-chess.net/) website. It contains over one million modern games with players elo rating larger than 2200. For each position from those games a human played move, a random move and positions after those moves were extracted. From positions a bitboard features were generated. No hand selected features were used. In total more then 160 millions training examples were extracted.

Siamese deep neural network with more than 2 million parameters was used for classification. After training it reached accuracy of 85.6% on the test set.

In a given position, the model can be used to predict a human move from all available legal moves. After training the accuracy for move prediction is 34%. This is a good result since on average there is ~31 legal move and no tree search was used in move prediction.

The model can also be used to play a chess game. In the beginning of the game it plays reasonably, since it has seen many similar positions in human play. As the game goes on, the position can deviate from human-played positions and then the quality of play deteriorates since there were no similar examples to learn from. The game played by the model for first 30 plies is shown below.  

### Game played by the model (30 plies):
<img src="game.gif" width="300"/>
