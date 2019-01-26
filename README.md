# chess_AI

Chess AI trained on human played games by supervised learning.

## Description

Chess AI is trained to distinguish strong human played moves from random played moves. 

Data was extracted from [KingBase Lite 2019](http://kingbase-chess.net/download/762) database from [KingBase](http://www.kingbase-chess.net/) website. It contains over one million modern games with players elo rating larger then 2200. For each position from those games a human played move, a random move and positions after those moves were extracted. From positions a bitboard features were generated. No hand selected features were used. In total more then 160 millions training examples were extracted.



### Game played by the model (30 plies):
<img src="game.gif" width="300"/>
