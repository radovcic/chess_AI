"""
Module for supervised learning on human played chess games.
"""


import chess
import chess.pgn
import numpy as np
from scipy import sparse
import time


def features(board):
    
    """
    Argument:
        board -- chess board position
    
    Returns:
        x -- features, numpy array of dim (774,1)
        x[0:768] -- bitboards for 12 pieces on 64 squares
        x[768:772] -- castling rights
        x[772] -- en passant
        x[773] -- turn
    """
    
    x = np.zeros((774,1), dtype = np.int8)
    
    pieces_positions = board.piece_map()
    
    for square in pieces_positions:
        piece = pieces_positions[square]
        figure = piece.piece_type
        color = piece.color
        x[64 * (6 * color + figure - 1) + square,0] = 1
    
    x[768,0] = board.has_kingside_castling_rights(1)
    x[769,0] = board.has_queenside_castling_rights(1)
    x[770,0] = board.has_kingside_castling_rights(0)
    x[771,0] = board.has_queenside_castling_rights(0)
    x[772,0] = board.has_legal_en_passant() == True
    x[773,0] = board.turn
    
    return x


def count_games(pgn_file):
    
    """
    Argument:
        pgn_file -- pgn file with chess games
    
    Returns:
        counter -- number of chess games in pgn file 
    """
    
    pgn_file.seek(0)
             
    counter = 0

    while True:
        headers = chess.pgn.read_headers(pgn_file)
        if headers is None:
            break
        else:
            counter += 1
 
    return counter


def pgn_to_data(pgn_file, number_of_games):
    
    """
    Argument:
        pgn_file -- pgn file with chess games
        number_of_games -- number of games in pgn file
    
    Returns:
        data -- data from pgn games
        positions_plies -- number of plies in a position to play
        games_plies -- total number of plies in the game
    """   
    
    games_plies = np.empty(number_of_games, dtype = np.int16)
    positions_plies = np.empty(500*number_of_games, dtype = np.int16)
    data_temp = np.empty((10**6, 2*774+1), dtype = np.int8)
    data_list = []
    
    pgn_file.seek(0)

    index_plies = 0
    index = 0

    tic = time.process_time()
    tic_sub = time.process_time()

    for i in range(0,number_of_games):
    
        if (i+1) % 1000 == 0:
            toc_sub = time.process_time()             
            print ("Total computation time = " + "{0:0.2f}".format(toc_sub - tic) 
               + "s and for n = " + str(i+1) + " computation time = " + "{0:0.2f}".format(toc_sub - tic_sub) + "s.")
            tic_sub = time.process_time()      
        
        game = chess.pgn.read_game(pgn_file)
         
        plies_in_game = len(list(game.mainline_moves()))
        games_plies[i] = plies_in_game
        
        board = game.board()
        plies_played = 0
        
        for ply in game.mainline_moves():
            
            if ply == chess.Move.from_uci('0000'): break
            
            legal_moves_list = list(board.legal_moves)
            number_of_legal_moves = len(legal_moves_list)
            
            position_random_played = board.copy()
            board.push(ply)
            plies_played = plies_played + 1
            
            if number_of_legal_moves > 1:
                    
                if plies_played > 0 and plies_played <= 2 and np.random.randint(0,100) != 0: continue
                elif plies_played > 2 and plies_played <= 4 and np.random.randint(0,50) != 0: continue            
                elif plies_played > 4 and plies_played <= 6 and np.random.randint(0,25) != 0: continue            
                elif plies_played > 6 and plies_played <= 8 and np.random.randint(0,10) != 0: continue            
                elif plies_played > 8 and plies_played <= 10 and np.random.randint(0,5) != 0: continue           
                elif plies_played > 10 and plies_played <= 12 and np.random.randint(0,3) != 0: continue         
                elif plies_played > 12 and plies_played <= 14 and np.random.randint(0,2) != 0: continue           
                
                position_played = board.copy()
                move_played_index = legal_moves_list.index(ply)
                positions_plies[index_plies] = plies_played
                index_plies += 1
                 
                while True:                   
                    move_random_index = np.random.randint(0, number_of_legal_moves)                 
                    if move_played_index != move_random_index: break
                        
                position_random_played.push(legal_moves_list[move_random_index])
                
                features_1 = features(position_played) 
                features_2 = features(position_random_played)
                result = position_played.turn

                data_temp[2*index,:] = np.vstack((features_1,features_2,result)).T
                data_temp[2*index+1,:] = np.vstack((features_2,features_1,-result+1)).T
            
                index = index + 1
    
        if index > 499000:
            data_sparse = sparse.csr_matrix(data_temp[0:2*index,:])
            data_list.append(data_sparse)
            index = 0
    
    data_sparse = sparse.csr_matrix(data_temp[0:2*index,:])
    data_list.append(data_sparse)
    data = sparse.vstack(data_list)
    
    positions_plies = positions_plies[0:index_plies]
    
    toc = time.process_time()
    print("Total computation time = " + "{0:0.2f}".format(toc - tic) + "s.")

    return data, positions_plies, games_plies


def positions(game):
    
    """
    Argument:
        game -- chess game
    
    Returns:
        position_to_play -- chess board position to play
        position_played -- chess board position after playing the "best - human" move
        plies_played_count - 1 -- number of plies to reach position_to_play
    """
    
    plies_in_game = len(list(game.mainline_moves()))
    
    while True:
        
        plies_played = np.random.randint(1, plies_in_game+1)
                
        plies_played_count = 0
        board = game.board()
        for ply in game.mainline_moves():
            if plies_played_count == plies_played: break
            board.push(ply)
            plies_played_count = plies_played_count + 1

        position_played = board.copy()
        
        move_played = board.pop()
        
        position_to_play = board.copy()
              
        legal_moves_list = list(board.legal_moves)
        number_of_legal_moves = len(legal_moves_list)
        
        if number_of_legal_moves > 1: break     

    return position_to_play, position_played, plies_played_count - 1


def move_to_play(board, model, features):
    
    """
    Argument:
        board -- chess board position
        model -- model to use
        featuers -- function for featuers from board
    
    Returns:
        best_move -- best move in the position
        board_final -- board position after the best move
    """
    
    moves_legal = list(board.legal_moves)
    
    n = len(moves_legal)
    
    x = np.zeros((n,n), dtype = np.float)
    
    for i in range(n):
        board1 = board.copy()
        board1.push(moves_legal[i])
        features1 = features(board1)
        for j in range(n):
            board2 = board.copy()
            board2.push(moves_legal[j])
            features2 = features(board2)
            
            x[i,j] = model.predict([features1.T,features2.T])[0,0]

    if board.turn == 0:
        move_index = np.argmin(np.mean(x, axis = 0) - np.mean(x, axis = 1))
        
    elif board.turn == 1:
        move_index = np.argmax(np.mean(x, axis = 0) - np.mean(x, axis = 1))
    
    best_move = moves_legal[move_index]
    board_final = board.copy()
    board_final.push(best_move)
    
    return best_move, board_final

