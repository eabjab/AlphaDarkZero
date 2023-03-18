import numpy as np
from tqdm import tqdm
import chess
import torch
import torch.nn as nn
import torch.optim as optim
from FogOfWarBoard import FogOfWarChessBoard
import os
import chess.pgn
import chess.variant

# TODO organize this file and put functions where they belong

# def get_masked_board_state(board: FogOfWarChessBoard, player_color: chess.Color):
#     masked_board = chess.Board(None)
#
#     for square in chess.SQUARES:
#         piece = board.piece_at(square)
#
#         if piece:
#             if piece.color == player_color:
#                 # If the piece belongs to the current player, place it on the masked board
#                 masked_board.set_piece_at(square, piece)
#             else:
#                 # If the piece belongs to the opponent, check if it can be seen by the current player
#                 legal_moves = list(board.generate_pseudo_legal_moves())
#                 visible_squares = set(move.to_square for move in legal_moves if
#                                       move.from_square in board.pieces(chess.PAWN, player_color) or board.attackers(
#                                           player_color, square))
#
#                 if square in visible_squares:
#                     masked_board.set_piece_at(square, piece)
#
#     # If the current player is black, flip the board to get the canonical form
#     if not player_color:
#         masked_board = masked_board.mirror()
#
#     return masked_board


def get_masked_board_state(board, player_color):
    fog_board = board.copy(stack=False)
    fog_board.turn = player_color
    visible_squares = set()

    for square in chess.SQUARES:
        piece = fog_board.piece_at(square)
        if piece is None:
            # fog_board.remove_piece_at(square)
            continue

        if piece.color == player_color:
            legal_moves = list(fog_board.generate_pseudo_legal_moves(from_mask=chess.BB_SQUARES[square]))
            for move in legal_moves:
                if move.drop or move.promotion:
                    continue

                to_square = move.to_square
                # attacked_piece = board.piece_at(to_square)
                visible_squares.add(to_square)
                # fog_board.set_piece_at(to_square, attacked_piece)

            # if attacked_piece is not None and attacked_piece.color != player_color:
            #     fog_board.set_piece_at(to_square, attacked_piece)
            #
            # # Add the squares where the pawn can move without capturing (for pawns only)
            # if piece.piece_type == chess.PAWN and not attacked_piece:
            #     fog_board.set_piece_at(to_square, attacked_piece)

    # Remove all pieces that are not visible to the current player
    for square in chess.SQUARES:
        if square not in visible_squares and fog_board.piece_at(square) is not None and fog_board.piece_at(square).color != player_color:
            fog_board.set_piece_at(square, None)
    # If the current player is black, flip the board to get the canonical form
    if not player_color:
        fog_board = fog_board.mirror()

    return fog_board


def board_to_planes(board: chess.Board):
    piece_planes = np.zeros((12, 8, 8), dtype=np.uint8)

    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if piece:
            # Calculate the index of the plane for the current piece type and color
            plane_index = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)

            # Set the value of the corresponding cell to 1
            rank, file = divmod(square, 8)
            piece_planes[plane_index, rank, file] = 1

    return piece_planes


def action_to_move(action):
    from_square = action % 64
    to_square = action // 64

    move = chess.Move(from_square, to_square)
    return move


def move_to_action(move):
    from_square = move.from_square
    to_square = move.to_square

    action = from_square + 64 * to_square
    return action


def mask_action_vector(board):
    action_vector = np.zeros(4096, dtype=np.float32)
    for move in board.pseudo_legal_moves:
        action = move_to_action(move)
        action_vector[action] = 1
    return action_vector


def create_probability_nn_features(board_list, player_color):
    # Initialize empty planes for the input features
    input_features = np.zeros((185, 8, 8))

    # Iterate through the deque and create the piece planes for each board state
    for i, board in enumerate(board_list):
        all_piece_planes = board_to_planes(board.get_canonical_board(player_color))
        input_features[i*12:(i+1)*12, :, :] = all_piece_planes

    # Create the additional features: player color, total move count, player's castling rights, and no-progress count
    current_board = board_list[0]
    input_features[180, :, :] = int(player_color)
    input_features[181, :, :] = current_board.fullmove_number
    input_features[182, :, :] = int(current_board.has_kingside_castling_rights(player_color))
    input_features[183, :, :] = int(current_board.has_queenside_castling_rights(player_color))
    input_features[184, :, :] = current_board.halfmove_clock

    return input_features


# TODO change channel order
def create_rl_nn_features(board, prob_nn, player_color):
    # First, create the 6 binary planes for the player's pieces
    all_piece_planes = board_to_planes(board.get_canonical_board(player_color))
    player_planes = all_piece_planes[:6, :, :]

    # Then, create the 6 float planes for the probability NN predictions of the opponent's pieces
    # player_color = board.turn
    prob_planes = predict_probability_nn(prob_nn, board, player_color)

    # Create the additional features: player color, total move count, player's castling rights, and no-progress count
    color_plane = np.full((1, 8, 8), int(board.turn))
    move_count_plane = np.full((1, 8, 8), board.fullmove_number)
    castling_rights_planes = np.zeros((2, 8, 8))
    castling_rights_planes[0, :, :] = int(board.has_kingside_castling_rights(board.turn))
    castling_rights_planes[1, :, :] = int(board.has_queenside_castling_rights(board.turn))
    no_progress_count_plane = np.full((1, 8, 8), board.halfmove_clock)

    # Concatenate all planes
    input_features = np.concatenate(
        [player_planes, prob_planes, color_plane, move_count_plane, castling_rights_planes, no_progress_count_plane],
        axis=0)

    return input_features


# TODO figure out way to maintain canonical form alongside move stack

def train_probability_nn(probability_nn, prob_optim, board, device='cuda'):
    # probability_nn.train()

    # Create a list of masked board states for the current canonical board and up to 14 previous states
    masked_board_states = []
    current_board = board.copy()
    player_color = current_board.turn
    masked_board_states.append(get_masked_board_state(current_board, player_color))

    for _ in range(14):
        if len(current_board.move_stack) < 1:
            break
        current_board.pop()
        masked_board_states.append(get_masked_board_state(current_board, player_color))

    # Generate input features for the probability NN
    input_features = create_probability_nn_features(masked_board_states, player_color)

    # Convert input features to a PyTorch tensor and move it to the device
    input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).to(device)

    # Get the true piece planes for the opponent's pieces
    true_piece_planes = board_to_planes(current_board.get_canonical_board(player_color))[6:, :, :]
    num_piece_planes = np.zeros(shape=(6))
    for i, plane in enumerate(true_piece_planes):
        num_pieces = len(current_board.pieces(i + 1, not player_color))
        num_piece_planes[i] = num_pieces

    num_piece_tensor = torch.from_numpy(num_piece_planes.reshape((6,1,1))).to(torch.float32).unsqueeze(0).to(device)
    # Normalize the true_piece_planes tensor
    # normalized_true_piece_planes = true_piece_planes / true_piece_planes.sum(axis=(1, 2))

    # Convert true_piece_planes to a PyTorch tensor and move it to the device
    target_tensor = torch.tensor(true_piece_planes, dtype=torch.float32).unsqueeze(0).to(device)

    # Pass the input features through the probability NN
    output = probability_nn(input_tensor)

    output = output * num_piece_tensor
    # print(output.shape)
    # print(output)

    # Calculate the loss using the predictions and the true piece planes
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target_tensor)

    # Update the model weights
    # optimizer = optim.SGD(lr=0.01, momentum=0.9, params=probability_nn.parameters())
    prob_optim.zero_grad()
    loss.backward()
    prob_optim.step()

    return loss.item()


def predict_probability_nn(probability_nn, board, player_color, device='cuda'):
    # probability_nn.eval()

    # Create a list of masked board states for the current canonical board and up to 14 previous states
    masked_board_states = []
    current_board = board.copy()
    masked_board_states.append(get_masked_board_state(current_board, player_color))

    for _ in range(14):
        if len(current_board.move_stack) < 1:
            break
        current_board.pop()
        masked_board_states.append(get_masked_board_state(current_board, player_color))

    # Generate input features for the probability NN
    input_features = create_probability_nn_features(masked_board_states, player_color)

    # Convert input features to a PyTorch tensor and move it to the device
    input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).to(device)

    # Pass the input features through the probability NN
    output = probability_nn(input_tensor)

    # Convert the output tensor to a numpy array
    output_array = output.detach().cpu().numpy()

    # Reshape the output array to be 6x8x8
    output_array = output_array.reshape((6, 8, 8))

    # Normalize the output array by multiplying each plane by the number of pieces of that type
    for i, plane in enumerate(output_array):
        output_array[i, :, :] = plane * len(board.pieces(i + 1, not player_color))

    return output_array


def create_belief_state(board, prob_planes, player_color, device='cuda'):
    player_planes = board_to_planes(board.get_canonical_board(player_color))[:6, :, :]
    # Create the additional features: player color, total move count, player's castling rights, and no-progress count
    color_plane = np.full((1, 8, 8), int(player_color))
    move_count_plane = np.full((1, 8, 8), board.fullmove_number)
    castling_rights_planes = np.zeros((2, 8, 8))
    castling_rights_planes[0, :, :] = int(board.has_kingside_castling_rights(board.turn))
    castling_rights_planes[1, :, :] = int(board.has_queenside_castling_rights(board.turn))
    no_progress_count_plane = np.full((1, 8, 8), board.halfmove_clock)

    # Concatenate all planes
    input_features = np.concatenate(
        [player_planes, prob_planes, color_plane, move_count_plane, castling_rights_planes, no_progress_count_plane],
        axis=0)

    input_tensor = torch.tensor(input_features, dtype=torch.float32).to(device)

    return input_tensor


def train_models(board, rl_model, rl_optim, prob_model, prob_optim, discount_factor=0.99, device='cuda'):
    mse_loss = nn.MSELoss()

    # Reverse the game and build the game history
    game_history = []
    while board.move_stack:
        game_history.insert(0, board.copy(stack=True))
        board.pop()

    # game_result = board.result()
    # reward = 0
    # if game_result == "1-0":
    #     reward = 1
    # elif game_result == "0-1":
    #     reward = -1
    # else:
    #     reward = -0.1
    winner = None

    if not board.outcome() or board.outcome().winner is None:
        # reward = -0.1       # Assume a draw is slightly bad
        winner = None
    elif board.outcome().winner == chess.WHITE:
        # reward = -1
        winner = chess.WHITE
    else:   # board.outcome().winner == chess.BLACK:
        # reward = 1
        winner = chess.BLACK

    tot_val_loss = 0
    tot_prob_loss = 0
    for i, current_board in enumerate(game_history):
        rl_optim.zero_grad()

        input_features = create_rl_nn_features(current_board, prob_model, current_board.turn)
        input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).to(device)

        value = rl_model(input_tensor)
        target_value = 0

        if i + 10 < len(game_history):
            future_board = game_history[i + 10]
            future_input_features = create_rl_nn_features(future_board, prob_model, future_board.turn)
            future_input_tensor = torch.tensor(future_input_features, dtype=torch.float32).unsqueeze(0).to(device)
            future_value = rl_model(future_input_tensor).detach().cpu().numpy()
            target_value = discount_factor * future_value.item()
        else:
            if winner is None:
                target_value = -0.1
            elif winner == current_board.turn:
                target_value = 1
            elif winner != current_board.turn:
                target_value = -1
            # target_value = reward

        # Invert the reward for alternating players
        # target_value = -target_value

        target_tensor = torch.tensor(target_value, dtype=torch.float32).view(-1, 1).to(device)
        rl_loss = mse_loss(value, target_tensor)
        rl_loss.backward()
        rl_optim.step()
        tot_val_loss += rl_loss.item()

        # Train probability NN
        prob_loss = train_probability_nn(prob_model, prob_optim, current_board)
        tot_prob_loss += prob_loss

    return tot_val_loss / len(game_history), tot_prob_loss / len(game_history)


def choose_best_move(probability_nn, rl_model, board, eval_mode=False, device='cuda'):
    pseudo_legal_moves = list(board.generate_pseudo_legal_moves())
    belief_states = []
    player_color = board.turn
    # Run the masked board state through the probability NN
    predicted_opponent_pieces = predict_probability_nn(probability_nn, board, player_color)

    for move in pseudo_legal_moves:
        # Generate the resulting board state for each move
        board_copy = board.copy(stack=True)
        board_copy.push(move)
        # masked_board = get_masked_board_state(board_copy, player_color)

        # Combine the player's pieces with the predicted opponent pieces to create the belief state
        belief_state = create_belief_state(board_copy, predicted_opponent_pieces, player_color)
        belief_states.append(belief_state)

    # Convert belief states list to a tensor
    belief_states_tensor = torch.stack(belief_states)

    # Run the belief states through the RL model
    predicted_values = rl_model(belief_states_tensor).squeeze(1).detach().cpu().numpy()
    # print(predicted_values.shape)

    if eval_mode:
        # Choose the move with the highest predicted value
        best_move_index = np.argmax(predicted_values)
    else:
        # Choose a move based on the probability distribution of the predicted values
        probabilities = np.exp(predicted_values - np.max(predicted_values))
        probabilities /= np.sum(probabilities)
        best_move_index = np.random.choice(len(pseudo_legal_moves), p=probabilities)

    best_move = pseudo_legal_moves[best_move_index]
    return best_move


def self_play_loop(rl_model, rl_optim, prob_model, prob_optim, num_iterations, num_episodes, device):
    probability_nn = prob_model.to(device)
    rl_model = rl_model.to(device)

    for iteration in range(num_iterations):
        print(f"Starting iteration {iteration + 1}")

        # Create a directory to store the games for the current iteration
        games_dir = f"games_iteration_{iteration + 1}"
        os.makedirs(games_dir, exist_ok=True)

        game_history = []
        # Play the self-play games
        with tqdm(total=num_episodes, desc=f"Self-Play iteration {iteration + 1}/{num_iterations}") as pbar:
            for episode in range(num_episodes):
                # print(f"Playing self-play game {episode + 1}")
                game = chess.pgn.Game()
                board = FogOfWarChessBoard()
                game.setup(board)
                node = game

                while not board.is_game_over(claim_draw=True):
                    move = choose_best_move(probability_nn, rl_model, board, eval_mode=False)
                    board.push(move)
                    # print(board)
                    node = node.add_variation(move)

                game_history.append(board)
                pbar.set_postfix({"Game Result": board.outcome().result() if board.outcome() else "Draw", "Games Played": episode+1, "Games remaining": num_episodes - episode - 1, "Game length": len(board.move_stack)})
                pbar.update()
                # Save the game to disk
                with open(os.path.join(games_dir, f"game_{episode + 1}.pgn"), "w") as game_file:
                    print(game, file=game_file)

        # Train the models on the self-play games
        with tqdm(total=num_episodes, desc=f"Training iteration {iteration + 1}/{num_iterations}") as pbar:
            for episode in range(num_episodes):
                # print(f"Training on self-play game {episode + 1}")
                # with open(os.path.join(games_dir, f"game_{episode + 1}.pgn"), "r") as game_file:
                #     game = read_custom_variant_game(game_file)
                #     board = game.board()
                board = game_history[episode]
                # Train probability NN and RL model on the game
                rl_loss, prob_loss = train_models(board, rl_model, rl_optim, prob_model, prob_optim)
                # print(f"Value loss: {rl_loss}, Probability loss: {prob_loss}")
                # Update the progress bar with the training losses
                pbar.set_postfix({"Value loss": rl_loss, "Probability loss": prob_loss})
                pbar.update()

        # Save the models to disk
        torch.save(probability_nn.state_dict(), f"probability_nn_iteration_{iteration + 1}.pt")
        torch.save(rl_model.state_dict(), f"rl_model_iteration_{iteration + 1}.pt")

    print("Self-play loop completed.")



# class FogOfWarGame(chess.pgn.Game):
#     def board(self, *, chess960=None):
#         fen = self.headers.get("FEN", chess.STARTING_FEN)
#         board = FogOfWarChessBoard(fen=fen)
#         # board.apply_headers(self.headers, chess960=chess960)
#         return board
#
# def read_custom_variant_game(handle):
#     game = FogOfWarGame()
#     game.end().headers = chess.pgn.read_headers(handle)
#     return game if game.end().headers else None