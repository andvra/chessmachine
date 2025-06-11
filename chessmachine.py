import chess
import chess.pgn
import numpy as np
import pygame
import torch
from pprint import pprint
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# --- Step 1: Board to Tensor ---
def board_to_tensor(board: chess.Board) -> np.ndarray:
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            plane = piece_map[piece.piece_type] + (
                6 if piece.color == chess.BLACK else 0
            )
            tensor[plane, row, col] = 1.0
    return tensor


# --- Step 2: Result to Label ---
def result_to_label(result_str, board: chess.Board):
    if result_str == "1-0":
        return 1 if board.turn == chess.WHITE else 0
    elif result_str == "0-1":
        return 1 if board.turn == chess.BLACK else 0
    elif result_str == "1/2-1/2":
        return 0
    return None


# --- Step 3: PGN Parsing ---
def extract_labeled_positions_from_pgn(pgn_file_path, max_positions_per_game=30):
    X, y = [], []
    num_boards_read = 0
    with open(pgn_file_path) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            result = game.headers.get("Result")
            board = game.board()
            node = game
            count = 0
            while not node.is_end():
                node = node.variations[0]
                board.push(node.move)
                label = result_to_label(result, board)
                if label is None:
                    continue
                tensor = board_to_tensor(board)
                X.append(tensor)
                y.append(label)
                count += 1
                if count >= max_positions_per_game:
                    break
            num_boards_read += 1
            if num_boards_read % 1000 == 0:
                print(f"Processed {num_boards_read} games...")
            if num_boards_read >= 100:  # Limit for testing
                print("Reached 10,000 games limit for testing.")
                break
    return np.array(X), np.array(y)


# --- Step 4: Dataset Class ---
class ChessPositionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- Step 5: Model ---
class ChessPositionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# --- Step 6: Training ---
def train_model(pgn_file_path, batch_size=64, epochs=10):
    import time

    print("Loading data...")
    t_start = time.time()
    X, y = extract_labeled_positions_from_pgn(pgn_file_path)
    print("Got here")
    dataset = ChessPositionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    t_end = time.time()
    print("Done loading data in {:.2f} seconds".format(t_end - t_start))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = ChessPositionNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        t_start = time.time()
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        t_end = time.time()
        t_tot = t_end - t_start
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Time: {t_tot:.2f} seconds")

    return model


def interactive_chess_board(trained_model: ChessPositionNet):
    pygame.init()

    # Constants
    WIDTH, HEIGHT = 800, 800
    SQUARE_SIZE = WIDTH // 8

    # Initialize screen
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Interactive Chess")

    # Load chess board
    board = chess.Board()

    # Load piece images
    piece_images = {}
    pieces = ["pawn", "knight", "bishop", "rook", "queen", "king"]
    colors = ["white", "black"]
    for color in colors:
        for piece in pieces:
            piece_images[f"{color}_{piece}"] = pygame.image.load(
                f"assets/{color}_{piece}.svg"
            )

    # Draw the board
    def draw_board():
        colors = [(255, 240, 240), (30, 50, 30)]
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                pygame.draw.rect(
                    screen,
                    color,
                    pygame.Rect(
                        col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE
                    ),
                )

    # Draw pieces
    def draw_pieces():
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row = 7 - (square // 8)
                col = square % 8
                color = "white" if piece.color == chess.WHITE else "black"
                piece_name = piece.symbol().lower()
                piece_type = {
                    "p": "pawn",
                    "n": "knight",
                    "b": "bishop",
                    "r": "rook",
                    "q": "queen",
                    "k": "king",
                }[piece_name]
                image = piece_images[f"{color}_{piece_type}"]
                image = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))
                screen.blit(image, (col * SQUARE_SIZE, row * SQUARE_SIZE))

    def move():
        cur_best_output = -1
        cur_best_move = None
        can_claim_draw = False
        the_draw_move = None
        for cur_move in board.legal_moves:
            board.push(cur_move)
            if board.can_claim_draw():
                # Good to know, so we can use it if we are losing
                can_claim_draw = True
                the_draw_move = cur_move
                board.pop()
                print("Can claim draw with move: ", the_draw_move)
                continue
            input_tensor = torch.tensor(
                board_to_tensor(board).reshape(1, 12, 8, 8), dtype=torch.float32
            ).to(next(trained_model.parameters()).device)
            cur_output = trained_model.forward(input_tensor)
            if cur_output > cur_best_output:
                cur_best_output = cur_output
                cur_best_move = cur_move
            board.pop()

        if can_claim_draw and cur_best_output < 0.5:
            print("Claiming draw with move: ", the_draw_move)
            board.push(the_draw_move)
            return
        print("Move: ", cur_best_output.item(), cur_best_move)
        board.push(cur_best_move)

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:  # Press space to make a move
                    move()
        # move()
        draw_board()
        draw_pieces()
        pygame.display.flip()

    pygame.quit()


def ensure_model(fn: str):
    """
    Ensure the model file exists.
    If not, train a new model and save it.
    """
    import os
    from pathlib import Path

    # input_file = "data/lichess_db_standard_rated_2015-04.pgn"
    input_file = "data/small.pgn"
    if not os.path.exists(fn):
        print(f"Model file {fn} does not exist. Training a new model...")
        trained_model = train_model(input_file, epochs=100)
        Path(os.path.dirname(fn)).mkdir(parents=True, exist_ok=True)
        torch.save(trained_model.state_dict(), fn)
        print(f"Model saved to {fn}")
    else:
        trained_model = ChessPositionNet()
        trained_model.load_state_dict(torch.load(fn, weights_only=True))
        trained_model.eval()
        print(f"Model file {fn} already exists.")

    return trained_model


# --- Entry Point ---
if __name__ == "__main__":
    trained_model = ensure_model("models/chess_model.pth")

    interactive_chess_board(trained_model)
    # TODO:
    # - Send an empty board to interactive_chess_board
    # - Check: do we use current color as input when training?
    # - Add a way to make moves in the interactive board
    # - Add a way to save the model
    # - Add a way to load the model
    # - Add a way to evaluate the model
    # - Add a way to visualize the model's predictions
    # - Add a way to visualize the model's performance
    # - Evaluate all available chess positions and pick the best one
