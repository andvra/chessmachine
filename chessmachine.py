import chess
import chess.pgn
import numpy as np
import pygame
import os
from pathlib import Path
import time
from pprint import pprint
import torch
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

    offsets_of_games = []
    min_elo = 1800
    num_games_total = 0

    t_before = time.time()
    with open(pgn_file_path) as pgn_file:
        while True:
            num_games_total += 1
            cur_offset = pgn_file.tell()
            headers = chess.pgn.read_headers(pgn_file)
            if headers is None:
                break
            elo_black = int(headers.get("BlackElo"))
            elo_white = int(headers.get("WhiteElo"))
            if elo_black < min_elo or elo_white < min_elo:
                continue
            offsets_of_games.append(cur_offset)
    t_after = time.time()
    print(
        f"ELO limit {min_elo}. Keeping {len(offsets_of_games)}/{num_games_total} games. Took {t_after-t_before} seconds"
    )

    with open(pgn_file_path) as pgn_file:
        for cur_offset in offsets_of_games:
            pgn_file.seek(cur_offset)
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
def train_model(pgn_file_path, batch_size=256, epochs=10):
    import os
    import time

    print("Loading data...")
    dir_cache = "cache"
    Path(dir_cache).mkdir(parents=True, exist_ok=True)
    input_stem = Path(pgn_file_path).stem
    fn_dataset = Path(dir_cache).joinpath(input_stem + ".dataset")

    t_start = time.time()

    if os.path.exists(fn_dataset):
        print(f"Found cached dataset {fn_dataset}")
        dataset = torch.load(fn_dataset, weights_only=False)
    else:
        print("Generating new dataset")
        X, y = extract_labeled_positions_from_pgn(pgn_file_path)
        dataset = ChessPositionDataset(X, y)
        torch.save(dataset, fn_dataset)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    t_end = time.time()
    print("Done loading data in {:.2f} seconds".format(t_end - t_start))

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print(
        "Device name:",
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = ChessPositionNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    num_samples = len(dataloader.dataset)
    for epoch in range(epochs):
        total_loss = 0.0
        t_start = time.time()
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (idx + 1) % 1000 == 0:
                print(
                    f"Epoch {epoch}/{epochs}, batch {idx+1}/{num_samples//batch_size}, Loss: {loss.item():.4f}"
                )
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


def ensure_model(fn_input: str):
    """
    Ensure the model file exists.
    If not, train a new model and save it.
    """

    dir_models = "models"
    stem = Path(fn_input).stem
    fn_model = Path(dir_models).joinpath(stem + ".pth")

    if not os.path.exists(fn_model):
        print(f"Model file {fn_model} does not exist. Training a new model...")
        trained_model = train_model(fn_input, epochs=100)
        Path(os.path.dirname(fn_model)).mkdir(parents=True, exist_ok=True)
        torch.save(trained_model.state_dict(), fn_model)
        print(f"Model saved to {fn_model}")
    else:
        print(f"Loading existing model {fn_model}")
        trained_model = ChessPositionNet()
        trained_model.load_state_dict(torch.load(fn_model, weights_only=True))
        trained_model.eval()

    return trained_model


# --- Entry Point ---
if __name__ == "__main__":
    dir_data = "data"
    fn_input = Path(dir_data).joinpath("lichess_db_standard_rated_2015-04.pgn")
    # fn_input = Path(dir_data).joinpath("small.pgn")
    trained_model = ensure_model(fn_input)

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
