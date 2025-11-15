import chess
import chess.engine
import chess.pgn
import torch
from utils import encode

STOCKFISH = "/opt/homebrew/bin/stockfish"
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)

def cheap_eval(board, depth):
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    return info["score"].white().score(mate_score=32000)

def sf_eval(board, depth=12):
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    return info["score"].white().score(mate_score=32000)


def is_quiet(board):
    # 1. Check
    if board.is_check():
        return False

    # 2. No captures or promotions
    for move in board.legal_moves:
        if board.is_capture(move):
            return False
        if move.promotion is not None:
            return False

    # 3. Eval stability check (very cheap)
    E1 = cheap_eval(board, depth=2)
    E2 = cheap_eval(board, depth=4)

    if abs(E1 - E2) > 80:  # threshold
        return False

    return True

def material_count(board):
    values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }
    return sum(values[p.piece_type] for p in board.piece_map().values())

def is_opening(board):
    # Rule 1: skip first 10 plies
    if board.fullmove_number <= 5:
        return True

    # Rule 2: full material means early game
    if material_count(board) == 78:
        return True

    # Rule 3: castling rights → usually opening
    if board.has_castling_rights(chess.WHITE) or board.has_castling_rights(chess.BLACK):
        return True

    return False

def compute_dataset(pgn_path, max_size=50000):
    X = []
    y = []

    count = 0

    with open(pgn_path) as f:
        while count < max_size:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            board = game.board()

            for move in game.mainline_moves():
                board.push(move)

                if is_opening(board):
                    continue

                for candidate in board.legal_moves:
                    board.push(candidate)

                    if is_opening(board):
                        board.pop()
                        continue

                    if is_quiet(board):
                        # Encode & Evaluate
                        X.append(encode(board).squeeze(0))
                        y.append(sf_eval(board))
                        count += 1

                        print(f"Collected {count} positions", end="\r")

                        if count >= max_size:
                            board.pop()
                            break

                    board.pop()

    return torch.stack(X), torch.tensor(y, dtype=torch.float32)

pgn_path = "/Users/aaronkang/Downloads/games.pgn"
X, y = compute_dataset(pgn_path, max_size=100000)

torch.save({"X": X, "y": y}, "dataset2.pt")

engine.quit()

print("\nSaved dataset.pt")
print("Shapes:", X.shape, y.shape)



