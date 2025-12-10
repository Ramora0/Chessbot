import torch
# Changed from load_dataset, IterableDataset
from datasets import load_from_disk, Dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType
import chess  # Still needed for legality checks
import re
import wandb

# --- Configuration ---
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# Assuming the action-value dataset is stored locally in HuggingFace dataset format
DATASET_NAME = "/fs/scratch/PAS2836/lees_stuff/action_value"
OUTPUT_DIR = "./results_grpo"
MAX_COMPLETION_LENGTH = 512
# STOCKFISH_PATH removed as we are using pre-computed evals

# --- LoRA Configuration ---
LORA_R = 16              # Rank of LoRA matrices (8, 16, 32, 64)
LORA_ALPHA = 32          # Scaling factor (usually 2x rank)
LORA_DROPOUT = 0.05      # Dropout for LoRA layers
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]  # For Llama architecture

# --- Helper Functions ---

# Global counter for tracking reward function calls (for wandb logging)
_reward_call_counter = {'count': 0}


def board_to_piece_list(fen):
    """
    Convert a FEN string to an explicit piece-by-piece representation with full names.
    Example: "a2: White Pawn, b2: White Pawn, ..., e1: White King, ..."
    Also includes turn information and castling rights.
    """
    board = chess.Board(fen)

    # Map piece symbols to full names
    piece_names = {
        'P': 'Pawn',
        'N': 'Knight',
        'B': 'Bishop',
        'R': 'Rook',
        'Q': 'Queen',
        'K': 'King'
    }

    pieces = []

    # Iterate through all squares (a1 to h8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # Get square name (e.g., 'e4')
            square_name = chess.square_name(square)

            # Determine full piece name
            color = 'White' if piece.color == chess.WHITE else 'Black'
            piece_type = piece_names[piece.symbol().upper()]
            piece_name = f"{color} {piece_type}"

            pieces.append(f"{square_name}: {piece_name}")

    # Add turn information
    turn = "White" if board.turn == chess.WHITE else "Black"

    # Add castling rights
    castling = []
    if board.has_kingside_castling_rights(chess.WHITE):
        castling.append("White kingside")
    if board.has_queenside_castling_rights(chess.WHITE):
        castling.append("White queenside")
    if board.has_kingside_castling_rights(chess.BLACK):
        castling.append("Black kingside")
    if board.has_queenside_castling_rights(chess.BLACK):
        castling.append("Black queenside")

    castling_str = ", ".join(castling) if castling else "None"

    # Combine pieces with metadata
    result = ", ".join(pieces)
    result += f"\nCastling rights: {castling_str}"

    return turn, result


# --- Reward Functions ---


def action_value_reward_func(prompts, completions, **kwargs):
    """
    Reward function that uses pre-computed action-values from the dataset.
    1. Calculates reasoning bonus (soft reward for reasoning tokens)
    2. Extracts move from <answer>...</answer>
    3. Checks format (-0.1 if missing tags)
    4. Checks legality (0.0 if illegal)
    5. Uses provided p_win from eval_map for the move (Win probability as reward)

    Reward structure:
    - Missing <answer> tags: -0.1 (worst penalty)
    - Illegal move: 0.0
    - Legal move: p_win value (0.0 to 1.0)
    - Reasoning bonus: up to +0.05 for 256+ tokens of reasoning before <answer>

    Final reward = move_reward + reasoning_bonus
    """
    # Extract moves, p_win, and fen from kwargs (these come from the dataset)
    moves_batch = kwargs.get('moves', [])
    p_win_batch = kwargs.get('p_win', [])
    fen_batch = kwargs.get('fen', [])

    # Reconstruct eval_map for each example
    eval_maps = []
    best_p_wins = []  # Track best possible p_win for each position
    for moves, p_wins in zip(moves_batch, p_win_batch):
        eval_map = dict(zip(moves, p_wins))
        eval_maps.append(eval_map)
        best_p_wins.append(max(p_wins) if p_wins else 0.5)

    rewards = []

    # Tracking metrics
    format_errors = 0
    illegal_moves = 0
    legal_moves = 0
    total_win_pct_lost = 0.0
    legal_moves_win_pct_lost = 0.0  # Track win% lost only for legal moves
    total_reasoning_bonus = 0.0
    total_reasoning_tokens = 0

    # Log the first generation of the batch for inspection
    if len(prompts) > 0:
        print("-" * 50)
        # Truncate long prompts
        print(f"PROMPT (sample):\n{prompts[0][:500]}...")
        print(f"COMPLETION (sample):\n{completions[0]}")
        print("-" * 50)

    for completion, current_eval_map, fen, best_p_win in zip(completions, eval_maps, fen_batch, best_p_wins):
        reward = 0.0
        reasoning_bonus = 0.0

        # 1. Get FEN from dataset and create board
        try:
            board = chess.Board(fen)
        except ValueError:
            rewards.append(-0.1)
            format_errors += 1
            continue

        # 2. Calculate reasoning bonus before extracting move
        # Extract text before <answer> tag as reasoning
        answer_pos = completion.find("<answer>")
        if answer_pos > 0:
            reasoning_text = completion[:answer_pos].strip()
            # Estimate tokens using heuristic: ~4 characters per token
            reasoning_tokens = len(reasoning_text) / 4.0
            total_reasoning_tokens += reasoning_tokens

            # Soft reward: linear up to 256 tokens, max bonus of 0.05
            # This encourages reasoning without dominating the main reward
            reasoning_bonus = min(reasoning_tokens / 256.0, 1.0) * 0.05
            total_reasoning_bonus += reasoning_bonus

        # 3. Extract Move from <answer> tags
        move_match = re.search(r"<answer>(.*?)</answer>", completion)
        if not move_match:
            # Format error - missing <answer> tags (worst penalty)
            rewards.append(-0.1)
            format_errors += 1
            continue

        move_str = move_match.group(1).strip()

        # 4. Check Legality and look up eval_map
        try:
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                legal_moves += 1
                if move_str in current_eval_map:
                    # Reward is the pre-computed win probability for this move
                    reward = current_eval_map[move_str]
                    # Track win% lost (best possible - actual)
                    win_pct_lost = best_p_win - reward
                    total_win_pct_lost += win_pct_lost
                    legal_moves_win_pct_lost += win_pct_lost
                else:
                    # Legal move, but no pre-computed eval (e.g., very bad move not in top-N)
                    reward = 0.0
                    total_win_pct_lost += best_p_win
                    legal_moves_win_pct_lost += best_p_win
            else:
                # Illegal move
                illegal_moves += 1
                reward = 0.0
                total_win_pct_lost += best_p_win
        except ValueError:
            # Invalid UCI string (can't parse move)
            illegal_moves += 1
            reward = 0.0
            total_win_pct_lost += best_p_win

        # 5. Add reasoning bonus to final reward
        final_reward = reward + reasoning_bonus
        rewards.append(final_reward)

    # Print and log metrics
    total = len(rewards)
    if total > 0:
        format_correct_pct = ((total - format_errors) / total) * 100
        legal_pct = (legal_moves / total) * 100
        illegal_pct = (illegal_moves / total) * 100
        format_error_pct = (format_errors / total) * 100
        avg_win_pct_lost = (total_win_pct_lost / total) * 100
        avg_reward = sum(rewards) / total
        avg_reasoning_bonus = total_reasoning_bonus / total
        avg_reasoning_tokens = total_reasoning_tokens / total

        # Calculate average win% lost for legal moves only
        if legal_moves > 0:
            avg_legal_win_pct_lost = (
                legal_moves_win_pct_lost / legal_moves) * 100
        else:
            avg_legal_win_pct_lost = 0.0

        print(f"\n{'='*60}")
        print(f"BATCH METRICS:")
        print(
            f"  Correctly formatted: {format_correct_pct:.1f}% ({total - format_errors}/{total})")
        print(f"  Legal moves: {legal_pct:.1f}% ({legal_moves}/{total})")
        print(f"  Illegal moves: {illegal_pct:.1f}% ({illegal_moves}/{total})")
        print(
            f"  Format errors: {format_error_pct:.1f}% ({format_errors}/{total})")
        print(f"  Avg win% lost per move (all): {avg_win_pct_lost:.2f}%")
        print(
            f"  Avg win% lost per move (legal only): {avg_legal_win_pct_lost:.2f}%")
        print(f"  Avg reasoning tokens: {avg_reasoning_tokens:.1f}")
        print(f"  Avg reasoning bonus: {avg_reasoning_bonus:.4f}")
        print(f"  Avg reward: {avg_reward:.3f}")
        print(f"{'='*60}\n")

        # Log to wandb
        _reward_call_counter['count'] += 1
        wandb.log({
            "reward/format_correct_pct": format_correct_pct,
            "reward/legal_moves_pct": legal_pct,
            "reward/illegal_moves_pct": illegal_pct,
            "reward/format_errors_pct": format_error_pct,
            "reward/avg_win_pct_lost": avg_win_pct_lost,
            "reward/avg_win_pct_lost_legal_only": avg_legal_win_pct_lost,
            "reward/avg_reasoning_tokens": avg_reasoning_tokens,
            "reward/avg_reasoning_bonus": avg_reasoning_bonus,
            "reward/avg_reward": avg_reward,
            "reward/batch_count": _reward_call_counter['count'],
        })

    return rewards


# --- Main Training Script ---
def main():
    # Initialize wandb
    wandb.init(
        project="chessbot-grpo",
        name="grpo-deepseek-r1-lora",
        config={
            "model": MODEL_NAME,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "learning_rate": 5e-5,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "num_generations": 4,
            "temperature": 0.8,
            "max_completion_length": MAX_COMPLETION_LENGTH,
        }
    )

    # 1. Load Dataset
    # Load action-value dataset from disk. Shuffle and select a subset.
    raw_dataset = load_from_disk(DATASET_NAME)

    # Shuffle and take a small number of examples for initial training
    # This ensures a standard Dataset type and manageable size.
    dataset = raw_dataset.shuffle(seed=42).select(range(2048))

    # 2. Format Dataset for GRPO
    # Pre-compile template for faster formatting
    prompt_template = (
        "You are a chess expert playing the following game of chess as {}; determine the best move. "
        "Put your final move in UCI notation in <answer> tags like this: <answer>e2e4</answer>.\n"
        "Position (square: piece):\n{}\n\n<think>\n"
    )

    def format_data(examples):
        # Convert FEN to piece list format for each example
        prompts = []
        for fen in examples['fen']:
            try:
                piece_list = board_to_piece_list(fen)
                turn, piece_list = piece_list
                prompts.append(prompt_template.format(turn, piece_list))
            except Exception as e:
                # If FEN parsing fails, use an empty prompt (will be filtered out)
                print(f"Error parsing FEN: {fen}, Error: {e}")
                prompts.append("")
        return {"prompt": prompts}

    dataset = dataset.map(format_data, batched=True, batch_size=1000)

    # 3. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Filter out any empty prompts (from failed FEN parsing)
    def filter_valid_prompts(example):
        return len(example['prompt']) > 0

    dataset = dataset.filter(filter_valid_prompts)

    # 4. Configure GRPO

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        # Higher LR for LoRA training (was 2e-6 for full fine-tuning)
        learning_rate=1e-4,
        num_train_epochs=1,
        per_device_train_batch_size=32,
        # gradient_accumulation_steps=2,
        logging_steps=1,
        save_steps=100,
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_generations=8,
        temperature=0.8,
        # Memory optimization for H100
        gradient_checkpointing=True,  # Saves memory at cost of compute
        bf16=True,  # Use bfloat16 precision on H100
        # Enable wandb logging
        report_to="wandb",
    )

    # 5. Initialize LoRA Configuration
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # 6. Initialize GRPOTrainer with LoRA
    print(dataset)
    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=[action_value_reward_func],  # Renamed reward function
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,  # Enable LoRA training
    )

    # 7. Train
    print("Starting GRPO training with LoRA and pre-computed action-value rewards...")
    trainer.train()
    print(f"Training complete. Model saved to {OUTPUT_DIR}")

    trainer.save_model(OUTPUT_DIR)

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
