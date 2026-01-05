import torch
from datasets import load_from_disk
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType
import chess
import re
import wandb

from grpo_model import ChessLMWrapper, ChessLMConfig, CHESS_ENC_HF_PATH, LLM_HF_PATH
from tokenizer import create_tokenizer, process_fen

# --- Configuration ---
DATASET_NAME = "/fs/scratch/PAS2836/lees_stuff/action_value"
OUTPUT_DIR = "./results_grpo"
MAX_COMPLETION_LENGTH = 512

# --- LoRA Configuration ---
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

# Global counter for tracking reward function calls
_reward_call_counter = {'count': 0}


# --- Helper Functions ---

def board_to_piece_list(fen):
    """Convert FEN to piece-by-piece representation."""
    board = chess.Board(fen)
    piece_names = {
        'P': 'Pawn', 'N': 'Knight', 'B': 'Bishop',
        'R': 'Rook', 'Q': 'Queen', 'K': 'King'
    }

    pieces = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            square_name = chess.square_name(square)
            color = 'White' if piece.color == chess.WHITE else 'Black'
            piece_type = piece_names[piece.symbol().upper()]
            pieces.append(f"{square_name}: {color} {piece_type}")

    turn = "White" if board.turn == chess.WHITE else "Black"

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
    result = ", ".join(pieces) + f"\nCastling rights: {castling_str}"
    return turn, result


# --- Reward Function ---

def action_value_reward_func(prompts, completions, **kwargs):
    """
    Reward function using pre-computed action-values.
    """
    moves_batch = kwargs.get('moves', [])
    p_win_batch = kwargs.get('p_win', [])
    fen_batch = kwargs.get('fen', [])

    eval_maps = []
    best_p_wins = []
    for moves, p_wins in zip(moves_batch, p_win_batch):
        eval_map = dict(zip(moves, p_wins))
        eval_maps.append(eval_map)
        best_p_wins.append(max(p_wins) if p_wins else 0.5)

    rewards = []
    format_errors = 0
    illegal_moves = 0
    legal_moves = 0
    total_win_pct_lost = 0.0
    legal_moves_win_pct_lost = 0.0
    total_reasoning_bonus = 0.0
    total_reasoning_tokens = 0

    if len(prompts) > 0:
        print("-" * 50)
        print(f"PROMPT (sample):\n{prompts[0][:500]}...")
        print(f"COMPLETION (sample):\n{completions[0]}")
        print("-" * 50)

    for completion, current_eval_map, fen, best_p_win in zip(completions, eval_maps, fen_batch, best_p_wins):
        reward = 0.0
        reasoning_bonus = 0.0

        try:
            board = chess.Board(fen)
        except ValueError:
            rewards.append(-0.1)
            format_errors += 1
            continue

        answer_pos = completion.find("<answer>")
        if answer_pos > 0:
            reasoning_text = completion[:answer_pos].strip()
            reasoning_tokens = len(reasoning_text) / 4.0
            total_reasoning_tokens += reasoning_tokens
            reasoning_bonus = min(reasoning_tokens / 256.0, 1.0) * 0.05
            total_reasoning_bonus += reasoning_bonus

        move_match = re.search(r"<answer>(.*?)</answer>", completion)
        if not move_match:
            rewards.append(-0.1)
            format_errors += 1
            continue

        move_str = move_match.group(1).strip()

        try:
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                legal_moves += 1
                if move_str in current_eval_map:
                    reward = current_eval_map[move_str]
                    win_pct_lost = best_p_win - reward
                    total_win_pct_lost += win_pct_lost
                    legal_moves_win_pct_lost += win_pct_lost
                else:
                    reward = 0.0
                    total_win_pct_lost += best_p_win
                    legal_moves_win_pct_lost += best_p_win
            else:
                illegal_moves += 1
                reward = 0.0
                total_win_pct_lost += best_p_win
        except ValueError:
            illegal_moves += 1
            reward = 0.0
            total_win_pct_lost += best_p_win

        final_reward = reward + reasoning_bonus
        rewards.append(final_reward)

    # Log metrics
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

        if legal_moves > 0:
            avg_legal_win_pct_lost = (legal_moves_win_pct_lost / legal_moves) * 100
        else:
            avg_legal_win_pct_lost = 0.0

        print(f"\n{'='*60}")
        print(f"BATCH METRICS:")
        print(f"  Correctly formatted: {format_correct_pct:.1f}%")
        print(f"  Legal moves: {legal_pct:.1f}%")
        print(f"  Illegal moves: {illegal_pct:.1f}%")
        print(f"  Avg win% lost (all): {avg_win_pct_lost:.2f}%")
        print(f"  Avg win% lost (legal): {avg_legal_win_pct_lost:.2f}%")
        print(f"  Avg reward: {avg_reward:.3f}")
        print(f"{'='*60}\n")

        _reward_call_counter['count'] += 1
        wandb.log({
            "reward/format_correct_pct": format_correct_pct,
            "reward/legal_moves_pct": legal_pct,
            "reward/illegal_moves_pct": illegal_pct,
            "reward/avg_win_pct_lost": avg_win_pct_lost,
            "reward/avg_win_pct_lost_legal_only": avg_legal_win_pct_lost,
            "reward/avg_reasoning_tokens": avg_reasoning_tokens,
            "reward/avg_reasoning_bonus": avg_reasoning_bonus,
            "reward/avg_reward": avg_reward,
        })

    return rewards


class ChessGRPOTrainer(GRPOTrainer):
    """
    Custom GRPOTrainer that injects chess embeddings before each forward/generate call.

    The key insight: we override _prepare_inputs to inject chess embeddings into
    the model before GRPOTrainer's standard processing.
    """

    def __init__(self, chess_tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chess_tokenizer = chess_tokenizer

    def _prepare_inputs(self, inputs):
        """
        Override to inject chess embeddings before standard GRPO processing.
        """
        # Extract FEN from the batch and compute chess embeddings
        if 'fen' in inputs:
            fens = inputs['fen']
            chess_input_ids, chess_attention_mask = self._tokenize_chess_batch(fens)

            # Move to device
            chess_input_ids = chess_input_ids.to(self.model.device)
            chess_attention_mask = chess_attention_mask.to(self.model.device)

            # Set chess embeddings on the model
            self.model.set_chess_embeddings(chess_input_ids, chess_attention_mask)

        # Call parent's _prepare_inputs
        return super()._prepare_inputs(inputs)

    def _tokenize_chess_batch(self, fens):
        """Tokenize a batch of FEN strings for the chess encoder."""
        all_ids = []
        max_len = 0

        for fen in fens:
            processed = process_fen(fen)
            encoding = self.chess_tokenizer.encode(processed)
            ids = encoding.ids
            all_ids.append(ids)
            max_len = max(max_len, len(ids))

        # Pad to max length
        padded_ids = []
        attention_masks = []
        pad_id = self.chess_tokenizer.token_to_id("[PAD]")

        for ids in all_ids:
            pad_len = max_len - len(ids)
            padded = ids + [pad_id] * pad_len
            mask = [1] * len(ids) + [0] * pad_len
            padded_ids.append(padded)
            attention_masks.append(mask)

        return (
            torch.tensor(padded_ids, dtype=torch.long),
            torch.tensor(attention_masks, dtype=torch.long)
        )


def main():
    # Initialize wandb
    wandb.init(
        project="chessbot-grpo",
        name="grpo-chesslm-peft",
        config={
            "chess_encoder": CHESS_ENC_HF_PATH,
            "llm_model": LLM_HF_PATH,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "learning_rate": 5e-5,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "num_generations": 8,
            "temperature": 0.8,
            "max_completion_length": MAX_COMPLETION_LENGTH,
        }
    )

    # Load dataset
    print("Loading dataset...")
    raw_dataset = load_from_disk(DATASET_NAME)
    dataset = raw_dataset.shuffle(seed=42).select(range(2048))

    # Format dataset - prompts only include text, chess position comes via encoder
    prompt_template = (
        "You are a chess expert playing as {turn}. Determine the best move. "
        "Put your final move in UCI notation in <answer> tags like this: <answer>e2e4</answer>.\n\n<think>\n"
    )

    def format_data(examples):
        prompts = []
        for fen in examples['fen']:
            try:
                board = chess.Board(fen)
                turn = "White" if board.turn == chess.WHITE else "Black"
                prompts.append(prompt_template.format(turn=turn))
            except Exception as e:
                print(f"Error parsing FEN: {fen}, Error: {e}")
                prompts.append("")
        return {"prompt": prompts}

    dataset = dataset.map(format_data, batched=True, batch_size=1000)
    dataset = dataset.filter(lambda x: len(x['prompt']) > 0)

    # Initialize tokenizers
    print("Initializing tokenizers...")
    chess_tokenizer = create_tokenizer()
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_HF_PATH)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    # Initialize model
    print("Loading ChessLM model...")
    config = ChessLMConfig(
        chess_enc_path=CHESS_ENC_HF_PATH,
        llm_path=LLM_HF_PATH,
        freeze_chess_enc=True,
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = ChessLMWrapper(config, lora_config=lora_config)

    # Configure GRPO
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-4,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        logging_steps=1,
        save_steps=100,
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_generations=8,
        temperature=0.8,
        gradient_checkpointing=True,
        bf16=True,
        report_to="wandb",
        # Keep fen, moves, p_win columns for reward function
        remove_unused_columns=False,
    )

    # Initialize trainer
    print("Initializing trainer...")
    trainer = ChessGRPOTrainer(
        chess_tokenizer=chess_tokenizer,
        model=model,
        reward_funcs=[action_value_reward_func],
        args=training_args,
        train_dataset=dataset,
        processing_class=llm_tokenizer,
        peft_config=None,  # Already applied LoRA in model
    )

    # Train
    print("Starting GRPO training...")
    trainer.train()
    print(f"Training complete. Model saved to {OUTPUT_DIR}")

    trainer.save_model(OUTPUT_DIR)
    wandb.finish()


if __name__ == "__main__":
    main()
