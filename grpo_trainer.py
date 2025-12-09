import torch
# Changed from load_dataset, IterableDataset
from datasets import load_from_disk, Dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType
import chess  # Still needed for legality checks
import re
import math

# --- Configuration ---
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# Assuming the action-value dataset is stored locally in HuggingFace dataset format
DATASET_NAME = "/fs/scratch/PAS2836/lees_stuff/action_value"
OUTPUT_DIR = "./results_grpo"
MAX_PROMPT_LENGTH = 256
MAX_COMPLETION_LENGTH = 512  # Increased to allow for reasoning and move output
# STOCKFISH_PATH removed as we are using pre-computed evals

# --- Reward Functions ---


def action_value_reward_func(prompts, completions, **kwargs):
    """
    Reward function that uses pre-computed action-values from the dataset.
    1. Extracts move from <move>...</move>
    2. Checks legality (0.0 if illegal)
    3. Uses provided p_win from eval_map for the move (Win probability as reward)

    Note: While the prompt asks for reasoning, this reward function currently only
    evaluates the quality and legality of the chess move found within <move> tags.
    The reasoning text itself is not directly scored.
    """
    # Extract moves and p_win from kwargs (these come from the dataset)
    moves_batch = kwargs.get('moves', [])
    p_win_batch = kwargs.get('p_win', [])

    # Reconstruct eval_map for each example
    eval_maps = []
    for moves, p_wins in zip(moves_batch, p_win_batch):
        eval_map = dict(zip(moves, p_wins))
        eval_maps.append(eval_map)

    rewards = []

    # Log the first generation of the batch for inspection
    if len(prompts) > 0:
        print("-" * 50)
        print(f"PROMPT (sample):\n{prompts[0]}")
        print(f"COMPLETION (sample):\n{completions[0]}")
        # print(f"EVAL MAP (sample):\n{eval_maps[0]}") # Too verbose
        print("-" * 50)

    for prompt, completion, current_eval_map in zip(prompts, completions, eval_maps):
        reward = 0.0

        # 1. Parse FEN from prompt
        try:
            fen_match = re.search(r"FEN: (.*?)\n", prompt)
            if not fen_match:
                rewards.append(0.0)
                continue
            fen = fen_match.group(1).strip()
            board = chess.Board(fen)
        except ValueError:
            rewards.append(0.0)
            continue

        # 2. Extract Move
        move_match = re.search(r"<move>(.*?)</move>", completion)
        if not move_match:
            # Syntax error - illegal format.
            rewards.append(0.0)
            continue

        move_str = move_match.group(1).strip()

        # 3. Check Legality (against current FEN) and look up eval_map
        try:
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                if move_str in current_eval_map:
                    # Reward is the pre-computed win probability for this move
                    reward = current_eval_map[move_str]
                else:
                    # Legal move, but no pre-computed eval (e.g., very bad move not in top-N)
                    # Treat as a very bad move with 0.0 win probability.
                    reward = 0.0
            else:
                # Illegal move logic.
                reward = 0.0
        except ValueError:
            # Invalid UCI string
            reward = 0.0

        rewards.append(reward)

    # No engine.quit() needed
    return rewards


# --- Main Training Script ---
def main():
    # 1. Load Dataset
    # Load action-value dataset from disk. Shuffle and select a subset.
    raw_dataset = load_from_disk(DATASET_NAME)

    # Shuffle and take a small number of examples for initial training
    # This ensures a standard Dataset type and manageable size.
    dataset = raw_dataset.shuffle(seed=42).select(range(2048))

    # 2. Format Dataset for GRPO
    def format_data(examples):
        # Process batched examples
        prompts = []

        for fen in examples['fen']:
            # Structured Prompt (FEN is included here!)
            prompt = (
                "You are a chess expert. Given the following FEN position, determine the best move. Put your final move in UCI notation in <answer> tags.\n"
                "Keep your reasoning brief and succinct.\n<think>\n"
            )
            prompts.append(prompt)

        # Return only the prompt field. Keep original moves and p_win columns.
        # We'll reconstruct eval_map in the reward function.
        return {
            "prompt": prompts,
        }

    dataset = dataset.map(format_data, batched=True)

    # 3. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Filter dataset to ensure prompts fit within limits (replaces deprecated max_prompt_length arg)
    def filter_long_prompts(example):
        return len(tokenizer(example['prompt'])['input_ids']) <= MAX_PROMPT_LENGTH

    dataset = dataset.filter(filter_long_prompts)

    # 4. Configure GRPO

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-6,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        save_steps=100,
        # max_prompt_length=MAX_PROMPT_LENGTH, # Deprecated
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_generations=4,
        temperature=0.8,
    )

    # 5. Initialize GRPOTrainer
    print(dataset)
    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=[action_value_reward_func],  # Renamed reward function
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # 6. Train
    print("Starting GRPO training with pre-computed action-value rewards...")
    trainer.train()
    print(f"Training complete. Model saved to {OUTPUT_DIR}")

    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    main()
