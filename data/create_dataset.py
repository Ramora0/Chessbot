#!/usr/bin/env python
import os
import struct
import shutil
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
from datasets import Dataset, Features, Value, Sequence, concatenate_datasets
from tqdm.auto import tqdm

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

# Ensure ALL caches and temp files go to scratch, not home directory
SCRATCH_BASE = Path("/fs/scratch/PAS2836/lees_stuff")

# HF cache location
os.environ["HF_HOME"] = str(SCRATCH_BASE / "hf_cache")
os.environ["HF_DATASETS_CACHE"] = str(SCRATCH_BASE / "hf_cache" / "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(SCRATCH_BASE / "hf_cache" / "hub")

# Temp directories for Arrow and general temp files
temp_dir = SCRATCH_BASE / "tmp"
temp_dir.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(temp_dir)
os.environ["TEMP"] = str(temp_dir)
os.environ["TMP"] = str(temp_dir)

# Where the partitioned shards from step 1 live
PARTITION_DIR = SCRATCH_BASE / "searchless_chess_partitioned"

# Where to save the final HF dataset
OUT_DIR = SCRATCH_BASE / "action_value"

# Number of parallel workers (use all available cores)
N_WORKERS = cpu_count()

# -------------------------------------------------------------------
# BINARY RECORD FORMAT (from the partitioning step)
#   [uint16 fen_len][fen_bytes][uint16 move_len][move_bytes][float32 win_prob]
# -------------------------------------------------------------------

# HF schema: FEN string, variable-length list of UCI strings, list of win probs
FEATURES = Features(
    {
        "fen": Value("string"),
        "moves": Sequence(Value("string")),
        "p_win": Sequence(Value("float32")),
    }
)


def read_shard_records(shard_path: Path):
    """
    Generator over (fen: str, move_str: str, win_prob: float32)
    for a given shard-XXXX.bin file.
    """
    with open(shard_path, "rb") as f:
        while True:
            # Read FEN length
            fen_len_bytes = f.read(2)
            if not fen_len_bytes:
                break  # EOF
            if len(fen_len_bytes) < 2:
                raise IOError(f"Truncated shard file at {shard_path} (fen_len)")

            (fen_len,) = struct.unpack("<H", fen_len_bytes)
            fen_bytes = f.read(fen_len)
            if len(fen_bytes) < fen_len:
                raise IOError(f"Truncated shard file at {shard_path} (fen_bytes)")

            fen = fen_bytes.decode("utf-8")

            # Read move length
            move_len_bytes = f.read(2)
            if len(move_len_bytes) < 2:
                raise IOError(f"Truncated shard file at {shard_path} (move_len)")

            (move_len,) = struct.unpack("<H", move_len_bytes)
            move_bytes = f.read(move_len)
            if len(move_bytes) < move_len:
                raise IOError(f"Truncated shard file at {shard_path} (move_bytes)")

            move_str = move_bytes.decode("utf-8")

            # Read win_prob
            win_prob_bytes = f.read(4)
            if len(win_prob_bytes) < 4:
                raise IOError(f"Truncated shard file at {shard_path} (win_prob)")

            (win_prob,) = struct.unpack("<f", win_prob_bytes)

            yield fen, move_str, win_prob


def process_shard_chunk(shard_paths):
    """
    Process a chunk of shard files and return a HuggingFace Dataset.
    This function is called by each worker process.
    """
    def chunk_generator():
        for shard_path in shard_paths:
            moves_by_fen = defaultdict(list)
            pwin_by_fen = defaultdict(list)

            for fen, move_str, win_prob in read_shard_records(shard_path):
                moves_by_fen[fen].append(move_str)
                pwin_by_fen[fen].append(win_prob)

            # Emit one example per FEN
            for fen, moves in moves_by_fen.items():
                vals = pwin_by_fen[fen]
                moves_arr = list(moves)  # list[str]
                pwin_arr = np.asarray(vals, dtype=np.float32).tolist()

                yield {
                    "fen": fen,
                    "moves": moves_arr,
                    "p_win": pwin_arr,
                }

            # Let shard-level dicts get GC'd before the next shard
            del moves_by_fen
            del pwin_by_fen

    # Create dataset from this chunk's generator
    return Dataset.from_generator(chunk_generator, features=FEATURES)


def main():
    # Nuke old dataset dir if it exists so we don't mix runs
    if OUT_DIR.exists():
        print(f"Removing existing dataset directory: {OUT_DIR}")
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Find all shard files
    # ----------------------------------------------------------------
    shard_paths = sorted(PARTITION_DIR.glob("shard-*.bin"))
    if not shard_paths:
        raise RuntimeError(f"No shard-*.bin files found in {PARTITION_DIR}")

    print(f"Found {len(shard_paths)} shard files in {PARTITION_DIR}")
    print(f"Using {N_WORKERS} parallel workers")

    # ----------------------------------------------------------------
    # Split shards into chunks for parallel processing
    # ----------------------------------------------------------------
    chunk_size = max(1, len(shard_paths) // N_WORKERS)
    shard_chunks = [
        shard_paths[i:i + chunk_size]
        for i in range(0, len(shard_paths), chunk_size)
    ]

    print(f"Split into {len(shard_chunks)} chunks (~{chunk_size} shards per chunk)")

    # ----------------------------------------------------------------
    # Process chunks in parallel
    # ----------------------------------------------------------------
    print(f"\nProcessing shard chunks in parallel...")

    with Pool(processes=N_WORKERS) as pool:
        partial_datasets = []
        with tqdm(total=len(shard_chunks), desc="Processing chunks", unit="chunk") as pbar:
            for ds in pool.imap_unordered(process_shard_chunk, shard_chunks):
                partial_datasets.append(ds)
                pbar.update(1)

    print(f"\nCreated {len(partial_datasets)} partial datasets")

    # ----------------------------------------------------------------
    # Concatenate all partial datasets
    # ----------------------------------------------------------------
    print("Concatenating partial datasets...")
    final_dataset = concatenate_datasets(partial_datasets)

    print(f"Final dataset has {len(final_dataset):,} examples")

    # ----------------------------------------------------------------
    # Save to disk
    # ----------------------------------------------------------------
    print(f"Saving dataset to {OUT_DIR} ...")
    final_dataset.save_to_disk(str(OUT_DIR))
    print("Done.")


if __name__ == "__main__":
    main()