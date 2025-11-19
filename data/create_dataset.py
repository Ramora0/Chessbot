#!/usr/bin/env python
import os
import struct
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
from datasets import Dataset, Features, Value, Sequence
from tqdm.auto import tqdm

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

# HF cache location (you can also `export HF_HOME=...` in shell instead)
os.environ["HF_HOME"] = "/fs/scratch/PAS2836/lees_stuff/hf_cache"

# Where the partitioned shards from step 1 live
PARTITION_DIR = Path("/fs/scratch/PAS2836/lees_stuff/searchless_chess_partitioned")

# Where to save the final HF dataset
OUT_DIR = Path("/fs/scratch/PAS2836/lees_stuff/action_value")

# -------------------------------------------------------------------
# BINARY RECORD FORMAT (from the partitioning step)
#   [uint16 fen_len][fen_bytes][uint16 move_len][move_bytes][float32 win_prob]
# -------------------------------------------------------------------


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


def examples_generator():
    """
    Main generator for Dataset.from_generator.

    For each shard:
      - group (fen, move_str, p_win) by fen
      - yield one example per fen:
          {"fen": str, "moves": [str], "p_win": [float32]}
    """
    shard_paths = sorted(PARTITION_DIR.glob("shard-*.bin"))
    if not shard_paths:
        raise RuntimeError(f"No shard-*.bin files found in {PARTITION_DIR}")

    print(f"Found {len(shard_paths)} shard files in {PARTITION_DIR}")

    for shard_path in tqdm(shard_paths, desc="Building HF examples from shards"):
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


def main():
    # Nuke old dataset dir if it exists so we don't mix runs
    if OUT_DIR.exists():
        print(f"Removing existing dataset directory: {OUT_DIR}")
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # HF schema: FEN string, variable-length list of UCI strings, list of win probs
    features = Features(
        {
            "fen": Value("string"),
            "moves": Sequence(Value("string")),
            "p_win": Sequence(Value("float32")),
        }
    )

    print("Creating Hugging Face dataset from shards (this will take a while)...")
    ds = Dataset.from_generator(examples_generator, features=features)

    print(f"Saving dataset to {OUT_DIR} ...")
    ds.save_to_disk(str(OUT_DIR))
    print("Done.")


if __name__ == "__main__":
    main()