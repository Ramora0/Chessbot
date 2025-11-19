#!/usr/bin/env python
import sys
import struct
import glob
import zlib
import shutil
from pathlib import Path

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

# Path to searchless_chess/src (relative to where you run this script)
sys.path.insert(0, '../searchless_chess/src')

# Where the raw action_value .bag files live
BAG_DIR = Path("/fs/scratch/PAS2836/lees_stuff/searchless_chess_data/train")

# Where to write the partitioned shards
PARTITION_DIR = Path("/fs/scratch/PAS2836/lees_stuff/searchless_chess_partitioned")

# Number of shards to partition into (adjust if needed)
# 1024 is a good default for 15.3B pairs.
N_SHARDS = 1024

# Approx total number of action-value pairs (for tqdm only)
EST_TOTAL_RECORDS = 15_300_000_000

# -------------------------------------------------------------------
# IMPORTS FROM searchless_chess
# -------------------------------------------------------------------

import bagz
from apache_beam import coders
from tqdm.auto import tqdm

# Copied from src/constants.py
CODERS = {
    'fen': coders.StrUtf8Coder(),
    'move': coders.StrUtf8Coder(),
    'count': coders.BigIntegerCoder(),
    'win_prob': coders.FloatCoder(),
}
CODERS['state_value'] = coders.TupleCoder((
    CODERS['fen'],
    CODERS['win_prob'],
))
CODERS['action_value'] = coders.TupleCoder((
    CODERS['fen'],
    CODERS['move'],
    CODERS['win_prob'],
))
CODERS['behavioral_cloning'] = coders.TupleCoder((
    CODERS['fen'],
    CODERS['move'],
))


def main():
    # ----------------------------------------------------------------
    # Nuke old partition data to avoid double counting
    # ----------------------------------------------------------------
    if PARTITION_DIR.exists():
        print(f"Removing existing partition directory: {PARTITION_DIR}")
        shutil.rmtree(PARTITION_DIR)

    PARTITION_DIR.mkdir(parents=True, exist_ok=True)

    # Open all shard files in append-binary mode
    shard_files = [
        open(PARTITION_DIR / f"shard-{i:04d}.bin", "ab")
        for i in range(N_SHARDS)
    ]

    coder = CODERS['action_value']

    try:
        bag_paths = sorted(BAG_DIR.glob("action_value-*_data.bag"))
        if not bag_paths:
            print(f"No action_value-*_data.bag files found in {BAG_DIR}")
            return

        print(f"Found {len(bag_paths)} bag files in {BAG_DIR}")

        total_records = 0

        # Global progress bar over all records (using estimated total)
        with tqdm(
            total=EST_TOTAL_RECORDS,
            unit="rec",
            unit_scale=True,
            desc="Partitioning action-value pairs",
            smoothing=0.01,
        ) as pbar:

            for idx, bag_path in enumerate(bag_paths):
                print(f"[Pass1] Processing bag {idx+1}/{len(bag_paths)}: {bag_path}")
                reader = bagz.BagReader(str(bag_path))

                for raw_record in reader:
                    # Decode bytes -> (fen, move, win_prob)
                    fen, move, win_prob = coder.decode(raw_record)

                    fen_bytes = fen.encode("utf-8")
                    move_bytes = move.encode("utf-8")

                    if len(fen_bytes) > 65535:
                        raise ValueError(f"FEN too long ({len(fen_bytes)} bytes): {fen}")
                    if len(move_bytes) > 65535:
                        raise ValueError(f"Move string too long ({len(move_bytes)} bytes): {move}")

                    # Hash on FEN so all records for the same FEN go to the same shard
                    shard_idx = zlib.crc32(fen_bytes) % N_SHARDS
                    f = shard_files[shard_idx]

                    # Binary layout per record:
                    # [uint16 fen_len][fen_bytes][uint16 move_len][move_bytes][float32 win_prob]
                    f.write(struct.pack("<H", len(fen_bytes)))
                    f.write(fen_bytes)
                    f.write(struct.pack("<H", len(move_bytes)))
                    f.write(move_bytes)
                    f.write(struct.pack("<f", float(win_prob)))

                    total_records += 1
                    pbar.update(1)

        print(f"Done. Wrote {total_records:,} records into {N_SHARDS} shards in {PARTITION_DIR}")

    finally:
        for f in shard_files:
            f.close()


if __name__ == "__main__":
    main()