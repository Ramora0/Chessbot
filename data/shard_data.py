#!/usr/bin/env python
import sys
import struct
import glob
import zlib
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count, Manager
import os
import time

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

# Path to searchless_chess/src (relative to where you run this script)
sys.path.insert(0, '../searchless_chess/src')

# Where the raw action_value .bag files live
BAG_DIR = Path("/fs/scratch/PAS2836/lees_stuff/searchless_chess_data/train")

# Where to write the partitioned shards
PARTITION_DIR = Path("/fs/scratch/PAS2836/lees_stuff/searchless_chess_partitioned")

# Number of shards to partition into
N_SHARDS = 512

# Approx total number of action-value pairs (for tqdm only)
EST_TOTAL_RECORDS = 15_300_000_000

# Number of parallel workers (use all available cores)
N_WORKERS = cpu_count()

# How often to update the shared progress counter (every N records)
PROGRESS_UPDATE_INTERVAL = 10000

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


def process_bag_file(args):
    """
    Process a single bag file and write to temporary shard files.
    Returns the number of records processed.
    """
    bag_path, bag_idx, temp_dir, n_shards = args
    
    # Create temp directory for this bag's shards
    bag_temp_dir = temp_dir / f"bag_{bag_idx:04d}"
    bag_temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Open temp shard files for this bag
    temp_shard_files = [
        open(bag_temp_dir / f"shard_{i:04d}.bin", "wb")
        for i in range(n_shards)
    ]
    
    coder = CODERS['action_value']
    records_processed = 0
    
    try:
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
            shard_idx = zlib.crc32(fen_bytes) % n_shards
            f = temp_shard_files[shard_idx]
            
            # Binary layout per record:
            # [uint16 fen_len][fen_bytes][uint16 move_len][move_bytes][float32 win_prob]
            f.write(struct.pack("<H", len(fen_bytes)))
            f.write(fen_bytes)
            f.write(struct.pack("<H", len(move_bytes)))
            f.write(move_bytes)
            f.write(struct.pack("<f", float(win_prob)))
            
            records_processed += 1
    
    finally:
        for f in temp_shard_files:
            f.close()
    
    return records_processed


def merge_temp_shards(temp_dir, final_dir, n_shards, n_bags):
    """
    Merge all temporary shard files into final shard files.
    """
    print(f"\nMerging temporary shards into final {n_shards} shards...")
    
    with tqdm(total=n_shards, desc="Merging shards", unit="shard") as pbar:
        for shard_idx in range(n_shards):
            final_shard_path = final_dir / f"shard-{shard_idx:04d}.bin"
            
            with open(final_shard_path, "wb") as final_file:
                # Append all temp files for this shard
                for bag_idx in range(n_bags):
                    temp_shard_path = temp_dir / f"bag_{bag_idx:04d}" / f"shard_{shard_idx:04d}.bin"
                    
                    if temp_shard_path.exists():
                        with open(temp_shard_path, "rb") as temp_file:
                            # Copy in chunks for memory efficiency
                            while True:
                                chunk = temp_file.read(1024 * 1024)  # 1MB chunks
                                if not chunk:
                                    break
                                final_file.write(chunk)
            
            pbar.update(1)
    
    print(f"Merge complete. Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    print("Cleanup complete.")


def main():
    # ----------------------------------------------------------------
    # Setup directories
    # ----------------------------------------------------------------
    if PARTITION_DIR.exists():
        print(f"Removing existing partition directory: {PARTITION_DIR}")
        shutil.rmtree(PARTITION_DIR)
    
    PARTITION_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = PARTITION_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # ----------------------------------------------------------------
    # Find all bag files
    # ----------------------------------------------------------------
    bag_paths = sorted(BAG_DIR.glob("action_value-*_data.bag"))
    if not bag_paths:
        print(f"No action_value-*_data.bag files found in {BAG_DIR}")
        return
    
    print(f"Found {len(bag_paths)} bag files in {BAG_DIR}")
    print(f"Using {N_WORKERS} parallel workers")
    print(f"Partitioning into {N_SHARDS} shards")
    
    # ----------------------------------------------------------------
    # Process bag files in parallel
    # ----------------------------------------------------------------
    args_list = [
        (bag_path, idx, temp_dir, N_SHARDS)
        for idx, bag_path in enumerate(bag_paths)
    ]
    
    print(f"\nProcessing {len(bag_paths)} bag files in parallel...\n")
    
    total_records = 0
    with Pool(processes=N_WORKERS) as pool:
        # Use imap_unordered for progress tracking
        with tqdm(total=len(bag_paths), desc="Processing bags", unit="bag") as pbar:
            for records_count in pool.imap_unordered(process_bag_file, args_list):
                total_records += records_count
                pbar.update(1)
    
    print(f"\nProcessed {total_records:,} total records")
    
    # ----------------------------------------------------------------
    # Merge temporary shards into final shards
    # ----------------------------------------------------------------
    merge_temp_shards(temp_dir, PARTITION_DIR, N_SHARDS, len(bag_paths))
    
    print(f"\nDone! Wrote {total_records:,} records into {N_SHARDS} shards in {PARTITION_DIR}")


if __name__ == "__main__":
    main()