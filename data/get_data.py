import sys
from pathlib import Path

# The user wants to run this from an arbitrary directory and has specified
# the relative path to the src directory.
# We move this to the top so the subsequent imports can be found.
sys.path.insert(0, '../searchless_chess/src')

import bagz
from apache_beam import coders
from pprint import pprint
import os

# Copied from src/constants.py as requested by the user.
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


# Use the test data which is smaller and more predictable.
# Assumes the data has been downloaded via `data/download.sh`.
bag_path = Path("/fs/scratch/PAS2836/lees_stuff/searchless_chess_data/train/action_value-00000-of-02148_data.bag")

if not bag_path.exists():
    print(f"Error: Data file not found at {bag_path}")
    print("Please run `data/download.sh` from the project root to download the data.")
    raise SystemExit(1)

print(f"Using bag file: {bag_path}")

reader = bagz.BagReader(str(bag_path))

# Decode the record using the appropriate coder from constants.py
coder = CODERS['action_value']

print("\n=== Decoded Entries (First 10) ===")
count = 0
for raw_record in reader:
    if count >= 10:
        break

    fen, move, win_prob = coder.decode(raw_record)

    print(f"\n--- Entry {count + 1} ---")
    print(f"  FEN: {fen}")
    print(f"  Move: {move}")
    print(f"  Win Probability: {win_prob}")
    count += 1

if count == 0:
    print("No entries found in the bag file.")
