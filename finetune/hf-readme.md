---
license: cc-by-4.0
tags:
- chess
- stockfish
pretty_name: ChessBench Action-Values + Mate-in-N
size_categories:
- 100M<n<1B
---

# Dataset Summary

This dataset is an adaptation of the action-value data released with “Grandmaster-Level Chess Without Search” (DeepMind). It provides:
	1.	A reorganization of the original action-value format into a per-position structure:
	•	each FEN → list of all legal moves → per-move win probability (as provided in the upstream release).
	2.	An augmentation: for moves with predicted win probability 0% or 100%, we run Stockfish mate search and add a mate-in-N field per move (when a forced mate is detected).

Use cases
	•	Learning a policy/value model with richer per-position action structure
	•	Studying calibration of win-probability vs. mate-depth
	•	Training models to predict “mate imminence” signals beyond saturated winrates

Source Dataset (Upstream)
	•	Upstream: DeepMind “searchless chess” / ChessBench action-value release.
	•	Paper: “Grandmaster-Level Chess Without Search” (see upstream).
	•	Upstream license note: some portions are CC0 (lichess), remainder is CC BY 4.0.

# Schema
Each row contains:
{                                                                                                                                      
      "fen": str,           # FEN string                                                                                                 
      "moves": List[str],   # Legal moves in UCI format                                                                                  
      "p_win": List[float], # Win probability per move for side-to-move (unchanged from source)                                                           
      "mate": List[int],    # Mate depth per move (new field)                                                                            
}

Mate Definition (IMPORTANT)

Define:
	•	Is mate_in measured in plies or full moves?
	•	Is it from the side-to-move’s perspective?
	•	How do you encode “mate for the losing side” (e.g., negative values vs. separate field)?

Example convention (common in engine analysis):
	•	mate_in > 0 means side-to-move can force mate in N plies
	•	mate_in < 0 means side-to-move is getting mated in |N| plies
	•	mate_in = null if no forced mate found within limits

Splits

List your actual splits (or explain if none):
	•	train: …
	•	validation: …
	•	test: …

If you did not create official splits, say so explicitly and provide a script or recommended split strategy.

How Mate-in-N Was Computed

Mate-depth labels were generated with Stockfish. Stockfish is GPLv3 software; you’re not distributing Stockfish itself here, but you should still document provenance and reproduce settings.  ￼

Please fill in:
	•	Stockfish version: (e.g., Stockfish 16 / 17 / dev build)
	•	Command/config:
	•	Threads: __
	•	Hash: __
	•	Limit type: (nodes / depth / time): __
	•	Skill level (if any): __
	•	Syzygy tablebases (if any): __
	•	Criteria for “mate found”:
	•	Did you require the engine to report a mate score at PV root?
	•	How did you handle ambiguous cases (e.g., mate not found within limit)?

Exact trigger condition

State what “winrate 0%/100%” means in your code:
	•	Exactly 0.0 or 1.0?
	•	Or <= eps / >= 1-eps?

This affects label stability.

Recommended Citations

If you use this dataset, please cite:
	•	The upstream DeepMind paper/repo (action-value data source).  ￼
	•	This derived dataset (Zenodo DOI or HF citation if you add it).

(You can add BibTeX blocks here.)

License

License for this dataset repository: CC BY 4.0.

Rationale:
	•	This dataset redistributes an adapted form of the upstream data, which is licensed as: some portions CC0 (lichess), remainder CC BY 4.0, per DeepMind’s release.  ￼

Attribution & changes
	•	Upstream attribution: DeepMind “searchless chess” action-value release.
	•	Changes: reorganization into per-FEN legal-move lists + mate-in-N augmentation via Stockfish.

Limitations / Known Issues
	•	Mate-in-N labels depend on Stockfish settings and search limits; “mate not found” does not imply “no forced mate exists.”
	•	If winrates are quantized or clipped upstream, exact 0/1 may reflect saturation rather than certainty.
	•	Legal move generation and UCI formatting must match engine rules; verify consistency on underpromotions, castling rights, en passant, etc.
