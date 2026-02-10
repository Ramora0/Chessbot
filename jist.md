I am attempting to train a multimodal chess LLM that fundamentally understands chess and can play at a high level. I've trained an 80m chess transformer to reasonable playing strength (~2300 elo in lichess puzzles, substantially higher playing strength) one shot. I've done some initial experiments training a projector, but they haven't gone well. Here are the details of both the chess model and the projector experiments:

# Chess Model
I train on 500 million chess positions where every legal move is played, then 50ms stockfish is run (on every move) to determine a win% for that move. In addition, we extract the mate-in-n information, so when the winrate is 0% or 100% we know the number of moves to mate. We train on many auxilary objectives. Unless otherwise noted, each objective is attention pooled from the full 70 token sequence to produce a vector to apply our loss on.
1. Move winrates
We apply three losses on a pooled vector: BCE, CE, and hinge
First, we go from 768 -> 1968, the number of possible moves, and produce logits for every possible move. We sigmoid these and apply BCE loss to target the win% of every legal move, giving the model an understanding not only which moves are good but which are bad and by how much.
Second, to encourage the model to focus on getting the top ranking right, we apply softmax on the unsigmoided versions of the TARGET winrates to create a target distribution, and apply softmax on our own loss and apply CE to match this target distribution so we can get the correct moves right.
Third, we apply an annealed hinge loss on all illegal moves to enocurage their logits to go smaller (to -5) so down the line an external model can learn to ignore them.
2. Winrate
We train a value head which classifies the position into 128 win-value buckets (we train CE against a relaxed version of the distribution given the target winrate) so the model gets a sense of how the position is doing.
3. Reconstruction
On each of the 70 output hidden states, we apply a simple linear layer and ensure the model can reconstruct the input so later down the line another model at least has access to this.
4. Control
On each of the 64 board tokens, we apply a loss so the model predicts two things; the number of white and black attackers on a given square, so it learns control maps of the board. We apply a simple MSE loss on the target control and the generated logit.

# Projector Experiments
Projector experiments did not go well. We freeze the encoder and LLM during all fine tuning. We used a SOTA 8b model.
### QA Generation
We generate a set of 50,000 questions from random chess positions. We ask 10 categories of questions, each with 2-3 variants. These are questions like "What piece is on e4?" "Is white in check?" "What is the best move?". Some of these just ask about the position, while other ask specifically about our model's outputs, like who is winning and what is the best move.
### Experiment 1
We tried a 768 -> 2048 -> 4096 mlp to apply to each of the final 70 hidden states. This produced extremely large magnitudes, avg of around 90 (Qwen's was ~1.2). The model would answer extremely briefly, often very incorrect, and would oose all ability to produce longer answers about anything.
### Experiment 2
We added a layernorm after targetting Qwen's embedding distribution, which hurt loss a little bit, but when we did inference the model had no idea a chess position was provided; it would often say "they asked about what is on e4, but for that I would need to see the board position".
### Experiment 3
We switched to a multi-layer multi-head cross-attention scheme of 16 tokens. It worked slightly better; training and validation loss were worse, but the outputs were semi-reasonable, but still wrong a majority of the time.
### Experiment 4
We tried taking out all remotely difficult questions and were left with just reconstruction-level questions, and the model was still not able to reconstruct the board state.

# Final Thoughts
I have several ideas:
1. The original encoder is not wide enough; its loss is dominated by "find the best move", which doesn't produce very helpful outputs in teh final layer; we need to add better, more fundamental objectives to the encoder, or take hidden states from the n-1 layer.
2. The outputs of the encoder are horrible for the LLM, and are nearly impossible to translate well; apply some sort of contrastive loss on the model during training to get a better embedding space.
3. The projector is not powerful enough; the encoder has learned good representations, but its very difficult to get them to align meaningfully with the LLM's space; we need to get a more powerful projector, and maybe more and better diversified data.

Which fix does the data best support? What would you guess is going on?