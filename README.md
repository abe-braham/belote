# Belote Bots (work in progress)

## 1) Goal

We compare two approaches for playing Belote Contrée:

- a probability-driven heuristic bot that tries to choose actions with the best expected value
- a reinforcement learning bot (early version) that learns a policy for card play from self-play and score feedback

The main question is: if we push both approaches as far as we reasonably can, which one performs better and why?

Evaluation is currently done by running many hands and measuring score differential over time.

## 2) Rules

The game uses a 32-card deck and 4 players.

Card ranking / values:

- Trump suit: J (20), 9 (14), A (11), 10 (10), K (4), Q (3), 8 (0), 7 (0)
- Non-trump suit: A (11), 10 (10), K (4), Q (3), J (2), 9 (0), 8 (0), 7 (0)

Bidding:

- contracts: 90..160 by 10 in any suit, or Capot
- bidding starts with the player to the dealer’s right and proceeds clockwise
- ends after 3 passes following a bid, or after Capot, or after Contre (optionally Sur-contre)
- Contre doubles, Sur-contre quadruples

Play:

- player to dealer’s right leads the first trick
- follow suit if possible
- if void in suit, play trump if possible
- if trump has been played, overtrump if possible unless your partner is currently winning (then you may discard)

Belote:

- if a player holds K+Q of trump, the team receives +20 points (awarded even if the contract is lost)

Scoring:

- trick points sum to 152 + 10 for last trick = 162
- rounding is done to the nearest 10
- if the bidding team makes the contract, both teams score their rounded totals
- if the bidding team fails, it scores 0 and the defense receives the rounded total for the hand
- Capot:
  - if Capot was bid and succeeds: 500 points
  - if Capot was bid and fails: opponents get 500 points
  - if Capot was not bid but a team takes all 8 tricks: 250 points
- Contre / Sur-contre multiply the contract outcome

## 3) Current status, how to run, and next steps

### What’s implemented today

Heuristic bot (`bots.HeuristicBot`):

- Monte Carlo determinization for hidden hands consistent with public info
- rollouts for each legal play to estimate expected score differential
- bidding is heuristic + light simulation to pick a contract and decide on Contre/Sur-contre

RL bot (early) (`rl.RLAgent`, `rl.RLBot`):

- a small policy network over a simple feature encoding
- action masking so it only picks legal cards
- a minimal REINFORCE-style training loop (`--train-rl`)
- bidding is still heuristic-ish (very basic); the learning is focused on card play

### Running matches

Install:

```bash
pip install -r requirements.txt
```

Run a quick evaluation:

```bash
python main.py --hands 300 --team0 heuristic --team1 greedy
```

Try the (untrained) RL bot:

```bash
python main.py --hands 200 --team0 rl --team1 greedy
```

Train the RL bot (WIP):

```bash
python main.py --train-rl --updates 200 --games-per-update 64 --model-path models/rl_policy.pt
```

Then evaluate the trained model:

```bash
python main.py --hands 300 --team0 rl --team1 heuristic --model-path models/rl_policy.pt
```

### Metrics used for evaluation

Current default reporting is:

- total team0 score, total team1 score
- score differential (team0 - team1) over N hands

Planned extensions:

- contract success rate by level and trump suit
- distribution of bids (aggression) and Contre/Sur-contre frequency
- trick-level metrics (how often a bot wins with minimal overtrumps, how often it gives away 10/A, etc.)

### Next steps

- Improve the RL training loop with PPO-style update and entropy regularization.
- Make bidding and play evaluation faster and more stable.
- Add proper experiment logging with JSONL and a repeatable benchmark suite with fixed seeds.
- Add a third baseline such as an imperfect-information tree search bot.
