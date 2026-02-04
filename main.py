from __future__ import annotations

import argparse
import random
import os

import bots
import game


def build_players(team0: str, team1: str, seed: int, model_path) -> list:
    rng = random.Random(seed)

    rl_agents = {}
    def make(seat: int, kind: str):
        if kind == "random":
            return bots.RandomBot(seed=rng.randrange(1_000_000_000))
        if kind == "greedy":
            return bots.GreedyBot(seed=rng.randrange(1_000_000_000))
        if kind == "heuristic":
            return bots.HeuristicBot(seed=rng.randrange(1_000_000_000))
        if kind == "rl":
            import rl
            team = 0 if seat % 2 == 0 else 1
            if team not in rl_agents:
                agent = rl.RLAgent(seed=rng.randrange(1_000_000_000))
                if model_path and os.path.exists(model_path):
                    agent.load(model_path)
                rl_agents[team] = agent
            return rl_agents[team].bot(seat, train=False)
        raise ValueError(kind)

    return [make(0, team0), make(1, team1), make(2, team0), make(3, team1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hands", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--team0", type=str, default="heuristic", choices=["random", "greedy", "heuristic", "rl"])
    p.add_argument("--team1", type=str, default="greedy", choices=["random", "greedy", "heuristic", "rl"])
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--train-rl", action="store_true")
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--games-per-update", type=int, default=64)
    args = p.parse_args()

    if args.train_rl:
        import rl
        rl.train_wip(updates=args.updates, hands_per_update=args.games_per_update, seed=args.seed, save_path=args.model_path or "models/rl_policy.pt")
        return

    players = build_players(args.team0, args.team1, args.seed, args.model_path)
    res = game.play_match(players, hands=args.hands, seed=args.seed)
    print(res)


if __name__ == "__main__":
    main()
