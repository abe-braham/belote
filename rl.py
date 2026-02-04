from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import random
import os

import torch
import torch.nn as nn
import torch.optim as optim

import engine
import bots
import game


CARD_LIST = engine.all_cards()
CARD_TO_IDX = {c: i for i, c in enumerate(CARD_LIST)}


def encode_obs(obs: Dict) -> torch.Tensor:
    hand = obs["hand"]
    trump = obs["contract"]["trump"]
    multiplier = obs["multiplier"]
    player = obs["player"]
    current = obs["current_trick"]
    contract_value = obs["contract"].get("value") or 0

    x = torch.zeros(74, dtype=torch.float32)
    for c in hand:
        x[CARD_TO_IDX[c]] = 1.0
    for _, c in current:
        x[32 + CARD_TO_IDX[c]] = 1.0
    x[64 + engine.SUITS.index(trump)] = 1.0
    x[68] = contract_value / 160.0
    x[69] = multiplier / 4.0
    x[70 + player] = 1.0
    return x


class PolicyNet(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(74, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RLAgent:
    def __init__(self, seed: int = 0, device: Optional[str] = None):
        self.rng = random.Random(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy = PolicyNet().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self._logprobs: List[Tuple[int, torch.Tensor]] = []

    def reset(self) -> None:
        self._logprobs = []

    def bot(self, player: int, train: bool = True):
        return RLBot(agent=self, player=player, train=train)

    def pick_card(self, obs: Dict, legal: List[engine.Card], train: bool) -> engine.Card:
        x = encode_obs(obs).to(self.device)
        logits = self.policy(x)
        mask = torch.full((32,), -1e9, device=self.device)
        legal_idx = [CARD_TO_IDX[c] for c in legal]
        mask[legal_idx] = 0.0
        logits = logits + mask
        if train:
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logp = dist.log_prob(a)
            self._logprobs.append((engine.team_of(obs["player"]), logp))
            return CARD_LIST[int(a.item())]
        idx = int(torch.argmax(logits).item())
        return CARD_LIST[idx]

    def finish_hand(self, score0: int, score1: int) -> torch.Tensor:
        diff = float(score0 - score1)
        if not self._logprobs:
            return torch.tensor(0.0, device=self.device)
        loss = 0.0
        for team, logp in self._logprobs:
            r = diff if team == 0 else -diff
            loss = loss + (-logp * r)
        return loss / len(self._logprobs)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy.load_state_dict(torch.load(path, map_location=self.device))


class RLBot(bots.Bot):
    def __init__(self, agent: RLAgent, player: int, train: bool = False):
        self.agent = agent
        self.player = player
        self.train = train

    def choose_bid(self, obs: Dict, actions: List[Dict]) -> Dict:
        current = obs.get("current_contract")
        if current is not None:
            for a in actions:
                if a.get("type") == "pass":
                    return a
        hand = obs["hand"]
        best_suit = None
        best = -1
        for s in engine.SUITS:
            v = sum(engine.TRUMP_VALUES[c.rank] for c in hand if c.suit == s)
            if v > best:
                best = v
                best_suit = s
        if best >= 35:
            for a in actions:
                if a.get("type") == "bid" and a.get("trump") == best_suit and a.get("value") == 90:
                    return a
        for a in actions:
            if a.get("type") == "pass":
                return a
        return actions[0]

    def choose_play(self, obs: Dict, legal: List[engine.Card]) -> engine.Card:
        return self.agent.pick_card(obs, legal, train=self.train)


def train_wip(
    updates: int = 200,
    hands_per_update: int = 64,
    seed: int = 0,
    save_path: str = "models/rl_policy.pt",
) -> None:
    agent = RLAgent(seed=seed)
    rng = random.Random(seed)
    dealer = rng.randrange(4)

    for u in range(updates):
        agent.optimizer.zero_grad()
        agent.reset()
        total_loss = 0.0
        total_reward = 0.0

        for _ in range(hands_per_update):
            agent.reset()
            b0 = agent.bot(0, train=True)
            b2 = agent.bot(2, train=True)
            opponents = [bots.GreedyBot(seed=rng.randrange(1_000_000)), bots.GreedyBot(seed=rng.randrange(1_000_000))]
            players = [b0, opponents[0], b2, opponents[1]]
            s0, s1, _ = game.play_hand(players, dealer=dealer, seed=rng.randrange(1_000_000_000))
            dealer = (dealer - 1) % 4
            total_reward += (s0 - s1)
            total_loss = total_loss + agent.finish_hand(s0, s1)

        total_loss.backward()
        agent.optimizer.step()
        if (u + 1) % 10 == 0:
            avg = total_reward / hands_per_update
            print(f"update {u+1} avg_diff {avg:.1f}")
    agent.save(save_path)
