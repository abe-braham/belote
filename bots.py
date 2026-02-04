from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import random

import engine


CARD_LIST = engine.all_cards()
CARD_TO_IDX = {c: i for i, c in enumerate(CARD_LIST)}


def _win_power(card: engine.Card, lead_suit: str, trump: str, has_trump: bool) -> int:
    if has_trump:
        if card.suit != trump:
            return -1
        return 100 - engine.trump_rank(card)
    if card.suit != lead_suit:
        return -1
    return 100 - engine.nontrump_rank(card)


def winning_cards(player: int, legal: List[engine.Card], trick: List[Tuple[int, engine.Card]], trump: str) -> List[engine.Card]:
    if not trick:
        return []
    out = []
    for c in legal:
        w = engine.trick_winner(trick + [(player, c)], trump)
        if w == player:
            out.append(c)
    return out


def pick_weakest_winner(player: int, legal: List[engine.Card], trick: List[Tuple[int, engine.Card]], trump: str) -> Optional[engine.Card]:
    wins = winning_cards(player, legal, trick, trump)
    if not wins:
        return None
    lead = trick[0][1].suit
    scored = []
    for c in wins:
        t = trick + [(player, c)]
        has_trump = any(x.suit == trump for _, x in t)
        scored.append((_win_power(c, lead, trump, has_trump), c))
    scored.sort()
    return scored[0][1]


def lowest_discard(legal: List[engine.Card], trump: str) -> engine.Card:
    return min(legal, key=lambda c: (engine.card_value(c, trump), c.suit, CARD_TO_IDX[c]))


def lead_choice(legal: List[engine.Card], trump: str) -> engine.Card:
    nontrumps = [c for c in legal if c.suit != trump]
    if nontrumps:
        high = max(nontrumps, key=lambda c: engine.card_value(c, trump))
        if engine.card_value(high, trump) >= 10:
            return high
    return lowest_discard(legal, trump)


def rollout_policy(player: int, hand: List[engine.Card], trick: List[Tuple[int, engine.Card]], trump: str) -> engine.Card:
    legal = engine.legal_plays(hand, trick, player, trump)
    if trick:
        w = pick_weakest_winner(player, legal, trick, trump)
        if w is not None:
            return w
        return lowest_discard(legal, trump)
    return lead_choice(legal, trump)


def _played_counts(tricks: List[List[Tuple[int, engine.Card]]], current: List[Tuple[int, engine.Card]]) -> List[int]:
    c = [0, 0, 0, 0]
    for t in tricks:
        for p, _ in t:
            c[p] += 1
    for p, _ in current:
        c[p] += 1
    return c


def sample_hidden_hands(obs: Dict, rng: random.Random) -> Optional[List[List[engine.Card]]]:
    me = obs["player"]
    hand_me = list(obs["hand"])

    deck = set(engine.all_cards())
    seen = set(hand_me)
    for t in obs["tricks"]:
        for _, c in t:
            seen.add(c)
    for _, c in obs["current_trick"]:
        seen.add(c)

    unseen = [c for c in deck if c not in seen]
    counts = _played_counts(obs["tricks"], obs["current_trick"])
    remaining_slots = [8 - counts[p] for p in range(4)]
    remaining_slots[me] = len(hand_me)

    void_suits = obs.get("void_suits", [set() for _ in range(4)])

    attempts = 0
    while attempts < 40:
        attempts += 1
        hands = [[] for _ in range(4)]
        hands[me] = list(hand_me)
        slots = list(remaining_slots)
        pool = list(unseen)
        rng.shuffle(pool)
        ok = True
        for card in pool:
            eligible = [p for p in range(4) if p != me and slots[p] > 0 and card.suit not in void_suits[p]]
            if not eligible:
                ok = False
                break
            p = rng.choice(eligible)
            hands[p].append(card)
            slots[p] -= 1
        if ok and all(slots[p] == 0 for p in range(4)):
            return hands
    return None


def simulate_from_obs(obs: Dict, hands: List[List[engine.Card]], rng: random.Random, fixed_play: Optional[engine.Card] = None) -> Tuple[int, int]:
    contract = obs["contract"]
    trump = contract["trump"]
    multiplier = obs["multiplier"]
    dealer = obs["dealer"]

    tricks = [[(p, c) for p, c in t] for t in obs["tricks"]]
    current = [(p, c) for p, c in obs["current_trick"]]
    leader = obs["leader"]
    me = obs["player"]

    if fixed_play is not None:
        if fixed_play not in hands[me]:
            return 0, 0
        current.append((me, fixed_play))
        hands[me] = [c for c in hands[me] if c != fixed_play]

    while len(tricks) < 8:
        while len(current) < 4:
            p = (leader + len(current)) % 4
            card = rollout_policy(p, hands[p], current, trump)
            current.append((p, card))
            hands[p] = [c for c in hands[p] if c != card]
        winner = engine.trick_winner(current, trump)
        tricks.append(current)
        current = []
        leader = winner

    by_player = [[] for _ in range(4)]
    for t in tricks:
        for p, c in t:
            by_player[p].append(c)
    if any(len(by_player[p]) != 8 for p in range(4)):
        return 0, 0

    sim = engine.Hand(dealer=dealer, rng=rng)
    sim.initial_hands = [list(by_player[p]) for p in range(4)]
    sim.contract = dict(contract)
    sim.multiplier = multiplier
    sim.tricks = tricks
    return sim.score()


def _contract_fields(current: Optional[Dict]) -> Tuple[int, bool]:
    if not current:
        return 0, False
    if current.get("capot"):
        return 999, True
    return int(current.get("value", 0)), False


def _action_eq(a: Dict, b: Dict) -> bool:
    if a.get("type") != b.get("type"):
        return False
    if a.get("type") in ("bid", "capot"):
        return a.get("trump") == b.get("trump") and a.get("value") == b.get("value")
    return True


class Bot:
    def choose_bid(self, obs: Dict, actions: List[Dict]) -> Dict:
        raise NotImplementedError

    def choose_play(self, obs: Dict, legal: List[engine.Card]) -> engine.Card:
        raise NotImplementedError


class RandomBot(Bot):
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def choose_bid(self, obs: Dict, actions: List[Dict]) -> Dict:
        return self.rng.choice(actions)

    def choose_play(self, obs: Dict, legal: List[engine.Card]) -> engine.Card:
        return self.rng.choice(legal)


class GreedyBot(Bot):
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def choose_bid(self, obs: Dict, actions: List[Dict]) -> Dict:
        current = obs.get("current_contract")
        cur_value, cur_capot = _contract_fields(current)
        hand = obs["hand"]

        best_suit = None
        best_score = -1
        for s in engine.SUITS:
            score = sum(engine.TRUMP_VALUES[c.rank] for c in hand if c.suit == s)
            ranks = {c.rank for c in hand if c.suit == s}
            if "K" in ranks and "Q" in ranks:
                score += 12
            if score > best_score:
                best_score = score
                best_suit = s

        if cur_capot:
            for a in actions:
                if a.get("type") == "pass":
                    return a
            return actions[0]

        target = None
        if best_score >= 35:
            target = 90
            for v in (100, 110, 120, 130, 140, 150, 160):
                if best_score >= 35 + (v - 90) * 0.6:
                    target = v

        if target is not None:
            target = max(target, cur_value + 10 if cur_value else target)
            for a in actions:
                if a.get("type") == "bid" and a.get("trump") == best_suit and a.get("value") == target:
                    return a

        bids = [a for a in actions if a.get("type") == "bid"]
        if bids:
            higher = [a for a in bids if a.get("value", 0) > cur_value]
            if higher and best_score >= 42:
                return min(higher, key=lambda x: x["value"])
        for a in actions:
            if a.get("type") == "pass":
                return a
        return actions[0]

    def choose_play(self, obs: Dict, legal: List[engine.Card]) -> engine.Card:
        trump = obs["contract"]["trump"]
        p = obs["player"]
        trick = obs["current_trick"]
        if trick:
            w = pick_weakest_winner(p, legal, trick, trump)
            if w is not None:
                return w
            return lowest_discard(legal, trump)
        return lead_choice(legal, trump)


class HeuristicBot(Bot):
    def __init__(self, seed: int = 0, simulations_play: int = 28, simulations_bid: int = 16):
        self.rng = random.Random(seed)
        self.simulations_play = simulations_play
        self.simulations_bid = simulations_bid

    def _team_diff(self, obs: Dict, score0: int, score1: int) -> int:
        me_team = engine.team_of(obs["player"])
        return (score0 - score1) if me_team == 0 else (score1 - score0)

    def _ev_play(self, obs: Dict, card: engine.Card) -> float:
        total = 0.0
        n = 0
        for _ in range(self.simulations_play):
            hands = sample_hidden_hands(obs, self.rng)
            if hands is None:
                continue
            s0, s1 = simulate_from_obs(obs, hands, self.rng, fixed_play=card)
            total += self._team_diff(obs, s0, s1)
            n += 1
        return total / n if n else -1e9

    def choose_play(self, obs: Dict, legal: List[engine.Card]) -> engine.Card:
        if len(legal) == 1:
            return legal[0]
        best = None
        best_ev = -1e18
        for c in legal:
            ev = self._ev_play(obs, c)
            if ev > best_ev:
                best_ev = ev
                best = c
        return best if best is not None else legal[0]

    def _estimate_points_for_trump(self, hand: List[engine.Card], trump: str) -> float:
        pts = sum(engine.card_value(c, trump) for c in hand)
        ranks = {c.rank for c in hand if c.suit == trump}
        if "K" in ranks and "Q" in ranks:
            pts += 12
        length = sum(1 for c in hand if c.suit == trump)
        pts += max(0, length - 3) * 2.5
        return pts

    def _bid_candidates(self, current: Optional[Dict]) -> List[int]:
        if not current:
            return [90, 100, 110, 120, 130, 140, 150, 160]
        if current.get("capot"):
            return []
        v = int(current.get("value", 0))
        return [x for x in (90, 100, 110, 120, 130, 140, 150, 160) if x > v]

    def _simulate_bid_points(self, obs: Dict, trump: str, bidder_team: int) -> List[int]:
        dealer = obs["dealer"]
        me = obs["player"]
        my_hand = list(obs["hand"])
        results = []

        for _ in range(self.simulations_bid):
            d = engine.deck_shuffle(self.rng)
            used = set(my_hand)
            pool = [c for c in d if c not in used]
            self.rng.shuffle(pool)

            hands = [[] for _ in range(4)]
            hands[me] = list(my_hand)
            idx = 0
            for p in range(4):
                if p == me:
                    continue
                hands[p] = pool[idx : idx + 8]
                idx += 8

            sim = engine.Hand(dealer=dealer, rng=self.rng)
            sim.initial_hands = [list(h) for h in hands]
            sim.contract = {"trump": trump, "bidder_team": bidder_team, "capot": False, "value": 90}
            sim.multiplier = 1

            leader = (dealer - 1) % 4
            tricks = []
            current = []
            while len(tricks) < 8:
                while len(current) < 4:
                    p = (leader + len(current)) % 4
                    card = rollout_policy(p, hands[p], current, trump)
                    current.append((p, card))
                    hands[p] = [c for c in hands[p] if c != card]
                winner = engine.trick_winner(current, trump)
                tricks.append(current)
                current = []
                leader = winner

            sim.tricks = tricks
            raw0, raw1 = sim.raw_trick_points(trump)
            r0, r1 = engine.round_pair(raw0, raw1)
            results.append(r0 if bidder_team == 0 else r1)

        return results

    def _maybe_contre(self, obs: Dict, actions: List[Dict]) -> Optional[Dict]:
        contre = [a for a in actions if a.get("type") == "contre"]
        if not contre:
            return None
        current = obs.get("current_contract")
        if not current or current.get("capot"):
            return None
        trump = current["trump"]
        my_team = engine.team_of(obs["player"])
        bidder_team = current["bidder_team"]
        if my_team == bidder_team:
            return None
        strength = self._estimate_points_for_trump(obs["hand"], trump)
        if strength >= 60:
            return contre[0]
        return None

    def _maybe_surcontre(self, obs: Dict, actions: List[Dict]) -> Optional[Dict]:
        sur = [a for a in actions if a.get("type") == "surcontre"]
        if not sur:
            return None
        current = obs.get("current_contract")
        if not current:
            return None
        trump = current["trump"]
        strength = self._estimate_points_for_trump(obs["hand"], trump)
        if strength >= 70:
            return sur[0]
        return None

    def choose_bid(self, obs: Dict, actions: List[Dict]) -> Dict:
        m = self._maybe_surcontre(obs, actions)
        if m is not None:
            return m
        m = self._maybe_contre(obs, actions)
        if m is not None:
            return m

        current = obs.get("current_contract")
        my_team = engine.team_of(obs["player"])
        cur_value, cur_capot = _contract_fields(current)
        if cur_capot:
            for a in actions:
                if a.get("type") == "pass":
                    return a
            return actions[0]

        suits = []
        for s in engine.SUITS:
            suits.append((self._estimate_points_for_trump(obs["hand"], s), s))
        suits.sort(reverse=True)
        best_suit = suits[0][1]
        best_est = suits[0][0]

        capot_actions = [a for a in actions if a.get("type") == "capot" and a.get("trump") == best_suit]
        if capot_actions and best_est >= 88:
            return capot_actions[0]

        bid_values = self._bid_candidates(current)
        if not bid_values:
            for a in actions:
                if a.get("type") == "pass":
                    return a
            return actions[0]

        samples = self._simulate_bid_points(obs, best_suit, my_team)
        if not samples:
            for a in actions:
                if a.get("type") == "pass":
                    return a
            return actions[0]
        samples.sort()

        def p_success(v: int) -> float:
            ok = sum(1 for x in samples if x >= v)
            return ok / len(samples)

        chosen_value = None
        for v in reversed(bid_values):
            if v <= cur_value:
                continue
            if p_success(v) >= 0.6:
                chosen_value = v
                break

        if chosen_value is None:
            for a in actions:
                if a.get("type") == "pass":
                    return a
            return actions[0]

        desired = {"type": "bid", "trump": best_suit, "value": chosen_value}
        for a in actions:
            if _action_eq(a, desired):
                return a

        higher = [a for a in actions if a.get("type") == "bid" and a.get("value", 0) > cur_value]
        if higher:
            same = [a for a in higher if a.get("trump") == best_suit]
            if same:
                return min(same, key=lambda x: x["value"])
            return min(higher, key=lambda x: x["value"])

        for a in actions:
            if a.get("type") == "pass":
                return a
        return actions[0]
