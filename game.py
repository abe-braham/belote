from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import random

import engine


def _all_bid_actions(min_value: int = 90) -> List[Dict]:
    out = []
    for v in (90, 100, 110, 120, 130, 140, 150, 160):
        if v < min_value:
            continue
        for s in engine.SUITS:
            out.append({"type": "bid", "trump": s, "value": v})
    for s in engine.SUITS:
        out.append({"type": "capot", "trump": s})
    return out


def _legal_bid_actions(player: int, current: Optional[Dict], multiplier: int) -> List[Dict]:
    actions = [{"type": "pass"}]
    if current is None:
        actions.extend(_all_bid_actions(90))
        return actions

    if current.get("capot", False):
        if multiplier == 1 and engine.team_of(player) != current["bidder_team"]:
            actions.append({"type": "contre"})
        return actions

    v = current.get("value", 0)
    actions.extend(_all_bid_actions(v + 10))
    if multiplier == 1 and engine.team_of(player) != current["bidder_team"]:
        actions.append({"type": "contre"})
    return actions


def _pick_action(bot, obs: Dict, actions: List[Dict]) -> Dict:
    a = bot.choose_bid(obs, list(actions))
    if not isinstance(a, dict):
        return actions[0]
    for x in actions:
        if a.get("type") == x.get("type") and a.get("trump") == x.get("trump") and a.get("value") == x.get("value"):
            return x
    return actions[0]


def run_bidding(hand: engine.Hand, bots: List, rng: random.Random) -> Tuple[Optional[Dict], int]:
    start = (hand.dealer - 1) % 4
    turn = start
    current = None
    multiplier = 1
    pass_count = 0

    while True:
        actions = _legal_bid_actions(turn, current, multiplier)
        obs = {"phase": "bidding", **hand.public_state(turn), "current_contract": dict(current) if current else None}
        action = _pick_action(bots[turn], obs, actions)
        hand.bids.append((turn, dict(action)))

        if action["type"] == "pass":
            if current is None:
                pass_count += 1
                if pass_count >= 4:
                    return None, 1
            else:
                pass_count += 1
                if pass_count >= 3:
                    return current, multiplier
        elif action["type"] == "bid":
            current = {
                "trump": action["trump"],
                "value": action["value"],
                "capot": False,
                "bidder_player": turn,
                "bidder_team": engine.team_of(turn),
            }
            pass_count = 0
        elif action["type"] == "capot":
            current = {
                "trump": action["trump"],
                "value": None,
                "capot": True,
                "bidder_player": turn,
                "bidder_team": engine.team_of(turn),
            }
            opp = [(turn + i) % 4 for i in (1, 3)]
            countered = False
            for p in opp:
                actions2 = [{"type": "pass"}, {"type": "contre"}]
                obs2 = {"phase": "bidding", **hand.public_state(p), "current_contract": dict(current)}
                a2 = _pick_action(bots[p], obs2, actions2)
                hand.bids.append((p, dict(a2)))
                if a2["type"] == "contre":
                    multiplier = 2
                    countered = True
                    break
            if countered:
                bidder = current["bidder_player"]
                actions3 = [{"type": "pass"}, {"type": "surcontre"}]
                obs3 = {"phase": "bidding", **hand.public_state(bidder), "current_contract": dict(current)}
                a3 = _pick_action(bots[bidder], obs3, actions3)
                hand.bids.append((bidder, dict(a3)))
                if a3["type"] == "surcontre":
                    multiplier = 4
            return current, multiplier
        elif action["type"] == "contre":
            multiplier = 2
            bidder = current["bidder_player"]
            actions3 = [{"type": "pass"}, {"type": "surcontre"}]
            obs3 = {"phase": "bidding", **hand.public_state(bidder), "current_contract": dict(current)}
            a3 = _pick_action(bots[bidder], obs3, actions3)
            hand.bids.append((bidder, dict(a3)))
            if a3["type"] == "surcontre":
                multiplier = 4
            return current, multiplier

        turn = (turn + 1) % 4


def play_hand(bots: List, dealer: int, seed: int = 0) -> Tuple[int, int, Optional[Dict]]:
    rng = random.Random(seed)
    hand = engine.Hand(dealer=dealer, rng=rng)
    hand.deal()
    contract, mult = run_bidding(hand, bots, rng)
    if contract is None:
        return 0, 0, None
    hand.contract = {
        "trump": contract["trump"],
        "value": contract["value"] if not contract.get("capot") else None,
        "capot": bool(contract.get("capot", False)),
        "bidder_team": contract["bidder_team"],
        "bidder_player": contract["bidder_player"],
    }
    if not hand.contract["capot"] and hand.contract["value"] is None:
        hand.contract["value"] = 90
    hand.multiplier = mult

    trump = hand.contract["trump"]
    leader = (dealer - 1) % 4
    for _ in range(8):
        for i in range(4):
            p = (leader + i) % 4
            obs = {"phase": "play", **hand.public_state(p), "leader": leader}
            legal = engine.legal_plays(hand.hands[p], hand.current_trick, p, trump)
            card = bots[p].choose_play(obs, legal)
            if card not in legal:
                card = legal[0]
            hand.play_card(p, card, trump)
        winner = hand.finish_trick(trump)
        leader = winner

    s0, s1 = hand.score()
    return s0, s1, hand.contract


def play_match(bots: List, hands: int = 200, seed: int = 0) -> Dict:
    rng = random.Random(seed)
    dealer = rng.randrange(4)
    total = [0, 0]
    for i in range(hands):
        s0, s1, _ = play_hand(bots, dealer=dealer, seed=rng.randrange(1_000_000_000))
        total[0] += s0
        total[1] += s1
        dealer = (dealer - 1) % 4
    diff = total[0] - total[1]
    return {"hands": hands, "team0": total[0], "team1": total[1], "diff": diff}
