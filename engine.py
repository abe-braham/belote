from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Iterable
import random


SUITS = ("S", "H", "D", "C")
RANKS = ("A", "10", "K", "Q", "J", "9", "8", "7")

TRUMP_ORDER = ("J", "9", "A", "10", "K", "Q", "8", "7")
NONTRUMP_ORDER = ("A", "10", "K", "Q", "J", "9", "8", "7")

TRUMP_VALUES = {"J": 20, "9": 14, "A": 11, "10": 10, "K": 4, "Q": 3, "8": 0, "7": 0}
NONTRUMP_VALUES = {"A": 11, "10": 10, "K": 4, "Q": 3, "J": 2, "9": 0, "8": 0, "7": 0}


@dataclass(frozen=True)
class Card:
    suit: str
    rank: str

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"


def team_of(player: int) -> int:
    return 0 if player % 2 == 0 else 1


def partner_of(player: int) -> int:
    return (player + 2) % 4


def all_cards() -> List[Card]:
    return [Card(s, r) for s in SUITS for r in RANKS]


def card_value(card: Card, trump: str) -> int:
    if card.suit == trump:
        return TRUMP_VALUES[card.rank]
    return NONTRUMP_VALUES[card.rank]


def trump_rank(card: Card) -> int:
    return TRUMP_ORDER.index(card.rank)


def nontrump_rank(card: Card) -> int:
    return NONTRUMP_ORDER.index(card.rank)


def trick_winner(trick: List[Tuple[int, Card]], trump: str) -> int:
    lead_suit = trick[0][1].suit
    has_trump = any(c.suit == trump for _, c in trick)
    best_player, best_card = trick[0]

    def better(a: Card, b: Card) -> bool:
        if has_trump:
            if a.suit == trump and b.suit != trump:
                return True
            if a.suit != trump and b.suit == trump:
                return False
            if a.suit == trump and b.suit == trump:
                return trump_rank(a) < trump_rank(b)
            if a.suit == lead_suit and b.suit != lead_suit:
                return True
            if a.suit != lead_suit and b.suit == lead_suit:
                return False
            if a.suit == lead_suit and b.suit == lead_suit:
                return nontrump_rank(a) < nontrump_rank(b)
            return False

        if a.suit == lead_suit and b.suit != lead_suit:
            return True
        if a.suit != lead_suit and b.suit == lead_suit:
            return False
        if a.suit == lead_suit and b.suit == lead_suit:
            return nontrump_rank(a) < nontrump_rank(b)
        return False

    for p, c in trick[1:]:
        if better(c, best_card):
            best_player, best_card = p, c
    return best_player


def points_in_trick(trick: List[Tuple[int, Card]], trump: str) -> int:
    return sum(card_value(c, trump) for _, c in trick)


def round_pair(a: int, b: int) -> Tuple[int, int]:
    ra = a % 10
    rb = b % 10
    casse = ra in (5, 6, 7) or rb in (5, 6, 7)

    def up(x: int) -> int:
        return ((x + 9) // 10) * 10

    def normal(x: int) -> int:
        r = x % 10
        if r <= 4:
            return (x // 10) * 10
        return up(x)

    if casse:
        return up(a), up(b)
    return normal(a), normal(b)


def legal_plays(hand: List[Card], trick: List[Tuple[int, Card]], player: int, trump: str) -> List[Card]:
    if not trick:
        return list(hand)

    lead_suit = trick[0][1].suit
    partner = partner_of(player)
    current_winner = trick_winner(trick, trump)
    partner_winning = team_of(current_winner) == team_of(partner)

    same_suit = [c for c in hand if c.suit == lead_suit]
    if same_suit:
        if lead_suit == trump:
            trumps_in_trick = [c for _, c in trick if c.suit == trump]
            best_trump = min(trumps_in_trick, key=trump_rank)
            higher = [c for c in same_suit if trump_rank(c) < trump_rank(best_trump)]
            return higher if higher else same_suit
        return same_suit

    trumps = [c for c in hand if c.suit == trump]
    if not trumps:
        return list(hand)

    trumps_in_trick = [c for _, c in trick if c.suit == trump]
    if not trumps_in_trick:
        return trumps

    best_trump = min(trumps_in_trick, key=trump_rank)
    higher = [c for c in trumps if trump_rank(c) < trump_rank(best_trump)]
    if higher and not partner_winning:
        return higher
    if partner_winning:
        return list(hand)
    return trumps


def cards_from_str(s: str) -> Card:
    s = s.strip().upper()
    if len(s) < 2:
        raise ValueError(s)
    suit = s[-1]
    rank = s[:-1]
    if suit not in SUITS or rank not in RANKS:
        raise ValueError(s)
    return Card(suit, rank)


def deck_shuffle(rng: random.Random) -> List[Card]:
    d = all_cards()
    rng.shuffle(d)
    return d


def bid_order(start: int) -> Iterable[int]:
    for i in range(4):
        yield (start + i) % 4


class Hand:
    def __init__(self, dealer: int, rng: random.Random):
        self.dealer = dealer
        self.rng = rng
        self.hands: List[List[Card]] = [[] for _ in range(4)]
        self.initial_hands: List[List[Card]] = [[] for _ in range(4)]
        self.bids: List[Tuple[int, Dict]] = []
        self.contract: Optional[Dict] = None
        self.multiplier: int = 1
        self.tricks: List[List[Tuple[int, Card]]] = []
        self.current_trick: List[Tuple[int, Card]] = []
        self.void_suits: List[set] = [set() for _ in range(4)]

    def deal(self) -> None:
        d = deck_shuffle(self.rng)
        for i in range(4):
            self.hands[i] = d[i * 8 : (i + 1) * 8]
            self.initial_hands[i] = list(self.hands[i])

    def public_state(self, player: int) -> Dict:
        return {
            "dealer": self.dealer,
            "player": player,
            "hand": list(self.hands[player]),
            "bids": list(self.bids),
            "contract": dict(self.contract) if self.contract else None,
            "multiplier": self.multiplier,
            "tricks": [[(p, c) for p, c in t] for t in self.tricks],
            "current_trick": [(p, c) for p, c in self.current_trick],
            "void_suits": [set(v) for v in self.void_suits],
        }

    def update_void(self, lead_suit: str, player: int, card: Card) -> None:
        if card.suit != lead_suit:
            self.void_suits[player].add(lead_suit)

    def play_card(self, player: int, card: Card, trump: str) -> None:
        if card not in self.hands[player]:
            raise ValueError("card not in hand")
        legal = legal_plays(self.hands[player], self.current_trick, player, trump)
        if card not in legal:
            raise ValueError("illegal play")
        if self.current_trick:
            lead_suit = self.current_trick[0][1].suit
            self.update_void(lead_suit, player, card)
        self.hands[player].remove(card)
        self.current_trick.append((player, card))

    def finish_trick(self, trump: str) -> int:
        if len(self.current_trick) != 4:
            raise ValueError("trick not complete")
        winner = trick_winner(self.current_trick, trump)
        self.tricks.append(self.current_trick)
        self.current_trick = []
        return winner

    def belote_team(self, trump: str) -> Optional[int]:
        for p in range(4):
            ranks = {c.rank for c in self.initial_hands[p] if c.suit == trump}
            if "K" in ranks and "Q" in ranks:
                return team_of(p)
        return None

    def raw_trick_points(self, trump: str) -> Tuple[int, int]:
        t = [0, 0]
        for trick in self.tricks:
            pts = points_in_trick(trick, trump)
            winner = trick_winner(trick, trump)
            t[team_of(winner)] += pts
        if self.tricks:
            last_winner = trick_winner(self.tricks[-1], trump)
            t[team_of(last_winner)] += 10
        return t[0], t[1]

    def tricks_won(self) -> Tuple[int, int]:
        w = [0, 0]
        for trick in self.tricks:
            winner = trick_winner(trick, self.contract["trump"])
            w[team_of(winner)] += 1
        return w[0], w[1]

    def score(self) -> Tuple[int, int]:
        if not self.contract:
            return 0, 0

        trump = self.contract["trump"]
        bidder_team = self.contract["bidder_team"]
        capot_bid = bool(self.contract.get("capot", False))
        contract_value = self.contract.get("value", None)

        belote_team = self.belote_team(trump)
        belote_bonus = [0, 0]
        if belote_team is not None:
            belote_bonus[belote_team] = 20

        won0, won1 = self.tricks_won()
        swept_team = None
        if won0 == 8:
            swept_team = 0
        elif won1 == 8:
            swept_team = 1

        if capot_bid:
            if swept_team == bidder_team:
                base = [0, 0]
                base[bidder_team] = 500
            else:
                base = [0, 0]
                base[1 - bidder_team] = 500
            base = [x * self.multiplier for x in base]
            return base[0] + belote_bonus[0], base[1] + belote_bonus[1]

        if swept_team is not None:
            base = [0, 0]
            base[swept_team] = 250
            base = [x * self.multiplier for x in base]
            return base[0] + belote_bonus[0], base[1] + belote_bonus[1]

        raw0, raw1 = self.raw_trick_points(trump)
        r0, r1 = round_pair(raw0, raw1)
        if contract_value is None:
            contract_value = 90

        base = [0, 0]
        if bidder_team == 0:
            bidder_points = r0
        else:
            bidder_points = r1

        if bidder_points >= contract_value:
            base[0] = r0
            base[1] = r1
        else:
            total = r0 + r1
            base[bidder_team] = 0
            base[1 - bidder_team] = total

        base = [x * self.multiplier for x in base]
        return base[0] + belote_bonus[0], base[1] + belote_bonus[1]
