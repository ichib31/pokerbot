from itertools import permutations
from typing import NamedTuple, Optional, Tuple

from core.game import Game

TERMINAL_HISTORIES = {"pp", "bp", "bb", "pbp", "pbb"}
ACTIONS = ["p", "b"]


class KuhnState(NamedTuple):
    cards: Optional[Tuple[int, int]]  # None until dealt
    history: str


class KuhnGame(Game):
    def initial_state(self) -> KuhnState:
        return KuhnState(cards=None, history="")

    def is_terminal(self, state: KuhnState) -> bool:
        return state.cards is not None and state.history in TERMINAL_HISTORIES

    def is_chance(self, state: KuhnState) -> bool:
        return state.cards is None

    def current_player(self, state: KuhnState) -> int:
        return len(state.history) % 2

    def actions(self, state: KuhnState) -> list:
        return ACTIONS

    def apply(self, state: KuhnState, action: str) -> KuhnState:
        return KuhnState(cards=state.cards, history=state.history + action)

    def chance_outcomes(self, state: KuhnState) -> list:
        deals = list(permutations([1, 2, 3], 2))
        prob = 1 / len(deals)
        return [(KuhnState(cards=deal, history=""), prob) for deal in deals]

    def payoff(self, state: KuhnState) -> float:
        cards, history = state.cards, state.history
        higher0 = cards[0] > cards[1]
        if history == "pp":
            return 1 if higher0 else -1
        if history.endswith("bb"):
            return 2 if higher0 else -2
        if history == "bp":
            return 1
        # remaining terminal: "pbp"
        return -1

    def infoset_key(self, state: KuhnState) -> str:
        player = self.current_player(state)
        return str(state.cards[player]) + state.history

    def num_actions(self, state: KuhnState) -> int:
        return len(ACTIONS)
