from abc import ABC, abstractmethod


class Game(ABC):
    @abstractmethod
    def initial_state(self): ...                 # root state (a chance node: the deal)
    @abstractmethod
    def is_terminal(self, state) -> bool: ...
    @abstractmethod
    def is_chance(self, state) -> bool: ...
    @abstractmethod
    def current_player(self, state) -> int: ...  # 0 or 1; decision nodes only
    @abstractmethod
    def actions(self, state) -> list: ...        # legal actions; decision nodes only
    @abstractmethod
    def apply(self, state, action): ...          # decision-node transition -> new state
    @abstractmethod
    def chance_outcomes(self, state) -> list: ...# list of (child_state, prob); probs sum to 1
    @abstractmethod
    def payoff(self, state) -> float: ...         # value to player 0; terminal only
    @abstractmethod
    def infoset_key(self, state) -> str: ...      # acting player's view; MUST exclude opponent's private card
    @abstractmethod
    def num_actions(self, state) -> int: ...      # == len(actions(state)); sizes regret/strategy arrays
