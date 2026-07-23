import numpy as np


class InformationSet:
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.regret_sum = np.zeros(n_actions)
        self.strategy_sum = np.zeros(n_actions)

    def get_strategy(self, realization_weight: float) -> np.ndarray:
        positive_regrets = np.maximum(self.regret_sum, 0)
        total = positive_regrets.sum()
        if total > 0:
            strategy = positive_regrets / total
        else:
            strategy = np.full(self.n_actions, 1 / self.n_actions)
        self.strategy_sum += realization_weight * strategy
        return strategy

    def get_average_strategy(self) -> np.ndarray:
        total = self.strategy_sum.sum()
        if total > 0:
            return self.strategy_sum / total
        return np.full(self.n_actions, 1 / self.n_actions)


class Trainer:
    def __init__(self, game):
        self.game = game
        self.node_map: dict[str, InformationSet] = {}

    def get_node(self, key: str, n_actions: int) -> InformationSet:
        if key not in self.node_map:
            self.node_map[key] = InformationSet(n_actions)
        return self.node_map[key]

    def cfr(self, state, p0: float, p1: float) -> float:
        game = self.game

        if game.is_terminal(state):
            return game.payoff(state)

        if game.is_chance(state):
            return sum(
                prob * self.cfr(child, p0 * prob, p1 * prob)
                for child, prob in game.chance_outcomes(state)
            )

        player = game.current_player(state)
        acts = game.actions(state)
        key = game.infoset_key(state)
        n = game.num_actions(state)
        node = self.get_node(key, n)
        strategy = node.get_strategy(p0 if player == 0 else p1)

        util0 = [0.0] * n
        for a in range(n):
            child = game.apply(state, acts[a])
            if player == 0:
                util0[a] = self.cfr(child, p0 * strategy[a], p1)
            else:
                util0[a] = self.cfr(child, p0, p1 * strategy[a])

        node_util0 = sum(strategy[a] * util0[a] for a in range(n))

        sign = 1 if player == 0 else -1
        opp_reach = p1 if player == 0 else p0
        for a in range(n):
            v_a = sign * util0[a]
            node_v = sign * node_util0
            node.regret_sum[a] += opp_reach * (v_a - node_v)

        return node_util0

    def train(self, iterations: int) -> float:
        total = 0.0
        for _ in range(iterations):
            total += self.cfr(self.game.initial_state(), 1.0, 1.0)
        return total / iterations

    def average_strategy(self) -> dict:
        return {key: node.get_average_strategy() for key, node in self.node_map.items()}
