import pytest

from core.cfr import Trainer
from core.exploitability import exploitability
from kuhn.game import KuhnGame, KuhnState

GAME_VALUE = -1 / 18


@pytest.mark.parametrize(
    "cards,history,expected",
    [
        ([3, 2], "pp", 1),    # K vs Q showdown, P0 higher
        ([2, 3], "pp", -1),   # Q vs K showdown, P0 lower
        ([3, 2], "bb", 2),    # K vs Q, bet+call showdown, P0 higher
        ([2, 3], "bb", -2),   # Q vs K, bet+call showdown, P0 lower
        ([3, 2], "bp", 1),    # P0 bets, P1 folds -> P0 wins
        ([2, 3], "bp", 1),    # P0 bets weaker card, P1 folds anyway -> P0 wins
        ([2, 3], "pbp", -1),  # P0 checks, P1 bets, P0 folds -> P1 wins
        ([3, 2], "pbp", -1),  # P0 folds regardless of card strength
    ],
)
def test_payoff(cards, history, expected):
    game = KuhnGame()
    state = KuhnState(cards=tuple(cards), history=history)
    assert game.payoff(state) == expected


def test_convergence():
    trainer = Trainer(KuhnGame())
    game_value = trainer.train(50000)
    assert game_value == pytest.approx(GAME_VALUE, abs=0.01)


def test_exploitability():
    trainer = Trainer(KuhnGame())
    trainer.train(50000)
    avg = trainer.average_strategy()
    assert exploitability(trainer.game, avg) < 0.01
