import pytest

from core.cfr import Trainer
from core.exploitability import exploitability
from leduc.game import LeducGame, LeducState


@pytest.mark.parametrize(
    "state,expected",
    [
        # P0 bets round1, P1 folds -> P0 wins P1's ante
        (LeducState(p0_card=2, p1_card=0, board=None, round=1, history="bf", contributed=(3, 1)), 1),
        # P0 checks, P1 bets, P0 folds -> P0 loses its ante
        (LeducState(p0_card=2, p1_card=0, board=None, round=1, history="kbf", contributed=(1, 3)), -1),
        # Round-2 showdown, P0 has higher rank (K vs J), no pair
        (LeducState(p0_card=2, p1_card=0, board=1, round=2, history="kc", contributed=(3, 3)), 3),
        # Round-2 showdown, P0 pairs the board and wins despite a lower "raw" rank
        (LeducState(p0_card=0, p1_card=2, board=3, round=2, history="kc", contributed=(3, 3)), 3),
        # Round-2 showdown, equal ranks, neither pairs -> split pot
        (LeducState(p0_card=0, p1_card=3, board=2, round=2, history="kc", contributed=(3, 3)), 0),
    ],
)
def test_payoff(state, expected):
    game = LeducGame()
    assert game.payoff(state) == expected


def test_convergence():
    game = LeducGame()
    trainer = Trainer(game)

    trainer.train(1000)
    e1 = exploitability(game, trainer.average_strategy())

    trainer.train(19000)
    e2 = exploitability(game, trainer.average_strategy())

    assert e2 < 0.1
    assert e2 < e1
