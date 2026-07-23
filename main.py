import sys

from core.cfr import Trainer
from core.exploitability import exploitability
from kuhn.game import KuhnGame
from leduc.game import LeducGame

CARD_NAMES = {1: "J", 2: "Q", 3: "K"}

CHECKPOINTS = {
    "kuhn": [1000, 5000, 20000, 50000, 100000],
    "leduc": [1000, 5000, 20000, 50000, 100000, 200000, 500000],
}


def print_kuhn_strategies(avg):
    for key in sorted(avg):
        card = CARD_NAMES[int(key[0])]
        history = key[1:] or "(root)"
        strat = avg[key]
        print(f"{card} {history:<6}: p={strat[0]:.3f} b={strat[1]:.3f}")


def print_kuhn_interpretation(avg):
    p2_bluff_j = avg["1p"][1]
    p2_call_q = avg["2b"][1]
    p2_k_bet = avg["3p"][1]
    p2_k_call = avg["3b"][1]
    p2_j_fold = avg["1b"][0]
    alpha = avg["1"][1]
    p1_k_bet = avg["3"][1]
    p1_q_call = avg["2pb"][1]

    print("\nInterpretation (P2's strategy is unique; P1's is a family in alpha):")
    print(f"  P2 bluff-J when checked to : {p2_bluff_j:.3f} (expect ~0.333)")
    print(f"  P2 call-Q vs bet           : {p2_call_q:.3f} (expect ~0.333)")
    print(f"  P2 K bets when checked to  : {p2_k_bet:.3f} (expect ~1.0)")
    print(f"  P2 K calls vs bet          : {p2_k_call:.3f} (expect ~1.0)")
    print(f"  P2 J folds vs bet          : {p2_j_fold:.3f} (expect ~1.0)")
    print(f"  P1 alpha (bet-J at root)   : {alpha:.3f}")
    print(f"  P1 bet-K at root           : {p1_k_bet:.3f} (expect ~3*alpha = {3 * alpha:.3f})")
    print(f"  P1 call-Q vs bet           : {p1_q_call:.3f} (expect ~alpha+1/3 = {alpha + 1 / 3:.3f})")


def print_leduc_strategies(avg):
    for key in sorted(avg):
        probs = " ".join(f"{p:.3f}" for p in avg[key])
        print(f"{key:<20}: {probs}")


def main():
    game_name = sys.argv[1] if len(sys.argv) > 1 else "leduc"
    if game_name not in CHECKPOINTS:
        raise SystemExit(f"unknown game: {game_name!r} (expected 'kuhn' or 'leduc')")

    game = KuhnGame() if game_name == "kuhn" else LeducGame()
    trainer = Trainer(game)

    print(f"Training {game_name} with vanilla CFR...\n")
    print(f"{'iterations':>12}  {'exploitability':>14}")

    done = 0
    for checkpoint in CHECKPOINTS[game_name]:
        trainer.train(checkpoint - done)
        done = checkpoint
        avg = trainer.average_strategy()
        print(f"{checkpoint:>12}  {exploitability(game, avg):>14.6f}")

    avg = trainer.average_strategy()
    print("\nAverage strategy by information set:")
    if game_name == "kuhn":
        print_kuhn_strategies(avg)
        print_kuhn_interpretation(avg)
    else:
        print_leduc_strategies(avg)


if __name__ == "__main__":
    main()
