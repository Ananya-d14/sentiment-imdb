import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentiment.config import BATCH_SIZE, EPOCHS  # noqa: E402
from sentiment.train import train_all  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train sentiment models on IMDB.")
    parser.add_argument("--models", nargs="+", default=["nn", "cnn", "lstm"],
                        choices=["nn", "cnn", "lstm"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--no-early-stop", action="store_true")
    args = parser.parse_args()

    results = train_all(
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stop=not args.no_early_stop,
        which=tuple(args.models),
    )

    print("\n=== Final test accuracies ===")
    for name, r in results.items():
        print(f"  {name:>5}: {r.test_accuracy:.4f}  (train: {r.train_seconds:.1f}s)")


if __name__ == "__main__":
    main()
