from pathlib import Path

from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    dataset = fetch_ucirepo(id=563)
    x = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    frame = x.copy()
    frame["churn"] = y.iloc[:, 0]
    out = RAW / "iranian_churn.csv"
    frame.to_csv(out, index=False)
    print(f"Saved {len(frame):,} customer rows to {out}")


if __name__ == "__main__":
    main()
