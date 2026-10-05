"""Data preparation: clean the raw dataset and create train/test splits."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "tourism.csv"         # loaded directly from the repo data folder
OUT_DIR = BASE / "data" / "processed"              # uploaded as a workflow artifact
TARGET = "ProdTaken"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop identifier / index columns: they carry no predictive signal
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")] + ["CustomerID"])

    # Fix inconsistent category spelling
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    # Remove exact duplicate records
    df = df.drop_duplicates()

    # Impute any missing values (defensive: the current file has none)
    for col in df.columns:
        if df[col].isna().any():
            fill = df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else df[col].mode()[0]
            df[col] = df[col].fillna(fill)
    return df


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print("Raw shape    :", df.shape)
    df = clean(df)
    print("Cleaned shape:", df.shape)

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtrain.to_csv(OUT_DIR / "Xtrain.csv", index=False)
    Xtest.to_csv(OUT_DIR / "Xtest.csv", index=False)
    ytrain.to_csv(OUT_DIR / "ytrain.csv", index=False)
    ytest.to_csv(OUT_DIR / "ytest.csv", index=False)

    print(f"Train: {Xtrain.shape}  buyers={ytrain.mean():.3f}")
    print(f"Test : {Xtest.shape}  buyers={ytest.mean():.3f}")
    print("Saved splits to", OUT_DIR, "->", sorted(p.name for p in OUT_DIR.iterdir()))


if __name__ == "__main__":
    main()
