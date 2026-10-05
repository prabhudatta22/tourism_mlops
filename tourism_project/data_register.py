"""Data registration: validate the raw dataset in tourism_project/data and print a summary."""
import sys
from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups", "ProductPitched",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def main() -> None:
    if not DATA_PATH.exists():
        sys.exit(f"ERROR: dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # 1) schema check: every expected column must be present
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: missing expected columns: {missing}")
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    # 2) target check: ProdTaken must be binary 0/1
    if not set(df["ProdTaken"].dropna().unique()) <= {0, 1}:
        sys.exit("ERROR: target ProdTaken is not binary")

    # 3) summary
    print("=" * 60)
    print("DATASET REGISTERED:", DATA_PATH.name)
    print("=" * 60)
    print(f"Rows x Columns     : {df.shape[0]} x {df.shape[1]}")
    print(f"Expected columns   : all {len(EXPECTED_COLUMNS)} present")
    print(f"Extra columns      : {extra or 'none'}")
    print(f"Duplicate rows     : {df.duplicated().sum()}")
    print(f"Duplicate customers: {df['CustomerID'].duplicated().sum()}")
    print(f"Missing values     : {int(df.isna().sum().sum())}")
    print(f"Target balance     : {df['ProdTaken'].value_counts(normalize=True).round(3).to_dict()}")
    print("\nColumn types:")
    print(df[EXPECTED_COLUMNS].dtypes.to_string())
    print("\nNumeric summary:")
    print(df[EXPECTED_COLUMNS].describe().T.round(2).to_string())
    print("\nValidation PASSED")


if __name__ == "__main__":
    main()
