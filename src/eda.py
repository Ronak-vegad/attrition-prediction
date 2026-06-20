from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def find_dataset(data_dir: Path) -> Path:
    candidates = [
        data_dir / "HR-Employee-Attrition.csv",
        data_dir / "WA_Fn-UseC_-HR-Employee-Attrition.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "CSV not found. Put the dataset at data/HR-Employee-Attrition.csv "
        "(or data/WA_Fn-UseC_-HR-Employee-Attrition.csv)."
    )


def save_countplot(df: pd.DataFrame, out_dir: Path, col: str, fname: str) -> None:
    plt.figure(figsize=(8, 4))
    ax = sns.countplot(data=df, x=col, hue="Attrition")
    ax.set_title(f"{col} by Attrition")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=200)
    plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    out_dir = project_root / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    csv_path = find_dataset(data_dir)
    print("Using dataset:", csv_path)
    df = pd.read_csv(csv_path)

    # Attrition rate
    plt.figure(figsize=(5, 4))
    ax = sns.countplot(data=df, x="Attrition")
    ax.set_title("Attrition class distribution")
    for c in ax.containers:
        ax.bar_label(c)
    plt.tight_layout()
    plt.savefig(out_dir / "attrition_rate.png", dpi=200)
    plt.close()

    # Demographics / job attributes
    for col, fname in [
        ("Gender", "attrition_by_gender.png"),
        ("MaritalStatus", "attrition_by_marital_status.png"),
        ("Department", "attrition_by_department.png"),
        ("JobRole", "attrition_by_jobrole.png"),
    ]:
        save_countplot(df, out_dir, col, fname)

    # Numeric distributions
    num_cols = [
        "Age",
        "DistanceFromHome",
        "MonthlyIncome",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
    ]
    for col in [c for c in num_cols if c in df.columns]:
        plt.figure(figsize=(7, 4))
        ax = sns.boxplot(data=df, x="Attrition", y=col)
        ax.set_title(f"{col} by Attrition")
        plt.tight_layout()
        plt.savefig(out_dir / f"box_{col}.png", dpi=200)
        plt.close()

    # OverTime deep dive
    if "OverTime" in df.columns:
        plt.figure(figsize=(5, 4))
        ax = sns.countplot(data=df, x="OverTime", hue="Attrition")
        ax.set_title("OverTime vs Attrition")
        plt.tight_layout()
        plt.savefig(out_dir / "overtime_attrition.png", dpi=200)
        plt.close()

        ct = pd.crosstab(df["OverTime"], df["Attrition"], normalize="index")
        ct.to_csv(out_dir / "overtime_crosstab.csv")

    # Correlation heatmap (numeric only)
    df_corr = df.copy()
    if df_corr["Attrition"].dtype == object:
        df_corr["Attrition"] = df_corr["Attrition"].map({"Yes": 1, "No": 0})

    corr = df_corr.corr(numeric_only=True)
    if "Attrition" in corr.columns:
        target_corr = corr[["Attrition"]].sort_values(by="Attrition", ascending=False)
        plt.figure(figsize=(6, 10))
        sns.heatmap(target_corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation with Attrition (numeric)")
        plt.tight_layout()
        plt.savefig(out_dir / "corr_with_attrition.png", dpi=200)
        plt.close()

        target_corr.to_csv(out_dir / "corr_with_attrition.csv")

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()

