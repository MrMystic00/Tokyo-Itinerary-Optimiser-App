import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum
import random
from contraints import apply_constraints


def build_one_hot_matrix(attractions: pd.DataFrame) -> pd.DataFrame:
    """
    Build a one-hot encoded category matrix from attractions.

    Parameters
    ----------
    attractions : pd.DataFrame
        DataFrame containing category columns with the word 'categories'.

    Returns
    -------
    pd.DataFrame
        One-hot encoded binary matrix of categories.
    """
    df = attractions.copy().reset_index(drop=True)
    category_cols = [col for col in df.columns if "categories" in col.lower()]

    if len(category_cols) == 0:
        raise ValueError("No category columns found. Please ensure category columns contain the word 'categories'.")

    df["all_categories"] = df[category_cols].values.tolist()
    df["all_categories"] = df["all_categories"].apply(
        lambda x: [str(i).strip() for i in x if pd.notna(i) and str(i).strip() != ""]
    )

    categories_exploded = df[["all_categories"]].explode("all_categories")
    one_hot = pd.get_dummies(categories_exploded["all_categories"])
    one_hot_encoded = one_hot.groupby(categories_exploded.index).sum()
    one_hot_encoded = one_hot_encoded.reindex(df.index, fill_value=0)

    return one_hot_encoded


def one_hot_column_sums(attractions: pd.DataFrame) -> pd.Series:
    """
    Return the sum of each one-hot category column.

    Parameters
    ----------
    attractions : pd.DataFrame
        DataFrame containing category columns with the word 'categories'.

    Returns
    -------
    pd.Series
        Column sums for each one-hot encoded category.
    """
    one_hot_encoded = build_one_hot_matrix(attractions)
    return one_hot_encoded.sum(axis=0).astype(int).sort_values(ascending=False)


def print_one_hot_column_sums(attractions: pd.DataFrame) -> None:
    """
    Print the one-hot category sums to stdout.

    Parameters
    ----------
    attractions : pd.DataFrame
        DataFrame containing category columns with the word 'categories'.
    """
    sums = one_hot_column_sums(attractions)
    print("One-hot category counts:")
    for category, count in sums.items():
        print(f"{category}: {count}")


def print_indicator_sums(attractions: pd.DataFrame) -> None:
    """
    Print the sums of each binary indicator used for constraints.

    Parameters
    ----------
    attractions : pd.DataFrame
        DataFrame containing category columns with the word 'categories'.
    """
    df = attractions.copy().reset_index(drop=True)
    one_hot_encoded = build_one_hot_matrix(df)

    def cols_containing(keyword):
        keyword = keyword.lower()
        return [col for col in one_hot_encoded.columns if keyword in str(col).lower()]

    def row_match_indicator(columns):
        if not columns:
            return pd.Series(0, index=df.index)
        return (one_hot_encoded[columns].sum(axis=1) > 0).astype(int)

    shopping_mall_ind = row_match_indicator(cols_containing("shopping mall"))
    temple_ind = row_match_indicator(cols_containing("temple"))
    garden_ind = row_match_indicator(cols_containing("garden"))
    info_ind = row_match_indicator(
        cols_containing("information centre") + cols_containing("information center")
    )
    amusement_ind = row_match_indicator(
        cols_containing("amusement park")
        + cols_containing("amusement center")
        + cols_containing("amusement centre")
        + cols_containing("theme park")
    )
    bar_ind = row_match_indicator(cols_containing("bar"))
    shinjuku_ind = df["city"].astype(str).str.lower().str.contains("shinjuku").astype(int)

    print("\nIndicator sums:")
    print(f"shopping_mall_ind: {int(shopping_mall_ind.sum())}")
    print(f"temple_ind: {int(temple_ind.sum())}")
    print(f"garden_ind: {int(garden_ind.sum())}")
    print(f"info_ind: {int(info_ind.sum())}")
    print(f"amusement_ind: {int(amusement_ind.sum())}")
    print(f"bar_ind: {int(bar_ind.sum())}")
    print(f"shinjuku_ind: {int(shinjuku_ind.sum())}")



def solve_attraction_finding_problem(attractions: pd.DataFrame, budget=None, n_select=9, constraints=None):
    """
    Solve the attraction selection problem using Gurobi.

    Parameters
    ----------
    attractions : pd.DataFrame
        Expected columns:
        - city
        - prices
        - one review-score column such as 'review_score' or 'totalScore'
        - category columns containing the word 'categories'
    budget : float or None
        Total budget E. If None, uses average price * n_select.
    n_select : int
        Number of attractions to select. Default = 9.
    constraints : list[dict] | dict | None
        Constraint payload from frontend/backend API.

    Returns
    -------
    selected_df : pd.DataFrame or None
    model : gurobipy.Model
    one_hot_encoded : pd.DataFrame
    """

    df = attractions.copy().reset_index(drop=True)

    # -------------------------------------------------
    # Basic column handling
    # -------------------------------------------------
    if "title" not in df.columns:
        df["title"] = [f"Attraction_{i+1}" for i in range(len(df))]

    if "city" not in df.columns:
        df["city"] = ""

    # Review score column
    if "review_score" in df.columns:
        score_col = "review_score"
    elif "totalScore" in df.columns:
        score_col = "totalScore"
    elif "rating" in df.columns:
        score_col = "rating"
    else:
        random.seed(42)
        df["review_score"] = [round(random.uniform(3.5, 5.0), 2) for _ in range(len(df))]
        score_col = "review_score"

    # Price column
    if "prices" not in df.columns:
        random.seed(42)
        df["prices"] = [random.randint(0, 4000) for _ in range(len(df))]

    df["prices"] = pd.to_numeric(df["prices"], errors="coerce").fillna(0)

    # -------------------------------------------------
    # Build one-hot encoded category matrix
    # -------------------------------------------------
    one_hot_encoded = build_one_hot_matrix(df)

    category_cols = [col for col in df.columns if "categories" in col.lower()]
    df["all_categories"] = df[category_cols].values.tolist()
    df["all_categories"] = df["all_categories"].apply(
        lambda x: [str(i).strip() for i in x if pd.notna(i) and str(i).strip() != ""]
    )

    # -------------------------------------------------
    # Sets and parameters
    # -------------------------------------------------
    I = range(len(df))
    r = df[score_col].to_dict()

    # Default budget = average price * number of attractions
    if budget is None:
        budget = df["prices"].mean() * n_select

    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    model = gp.Model("attraction_finding")
    model.setParam("OutputFlag", 0)

    # y[i] = 1 if attraction i is selected
    y = model.addVars(I, vtype=GRB.BINARY, name="y")

    # Objective: maximize total review score
    model.setObjective(quicksum(r[i] * y[i] for i in I), GRB.MAXIMIZE)

    # -------------------------------------------------
    # Constraints
    # -------------------------------------------------
    apply_constraints(
        model=model,
        y=y,
        df=df,
        one_hot_encoded=one_hot_encoded,
        constraints=constraints,
        budget=budget,
        n_select=n_select,
    )

    # -------------------------------------------------
    # Solve
    # -------------------------------------------------
    model.optimize()

    # -------------------------------------------------
    # Results
    # -------------------------------------------------
    if model.status == GRB.OPTIMAL:
        selected_idx = [i for i in I if y[i].X > 0.5]

        selected_df = df.loc[selected_idx, ["title", score_col, "prices", "city", "all_categories"]].copy()
        selected_df = selected_df.rename(columns={score_col: "review_score"})
        selected_df = selected_df.sort_values("review_score", ascending=False).reset_index(drop=True)

        total_score = selected_df["review_score"].sum()
        total_price = selected_df["prices"].sum()

        print(f"Objective value (total review score): {model.objVal:.4f}")
        print(f"Total entrance fee: {total_price:.2f}")
        print(f"Budget limit: {budget:.2f}")
        print("\nSelected attractions:")
        print(selected_df.to_string(index=False))

        return selected_df, model, one_hot_encoded

    else:
        print(f"Optimization ended with status {model.status}")
        return None, model, one_hot_encoded


def _prepare_attraction_dataframe(attractions: pd.DataFrame, weighted=False):
    df = attractions.copy().reset_index(drop=True)

    if "title" not in df.columns:
        df["title"] = [f"Attraction_{i+1}" for i in range(len(df))]

    if "city" not in df.columns:
        df["city"] = ""

    if "review_score" in df.columns:
        score_col = "review_score"
    elif "totalScore" in df.columns:
        score_col = "totalScore"
    elif "rating" in df.columns:
        score_col = "rating"
    else:
        random.seed(42)
        df["review_score"] = [round(random.uniform(3.5, 5.0), 2) for _ in range(len(df))]
        score_col = "review_score"

    if "prices" not in df.columns:
        random.seed(42)
        df["prices"] = [random.randint(0, 4000) for _ in range(len(df))]

    df["prices"] = pd.to_numeric(df["prices"], errors="coerce").fillna(0)

    one_hot_encoded = build_one_hot_matrix(df)
    category_cols = [col for col in df.columns if "categories" in col.lower()]
    df["all_categories"] = df[category_cols].values.tolist()
    df["all_categories"] = df["all_categories"].apply(
        lambda x: [str(i).strip() for i in x if pd.notna(i) and str(i).strip() != ""]
    )

    if weighted:
        r = pd.to_numeric(df[score_col], errors="coerce").fillna(0)
        e = pd.to_numeric(df["prices"], errors="coerce").fillna(0)

        r_min, r_max = r.min(), r.max()
        e_min, e_max = e.min(), e.max()

        df["r_i_norm"] = 1 if r_max == r_min else (r - r_min) / (r_max - r_min)
        df["e_i_norm"] = 0 if e_max == e_min else (e - e_min) / (e_max - e_min)

    return df, score_col, one_hot_encoded


def solve_attraction_finding_weighted(
    attractions,
    n_select=9,
    constraints=None,
    budget=None,
    preference="Premium Experience",
):
    df, score_col, one_hot_encoded = _prepare_attraction_dataframe(attractions, weighted=True)

    weights = {
        "Premium Experience": (0.8, 0.2),
        "Balanced Experience": (0.5, 0.5),
        "Budget-friendly": (0.2, 0.8),
    }
    if preference not in weights:
        raise ValueError("Invalid preference")
    w_r, w_e = weights[preference]

    model = gp.Model("attraction_finding_weighted")
    model.setParam("OutputFlag", 0)

    I = range(len(df))
    y = model.addVars(I, vtype=GRB.BINARY, name="y")

    r_norm = df["r_i_norm"].to_dict()
    e_norm = df["e_i_norm"].to_dict()

    model.setObjective(
        quicksum((w_r * r_norm[i] - w_e * e_norm[i]) * y[i] for i in I),
        GRB.MAXIMIZE,
    )

    if budget is None:
        budget = df["prices"].mean() * n_select

    apply_constraints(
        model=model,
        y=y,
        df=df,
        one_hot_encoded=one_hot_encoded,
        constraints=constraints,
        budget=budget,
        n_select=n_select,
    )

    model.optimize()

    if model.status == GRB.OPTIMAL:
        selected_idx = [i for i in I if y[i].X > 0.5]
        selected_df = df.loc[selected_idx, ["title", score_col, "prices", "city", "all_categories", "r_i_norm", "e_i_norm"]].copy()
        selected_df = selected_df.rename(columns={score_col: "review_score"})
        selected_df["weighted_score"] = (w_r * selected_df["r_i_norm"] - w_e * selected_df["e_i_norm"])
        selected_df = selected_df.sort_values("weighted_score", ascending=False).reset_index(drop=True)
        return selected_df, model, one_hot_encoded

    return None, model, one_hot_encoded
    
def load_attractions_data(filepath: str) -> pd.DataFrame:
    """
    Load attractions data from CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file
        
    Returns
    -------
    pd.DataFrame
        Loaded attractions data
    """
    df = pd.read_csv(filepath)
    return df


if __name__ == "__main__":
    # Load data
    csv_path = "../data/attractions_with_coordinates.csv"
    attractions = load_attractions_data(csv_path)
    
    print(f"Loaded {len(attractions)} attractions")
    print(f"Columns: {list(attractions.columns)}\n")

    print_one_hot_column_sums(attractions)
    print_indicator_sums(attractions)
    
    # Run optimization
    selected, model, encoded = solve_attraction_finding_problem(
        attractions, 
        budget=10000,
        n_select=9
    )
