import pandas as pd
from gurobipy import GRB, quicksum


CONSTRAINT_TYPES = [
    "exact_n",
    "budget",
    "attraction_required",
    "attraction_forbidden",
    "if_attraction_then_attraction",
    "if_all_attractions_then_all_attractions",
    "if_any_attraction_then_any_attraction",
    "not_both_attractions",
    "at_most_one_total_from_two_attraction_groups",
    "not_both_attraction_groups",
    "attraction_set_min",
    "attraction_set_max",
    "attraction_set_exact",
    "category_min",
    "category_max",
    "category_exact",
    "category_forbidden",
    "if_category_then_category",
    "not_both_categories",
    "if_category_count_at_least_p_then_category_count_at_least_q",
    "category_group_min",
    "category_group_max",
    "category_group_bounds",
    "if_all_categories_then_all_categories",
    "not_both_category_groups",
    "city_min",
    "city_max",
    "city_exact",
]


def _norm(text):
    return str(text or "").strip().lower()


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_constraints(constraints):
    if constraints is None:
        return []

    if isinstance(constraints, list):
        return constraints

    raise ValueError("constraints must be a list")


def validate_constraints(constraints):
    if not isinstance(constraints, list):
        raise ValueError("constraints must be a list")

    for idx, c in enumerate(constraints):
        if not isinstance(c, dict):
            raise ValueError(f"constraint[{idx}] must be an object")
        ctype = c.get("type")
        if ctype not in CONSTRAINT_TYPES:
            raise ValueError(f"Unsupported constraint type at index {idx}: {ctype}")


def _build_title_index(df):
    title_map = {}
    for i, title in enumerate(df["title"].astype(str).tolist()):
        title_map.setdefault(_norm(title), []).append(i)
    return title_map


def _title_indices(names, title_map):
    idxs = set()
    for name in _as_list(names):
        key = _norm(name)
        if key in title_map:
            for i in title_map[key]:
                idxs.add(i)
    return sorted(idxs)


def _category_indicator(one_hot_encoded, category):
    cat = _norm(category)
    cols = [c for c in one_hot_encoded.columns if _norm(c) == cat]
    if not cols:
        cols = [c for c in one_hot_encoded.columns if cat in _norm(c)]
    if not cols:
        return pd.Series(0, index=one_hot_encoded.index, dtype=int)
    return (one_hot_encoded[cols].sum(axis=1) > 0).astype(int)


def _category_group_indicator(one_hot_encoded, categories):
    group = pd.Series(0, index=one_hot_encoded.index, dtype=int)
    for cat in _as_list(categories):
        group = ((group > 0) | (_category_indicator(one_hot_encoded, cat) > 0)).astype(int)
    return group


def _city_indicator(df, city):
    city_norm = _norm(city)
    if not city_norm:
        return pd.Series(0, index=df.index, dtype=int)
    return df["city"].astype(str).str.lower().str.contains(city_norm, na=False).astype(int)


def apply_constraints(model, y, df, one_hot_encoded, constraints, budget=None, n_select=9):
    constraints = normalize_constraints(constraints)
    validate_constraints(constraints)

    I = range(len(df))
    prices = pd.to_numeric(df["prices"], errors="coerce").fillna(0).to_dict()
    title_map = _build_title_index(df)

    category_indicator_cache = {}
    category_presence_var_cache = {}

    def cat_indicator(cat):
        key = _norm(cat)
        if key not in category_indicator_cache:
            category_indicator_cache[key] = _category_indicator(one_hot_encoded, cat)
        return category_indicator_cache[key]

    def cat_expr(cat):
        ind = cat_indicator(cat)
        return quicksum(float(ind[i]) * y[i] for i in I)

    def city_expr(city):
        ind = _city_indicator(df, city)
        return quicksum(float(ind[i]) * y[i] for i in I)

    def title_expr(names):
        idxs = _title_indices(names, title_map)
        return quicksum(y[i] for i in idxs), idxs

    def category_presence_var(cat, suffix):
        key = _norm(cat)
        if key in category_presence_var_cache:
            return category_presence_var_cache[key]

        z = model.addVar(vtype=GRB.BINARY, name=f"z_cat_{suffix}_{len(category_presence_var_cache)}")
        ind = cat_indicator(cat)
        expr = quicksum(float(ind[i]) * y[i] for i in I)
        m = max(1, int(ind.sum()))

        model.addConstr(expr >= z, name=f"cat_presence_lb_{suffix}_{len(category_presence_var_cache)}")
        model.addConstr(expr <= m * z, name=f"cat_presence_ub_{suffix}_{len(category_presence_var_cache)}")

        category_presence_var_cache[key] = z
        return z

    has_exact_n = False
    has_budget = False

    for k, c in enumerate(constraints):
        ctype = c["type"]
        cname = f"c_{k}_{ctype}"

        if ctype == "exact_n":
            target = int(c.get("value", c.get("n", n_select)))
            model.addConstr(quicksum(y[i] for i in I) == target, name=cname)
            has_exact_n = True

        elif ctype == "budget":
            b = float(c.get("value", c.get("budget", budget if budget is not None else 0)))
            model.addConstr(quicksum(prices[i] * y[i] for i in I) <= b, name=cname)
            has_budget = True

        elif ctype == "attraction_required":
            expr, idxs = title_expr(c.get("attractions", []))
            if not idxs:
                raise ValueError(f"{cname}: no matching attractions")
            model.addConstr(expr == len(idxs), name=cname)

        elif ctype == "attraction_forbidden":
            expr, idxs = title_expr(c.get("attractions", []))
            if not idxs:
                continue
            model.addConstr(expr == 0, name=cname)

        elif ctype == "if_attraction_then_attraction":
            expr_a, idxs_a = title_expr([c.get("a")])
            expr_b, idxs_b = title_expr([c.get("b")])
            if not idxs_a or not idxs_b:
                raise ValueError(f"{cname}: a or b not found")
            model.addConstr(expr_a <= expr_b, name=cname)

        elif ctype == "if_all_attractions_then_all_attractions":
            P = c.get("P") or _as_list(c.get("a"))
            Q = c.get("Q") or _as_list(c.get("b"))
            expr_p, idxs_p = title_expr(P)
            expr_q, idxs_q = title_expr(Q)
            if not idxs_p or not idxs_q:
                raise ValueError(f"{cname}: P or Q has no matching attractions")
            model.addConstr(expr_p - len(idxs_p) + 1 <= expr_q, name=cname)

        elif ctype == "if_any_attraction_then_any_attraction":
            expr_p, idxs_p = title_expr(c.get("P", []))
            expr_q, idxs_q = title_expr(c.get("Q", []))
            if not idxs_p or not idxs_q:
                raise ValueError(f"{cname}: P or Q has no matching attractions")
            model.addConstr(expr_p / len(idxs_p) <= expr_q, name=cname)

        elif ctype == "not_both_attractions":
            expr_a, idxs_a = title_expr([c.get("a")])
            expr_b, idxs_b = title_expr([c.get("b")])
            if not idxs_a or not idxs_b:
                raise ValueError(f"{cname}: a or b not found")
            model.addConstr(expr_a + expr_b <= 1, name=cname)

        elif ctype == "at_most_one_total_from_two_attraction_groups":
            expr_g1, _ = title_expr(c.get("G1", []))
            expr_g2, _ = title_expr(c.get("G2", []))
            model.addConstr(expr_g1 + expr_g2 <= 1, name=cname)

        elif ctype == "not_both_attraction_groups":
            expr_g1, idxs_g1 = title_expr(c.get("G1", []))
            expr_g2, idxs_g2 = title_expr(c.get("G2", []))
            u = model.addVar(vtype=GRB.BINARY, name=f"u_attr_group_{k}")
            model.addConstr(expr_g1 <= len(idxs_g1) * u, name=f"{cname}_g1")
            model.addConstr(expr_g2 <= len(idxs_g2) * (1 - u), name=f"{cname}_g2")

        elif ctype == "attraction_set_min":
            expr, _ = title_expr(c.get("attractions", []))
            model.addConstr(expr >= int(c.get("value", 0)), name=cname)

        elif ctype == "attraction_set_max":
            expr, _ = title_expr(c.get("attractions", []))
            model.addConstr(expr <= int(c.get("value", 0)), name=cname)

        elif ctype == "attraction_set_exact":
            expr, _ = title_expr(c.get("attractions", []))
            model.addConstr(expr == int(c.get("value", 0)), name=cname)

        elif ctype == "category_min":
            model.addConstr(cat_expr(c.get("category")) >= int(c.get("value", 0)), name=cname)

        elif ctype == "category_max":
            model.addConstr(cat_expr(c.get("category")) <= int(c.get("value", 0)), name=cname)

        elif ctype == "category_exact":
            model.addConstr(cat_expr(c.get("category")) == int(c.get("value", 0)), name=cname)

        elif ctype == "category_forbidden":
            model.addConstr(cat_expr(c.get("category")) == 0, name=cname)

        elif ctype == "if_category_then_category":
            za = category_presence_var(c.get("a"), f"{k}_a")
            zb = category_presence_var(c.get("b"), f"{k}_b")
            model.addConstr(za <= zb, name=cname)

        elif ctype == "not_both_categories":
            za = category_presence_var(c.get("a"), f"{k}_a")
            zb = category_presence_var(c.get("b"), f"{k}_b")
            model.addConstr(za + zb <= 1, name=cname)

        elif ctype == "if_category_count_at_least_p_then_category_count_at_least_q":
            p = int(c.get("p", 0))
            q = int(c.get("q", 0))
            expr_a = cat_expr(c.get("a"))
            expr_b = cat_expr(c.get("b"))
            M = len(df)
            t = model.addVar(vtype=GRB.BINARY, name=f"t_cat_count_{k}")
            model.addConstr(expr_a >= p * t, name=f"{cname}_a_lb")
            model.addConstr(expr_a <= (p - 1) + M * t, name=f"{cname}_a_ub")
            model.addConstr(expr_b >= q * t, name=f"{cname}_b_lb")

        elif ctype == "category_group_min":
            ind = _category_group_indicator(one_hot_encoded, c.get("categories", []))
            expr = quicksum(float(ind[i]) * y[i] for i in I)
            model.addConstr(expr >= int(c.get("value", 0)), name=cname)

        elif ctype == "category_group_max":
            ind = _category_group_indicator(one_hot_encoded, c.get("categories", []))
            expr = quicksum(float(ind[i]) * y[i] for i in I)
            model.addConstr(expr <= int(c.get("value", 0)), name=cname)

        elif ctype == "category_group_bounds":
            ind = _category_group_indicator(one_hot_encoded, c.get("categories", []))
            expr = quicksum(float(ind[i]) * y[i] for i in I)
            model.addConstr(expr >= int(c.get("L", 0)), name=f"{cname}_lb")
            model.addConstr(expr <= int(c.get("U", len(df))), name=f"{cname}_ub")

        elif ctype == "if_all_categories_then_all_categories":
            C1 = _as_list(c.get("C1", []))
            C2 = _as_list(c.get("C2", []))
            z1 = [category_presence_var(cat, f"{k}_c1_{j}") for j, cat in enumerate(C1)]
            z2 = [category_presence_var(cat, f"{k}_c2_{j}") for j, cat in enumerate(C2)]
            if not z1 or not z2:
                continue
            for j, z in enumerate(z2):
                model.addConstr(quicksum(z1) - len(z1) + 1 <= z, name=f"{cname}_{j}")

        elif ctype == "not_both_category_groups":
            C1 = _as_list(c.get("C1", []))
            C2 = _as_list(c.get("C2", []))
            z1 = [category_presence_var(cat, f"{k}_c1_{j}") for j, cat in enumerate(C1)]
            z2 = [category_presence_var(cat, f"{k}_c2_{j}") for j, cat in enumerate(C2)]
            u = model.addVar(vtype=GRB.BINARY, name=f"u_cat_group_{k}")
            model.addConstr(quicksum(z1) <= max(1, len(z1)) * u, name=f"{cname}_c1")
            model.addConstr(quicksum(z2) <= max(1, len(z2)) * (1 - u), name=f"{cname}_c2")

        elif ctype == "city_min":
            model.addConstr(city_expr(c.get("city")) >= int(c.get("value", 0)), name=cname)

        elif ctype == "city_max":
            model.addConstr(city_expr(c.get("city")) <= int(c.get("value", 0)), name=cname)

        elif ctype == "city_exact":
            model.addConstr(city_expr(c.get("city")) == int(c.get("value", 0)), name=cname)

    if not has_exact_n:
        model.addConstr(quicksum(y[i] for i in I) == int(n_select), name="default_exact_n")

    if not has_budget and budget is not None:
        model.addConstr(quicksum(prices[i] * y[i] for i in I) <= float(budget), name="default_budget")
