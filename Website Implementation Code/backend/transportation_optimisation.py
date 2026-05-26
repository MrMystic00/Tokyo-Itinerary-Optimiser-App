import collections
from pathlib import Path

import numpy as np
import pandas as pd
from gurobipy import GRB, Model, quicksum

VERBOSE = False


def _log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def _load_df(value, label):
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, (str, Path)):
        return pd.read_csv(value)
    raise TypeError(f"{label} must be a pandas DataFrame or CSV path")


def preprocess_data(df_lines, df_stations, df_joins, df_companies, df_locations):

    _log("🔧 Preprocessing data...")

    df_lines = _load_df(df_lines, "df_lines")
    df_stations = _load_df(df_stations, "df_stations")
    df_joins = _load_df(df_joins, "df_joins")
    df_companies = _load_df(df_companies, "df_companies")
    df_locations = _load_df(df_locations, "df_locations")

    df_lines["line_cd"] = df_lines["line_cd"].astype(str)
    df_lines["company_cd"] = df_lines["company_cd"].astype(str)

    df_stations["station_cd"] = df_stations["station_cd"].astype(str)
    df_stations["station_g_cd"] = df_stations["station_g_cd"].astype(str)
    df_stations["line_cd"] = df_stations["line_cd"].astype(str)
    df_stations["lat"] = df_stations["lat"].astype(float)
    df_stations["lon"] = df_stations["lon"].astype(float)

    df_joins["line_cd"] = df_joins["line_cd"].astype(str)
    df_joins["station_cd1"] = df_joins["station_cd1"].astype(str)
    df_joins["station_cd2"] = df_joins["station_cd2"].astype(str)

    df_companies["company_cd"] = df_companies["company_cd"].astype(str)

    df_locations["latitude"] = df_locations["latitude"].astype(float)
    df_locations["longitude"] = df_locations["longitude"].astype(float)

    df_locations = df_locations.reset_index(drop=True)
    df_locations["node_id"] = df_locations.index

    _log("✅ Data cleaned")

    return df_lines, df_stations, df_joins, df_companies, df_locations

def actual_distance(coord1, coord2):
    R = 6371

    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return R * c


def _haversine_vectorized(lat1, lon1, lat2_arr, lon2_arr):
    """
    Vectorized great-circle distance (km) from one point to arrays of points.
    """
    R = 6371.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2_arr)
    lon2_rad = np.radians(lon2_arr)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def balanced_geo_cluster(df_locations, K=3):
    """
    Cluster attractions into groups of size K using angular grouping
    around the global centroid.
    Returns list of DataFrames with columns: id, name, lat, lon
    """

    df = df_locations.copy().reset_index(drop=True)

    # standardise column names if needed
    if "node_id" in df.columns and "id" not in df.columns:
        df = df.rename(columns={"node_id": "id"})
    if "title" in df.columns and "name" not in df.columns:
        df = df.rename(columns={"title": "name"})
    if "latitude" in df.columns and "lat" not in df.columns:
        df = df.rename(columns={"latitude": "lat"})
    if "longitude" in df.columns and "lon" not in df.columns:
        df = df.rename(columns={"longitude": "lon"})

    # global centroid
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()

    # angle from centroid
    df["angle"] = df.apply(
        lambda r: np.arctan2(r["lat"] - center_lat, r["lon"] - center_lon),
        axis=1
    )

    # sort around the center
    df = df.sort_values("angle").reset_index(drop=True)

    clusters = []
    for i in range(0, len(df), K):
        cluster_df = df.iloc[i:i+K][["id", "name", "lat", "lon"]].copy()
        clusters.append(cluster_df)

    return clusters

def cluster_cost(cluster_df):
    coords = cluster_df[["lat", "lon"]].values.tolist()
    total = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            total += actual_distance(coords[i], coords[j])
    return total

def total_cluster_cost(clusters):
    return sum(cluster_cost(c) for c in clusters)

def refine_clusters_by_swapping(clusters, max_iter=20):
    """
    Improve clusters by swapping points between clusters
    if total within-cluster distance decreases.
    """

    clusters = [c.reset_index(drop=True).copy() for c in clusters]

    improved = True
    iteration = 0

    while improved and iteration < max_iter:
        improved = False
        iteration += 1

        current_best = total_cluster_cost(clusters)

        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                ca = clusters[a]
                cb = clusters[b]

                for i in range(len(ca)):
                    for j in range(len(cb)):
                        new_ca = ca.copy()
                        new_cb = cb.copy()

                        row_a = ca.iloc[i].copy()
                        row_b = cb.iloc[j].copy()

                        new_ca.iloc[i] = row_b
                        new_cb.iloc[j] = row_a

                        new_clusters = clusters.copy()
                        new_clusters[a] = new_ca
                        new_clusters[b] = new_cb

                        new_cost = total_cluster_cost(new_clusters)

                        if new_cost < current_best:
                            clusters = [c.copy() for c in new_clusters]
                            current_best = new_cost
                            improved = True

        # continue until no improving swap exists

    return clusters

def build_station_graph(df_stations, df_joins):

    _log("\n🔹 Building LINE-AWARE station graph...")

    graph = collections.defaultdict(list)

    # -------------------------
    # Create node: (station_g_cd, line_cd)
    # -------------------------
    station_nodes = df_stations[["station_g_cd", "line_cd"]].drop_duplicates()

    # -------------------------
    # 1) SAME LINE MOVEMENT
    # -------------------------
    df_sorted = df_stations.sort_values(["line_cd", "station_cd"])

    for line, group in df_sorted.groupby("line_cd"):

        stations = group["station_g_cd"].tolist()

        for i in range(len(stations) - 1):
            a = (stations[i], line)
            b = (stations[i+1], line)

            graph[a].append(b)
            graph[b].append(a)

    # -------------------------
    # 2) TRANSFERS (same station, different lines)
    # -------------------------
    for station, group in df_stations.groupby("station_g_cd"):

        lines = group["line_cd"].unique()

        for i in range(len(lines)):
            for j in range(i+1, len(lines)):

                a = (station, lines[i])
                b = (station, lines[j])

                graph[a].append(b)
                graph[b].append(a)

    return graph

def nearest_station(coord, df_stations):
    lat, lon = coord

    # Build/cache numeric coordinate arrays once per station DataFrame instance.
    cache_key = "_nearest_station_cache"
    cache = df_stations.attrs.get(cache_key)
    if cache is None:
        cache = {
            "lat": pd.to_numeric(df_stations["lat"], errors="coerce").to_numpy(),
            "lon": pd.to_numeric(df_stations["lon"], errors="coerce").to_numpy(),
        }
        df_stations.attrs[cache_key] = cache

    distances = _haversine_vectorized(lat, lon, cache["lat"], cache["lon"])
    best_idx = int(np.nanargmin(distances))
    best_station = df_stations.iloc[best_idx]
    best_dist = float(distances[best_idx])
    return best_station, best_dist

def train_route(src_name, dst_name, df_stations, graph, df_lines):

    _log(f"\n🚆 Finding route: {src_name} → {dst_name}")

    line_map = df_lines.set_index("line_cd")["line_name"].to_dict()
    
    src_rows = df_stations[df_stations["station_name"] == src_name]
    dst_rows = df_stations[df_stations["station_name"] == dst_name]

    if src_rows.empty or dst_rows.empty:
        return None

    # 🔥 MULTI-SOURCE BFS (all lines at source station)
    start_nodes = [(r["station_g_cd"], r["line_cd"]) for _, r in src_rows.iterrows()]
    end_stations = set(dst_rows["station_g_cd"])

    queue = collections.deque()
    visited = set()

    for node in start_nodes:
        queue.append((node, [node]))
        visited.add(node)

    while queue:

        (curr_station, curr_line), path = queue.popleft()

        # ✅ stop when we reach destination station (any line)
        if curr_station in end_stations:
            final_path = [s for (s, l) in path]
            _log("✅ Route found:", final_path)
            return final_path

        for nei_station, nei_line in graph[(curr_station, curr_line)]:
        
            curr_line_name = line_map.get(curr_line)
            nei_line_name = line_map.get(nei_line)
        
            # 🔥 KEY RULE
            # stay on same line_name OR allow transfer at same station
            if not (
                nei_line_name == curr_line_name
                or nei_station == curr_station
            ):
                continue
        
            if (nei_station, nei_line) not in visited:
                visited.add((nei_station, nei_line))
                queue.append(
                    ((nei_station, nei_line), path + [(nei_station, nei_line)])
                )

    _log("❌ No route found")
    return None

def compress_route(df_stations, station_sequence, df_lines):

    line_cd_to_name = df_lines.set_index("line_cd")["line_name"].to_dict()

    segments = []

    current_line = None
    current_segment = []

    for station_cd in station_sequence:

        matches = df_stations[df_stations["station_g_cd"] == str(station_cd)]

        if matches.empty:
            continue

        row = matches.iloc[0]
        station = row["station_name"]
        line_name = line_cd_to_name.get(row["line_cd"], "Unknown")

        if current_line is None:
            current_line = line_name
            current_segment = [station]
            continue

        if line_name == current_line:
            if station not in current_segment:
                current_segment.append(station)
        else:
            segments.append((current_line, current_segment))
            current_line = line_name
            current_segment = [station]

    if current_segment:
        segments.append((current_line, current_segment))

    return segments

def solve_tsp(cluster_df, df_stations, start_lat, start_lon):

    _log("\n🔹 Solving TSP...")

    # -------------------------
    # Nodes (DEPOT = START)
    # -------------------------
    nodes = ["START"] + [str(i) for i in cluster_df["id"]]
    
    coords = {"START": (start_lat, start_lon)}
    
    for _, row in cluster_df.iterrows():
        coords[str(row["id"])] = (row["lat"], row["lon"])

    # -------------------------
    # MODEL
    # -------------------------
    m = Model()
    x = m.addVars(nodes, nodes, vtype=GRB.BINARY)
    u = m.addVars(nodes, vtype=GRB.CONTINUOUS)

    # -------------------------
    # COST MATRIX
    # -------------------------
    c = {}
    for i in nodes:
        for j in nodes:
            if i != j:
                c[i, j] = actual_distance(coords[i], coords[j])

    # -------------------------
    # OBJECTIVE
    # -------------------------
    m.setObjective(
        quicksum(c[i, j] * x[i, j] for i in nodes for j in nodes if i != j),
        GRB.MINIMIZE
    )

    # -------------------------
    # CONSTRAINTS
    # -------------------------
    for i in nodes:
        m.addConstr(quicksum(x[i, j] for j in nodes if j != i) == 1)
        m.addConstr(quicksum(x[j, i] for j in nodes if j != i) == 1)

    start = "START"

    for i in nodes:
        for j in nodes:
            if i != j and i != start and j != start:
                m.addConstr(
                    u[i] - u[j] + len(nodes) * x[i, j] <= len(nodes) - 1
                )

    # -------------------------
    # SOLVE
    # -------------------------
    m.optimize()

    if m.status != GRB.OPTIMAL:
        return None

    # -------------------------
    # EXTRACT EDGES
    # -------------------------
    edges = [(i, j) for i in nodes for j in nodes if i != j and x[i, j].x > 0.5]

    # -------------------------
    # REBUILD ORDERED ROUTE
    # -------------------------
    route = ["START"]
    current = "START"

    visited = set(["START"])

    while True:
        next_nodes = [j for (i, j) in edges if i == current]
        if not next_nodes:
            break

        nxt = next_nodes[0]

        if nxt == "START":
            route.append("START")
            break

        route.append(nxt)
        visited.add(nxt)
        current = nxt

    return route
def pre_itinerary(coord1, coord2, name1, name2, df_stations, graph, df_lines):

    _log("\n🧭 Building pre-itinerary...")

    itinerary_rows = []

    # -------------------------
    # 1) START LOCATION
    # -------------------------
    itinerary_rows.append({
        "type": "location",
        "name": name1,
        "lat": coord1[0],
        "lon": coord1[1]
    })

    # -------------------------
    # 2) NEAREST STATIONS
    # -------------------------
    st1, _ = nearest_station(coord1, df_stations)
    st2, _ = nearest_station(coord2, df_stations)

    _log(f"🚆 Routing: {st1['station_name']} → {st2['station_name']}")

    # add start station
    itinerary_rows.append({
        "type": "station",
        "name": st1["station_name"],
        "station_cd": st1["station_cd"],
        "station_g_cd": st1["station_g_cd"],
        "line_cd": st1["line_cd"],
        "lat": st1["lat"],
        "lon": st1["lon"]
    })

    # -------------------------
    # 3) TRAIN ROUTE (FULL PATH)
    # -------------------------
    route_path = train_route(
        st1["station_name"],
        st2["station_name"],
        df_stations,
        graph,
        df_lines
    )

    if route_path is None:
        route_path = []

    # -------------------------
    # 4) EXPAND ROUTE (FIXED VERSION)
    # -------------------------
    if len(route_path) > 0:

        cleaned_rows = []

        for g in route_path:

            # get all rows for this station group
            candidates = df_stations[df_stations["station_g_cd"] == g]

            # 🔥 pick ONE station only (prevents duplicates)
            row = candidates.sort_values("station_cd").iloc[0]

            cleaned_rows.append(row)

        route_df = pd.DataFrame(cleaned_rows)

        # remove endpoints (already added separately)
        route_df = route_df[
            ~route_df["station_cd"].isin([
                st1["station_cd"],
                st2["station_cd"]
            ])
        ]

        # final dedupe safety
        route_df = route_df.drop_duplicates(subset="station_cd")

        # append intermediate stations
        for _, row in route_df.iterrows():
            itinerary_rows.append({
                "type": "station",
                "name": row["station_name"],
                "station_cd": row["station_cd"],
                "station_g_cd": row["station_g_cd"],
                "line_cd": row["line_cd"],
                "lat": row["lat"],
                "lon": row["lon"]
            })

    else:
        _log("⚠️ No train path found (fallback: direct)")

    # -------------------------
    # 5) END STATION
    # -------------------------
    itinerary_rows.append({
        "type": "station",
        "name": st2["station_name"],
        "station_cd": st2["station_cd"],
        "station_g_cd": st2["station_g_cd"],
        "line_cd": st2["line_cd"],
        "lat": st2["lat"],
        "lon": st2["lon"]
    })

    # -------------------------
    # 6) END LOCATION
    # -------------------------
    itinerary_rows.append({
        "type": "location",
        "name": name2,
        "lat": coord2[0],
        "lon": coord2[1]
    })

    # -------------------------
    # 7) FINAL CLEAN
    # -------------------------
    df_itinerary = pd.DataFrame(itinerary_rows)

    # remove consecutive duplicate stations
    df_itinerary = df_itinerary.loc[
        (df_itinerary["type"] != "station") |
        (df_itinerary["station_cd"] != df_itinerary["station_cd"].shift())
    ]

    _log("\n✅ Pre-itinerary built:")
    _log(df_itinerary)

    return df_itinerary


def _normalize_locations_df(df_locations):
    df = df_locations.copy().reset_index(drop=True)
    if "node_id" in df.columns and "id" not in df.columns:
        df = df.rename(columns={"node_id": "id"})
    if "title" in df.columns and "name" not in df.columns:
        df = df.rename(columns={"title": "name"})
    if "latitude" in df.columns and "lat" not in df.columns:
        df = df.rename(columns={"latitude": "lat"})
    if "longitude" in df.columns and "lon" not in df.columns:
        df = df.rename(columns={"longitude": "lon"})
    if "id" not in df.columns:
        df["id"] = range(len(df))
    return df


def _cluster_by_days(df_locations, max_attractions_per_day, refine=False, max_refine_iter=5):
    df = _normalize_locations_df(df_locations)
    n = len(df)
    if n == 0:
        return []
    k = max(1, min(int(max_attractions_per_day), n))
    clusters = balanced_geo_cluster(df, K=k)
    clusters = [c for c in clusters if not c.empty]
    if refine and len(clusters) > 1:
        clusters = refine_clusters_by_swapping(clusters, max_iter=max_refine_iter)
    return clusters


def build_multi_day_itinerary(
    df_locations,
    max_attractions_per_day,
    start_lat,
    start_lon,
    line_file="../data/line20240426.csv",
    station_file="../data/station20240426.csv",
    join_file="../data/join20240426.csv",
    company_file="../data/company20240328.csv",
    start_name="START",
    refine_clusters=False,
    max_refine_iter=5,
):
    """
    Build day-wise itinerary records (locations + train stations) from selected attractions.
    Days are derived automatically from max_attractions_per_day.
    Returns:
      {
        "Day 1": [{"seq":1,"type":"location|station","name":...,"lat":...,"lon":...}, ...],
        ...
      }
    """
    df_lines, df_stations, df_joins, df_companies, df_locations_clean = preprocess_data(
        line_file, station_file, join_file, company_file, df_locations
    )
    src_locations = _normalize_locations_df(df_locations)
    location_meta = {}
    for _, row in src_locations.iterrows():
        location_meta[str(row.get("name", ""))] = {
            "review_score": row.get("review_score"),
            "city": row.get("city"),
            "all_categories": row.get("all_categories"),
        }

    graph = build_station_graph(df_stations, df_joins)

    clusters = _cluster_by_days(
        df_locations_clean,
        max_attractions_per_day,
        refine=refine_clusters,
        max_refine_iter=max_refine_iter,
    )
    results = {}

    for d_idx, cluster_df in enumerate(clusters, start=1):
        day_key = f"Day {d_idx}"

        route = solve_tsp(cluster_df, df_stations, start_lat, start_lon)
        if not route:
            route = ["START"] + [str(i) for i in cluster_df["id"].tolist()] + ["START"]

        id_to_info = {}
        for _, row in cluster_df.iterrows():
            id_to_info[str(row["id"])] = {
                "name": row["name"],
                "coord": (float(row["lat"]), float(row["lon"])),
            }

        points = []
        for rid in route:
            if rid == "START":
                points.append((start_name, (float(start_lat), float(start_lon))))
            elif rid in id_to_info:
                points.append((id_to_info[rid]["name"], id_to_info[rid]["coord"]))

        if len(points) < 2:
            results[day_key] = []
            continue

        day_frames = []
        for i in range(len(points) - 1):
            name1, coord1 = points[i]
            name2, coord2 = points[i + 1]
            leg_df = pre_itinerary(coord1, coord2, name1, name2, df_stations, graph, df_lines)
            if i > 0 and not leg_df.empty:
                leg_df = leg_df.iloc[1:].reset_index(drop=True)
            day_frames.append(leg_df)

        if day_frames:
            day_df = pd.concat(day_frames, ignore_index=True)
        else:
            day_df = pd.DataFrame(columns=["type", "name", "lat", "lon"])

        day_df = day_df.loc[
            (day_df["type"] != day_df["type"].shift())
            | (day_df["name"] != day_df["name"].shift())
        ].reset_index(drop=True)

        records = []
        for seq, (_, r) in enumerate(day_df.iterrows(), start=1):
            name = str(r["name"])
            point_type = str(r["type"])
            rec = {
                "seq": seq,
                "type": point_type,
                "name": name,
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
            }

            if point_type == "location":
                meta = location_meta.get(name, {})
                rec["review_score"] = meta.get("review_score")
                rec["city"] = meta.get("city")
                rec["all_categories"] = meta.get("all_categories")

            records.append(rec)
        results[day_key] = records

    return results

def print_day_itinerary(day_df, df_stations, df_lines, start_name="START"):

    print("\n🧭 ROUTE\n")

    # -------------------------
    # helper: replace START
    # -------------------------
    def display_name(name):
        return start_name if name == "START" else name

    # -------------------------
    # STEP 1: REMOVE CONSECUTIVE DUPLICATE LOCATIONS
    # -------------------------
    df = day_df.copy()

    df = df.loc[
        (df["type"] != "location") |
        (df["name"] != df["name"].shift())
    ].reset_index(drop=True)

    # -------------------------
    # STEP 2: SPLIT INTO BLOCKS
    # -------------------------
    blocks = []
    current_block = []

    for _, row in df.iterrows():

        current_block.append(row)

        # when we hit a location AFTER stations → block ends
        if row["type"] == "location" and len(current_block) > 1:
            blocks.append(pd.DataFrame(current_block))
            current_block = [row]

    if current_block:
        blocks.append(pd.DataFrame(current_block))

    # -------------------------
    # STEP 3: PRINT EACH BLOCK
    # -------------------------
    last_printed_location = None 
    
    for block in blocks:
    
        loc_rows = block[block["type"] == "location"]
    
        # -------------------------
        # START LOCATION
        # -------------------------
        start_loc = loc_rows.iloc[0]["name"]
    
        if start_loc != last_printed_location:
            print(f"\n📍 {display_name(start_loc)}")
            last_printed_location = start_loc
    
        # -------------------------
        # TRAIN SEGMENT
        # -------------------------
        station_seq = block[block["type"] == "station"]["station_g_cd"]
    
        if len(station_seq) > 0:
            station_seq = station_seq.astype(str).tolist()
    
            segments = compress_route(df_stations, station_seq, df_lines)
    
            for line, stations in segments:
                print(f"\n🚇 {line}")
                print("   " + " → ".join(stations))
    
        # -------------------------
        # END LOCATION
        # -------------------------
        if len(loc_rows) > 1:
            end_loc = loc_rows.iloc[-1]["name"]
    
            if end_loc != last_printed_location:
                print(f"\n📍 {display_name(end_loc)}")
                last_printed_location = end_loc

def save_itinerary_to_json(results, df_locations, start_name="START", filename="itinerary.json"):
    """
    Save full itinerary into JSON file with categories.

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
    df_locations : pd.DataFrame
    start_name : str
    filename : str
    """

    import json

    # 🔥 build lookup: name → categories
    if "all_categories" in df_locations.columns:
        category_map = dict(zip(df_locations["title"], df_locations["all_categories"]))
    else:
        category_map = {}

    json_data = {}

    for day, df in results.items():

        records = []

        for _, row in df.iterrows():

            name = row["name"]
            row_type = row["type"]

            # 🔥 CATEGORY LOGIC
            if name == start_name:
                category = "Accommodation"
            elif row_type == "station":
                category = "NA"
            else:
                category = category_map.get(name, "Unknown")

            # optional: clean list format
            if isinstance(category, list):
                category = ", ".join(category)

            records.append({
                "type": row_type,
                "name": name,
                "lat": row["lat"],
                "lon": row["lon"],
                "categories": category
            })

        json_data[day] = records

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Itinerary saved to {filename}")
