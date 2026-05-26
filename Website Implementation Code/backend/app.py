from flask import Flask, request, jsonify
from flask_cors import CORS
import csv
import pandas as pd
from optimisation import solve_attraction_finding_problem, solve_attraction_finding_weighted, load_attractions_data
from transportation_optimisation import build_multi_day_itinerary

app = Flask(__name__)
CORS(app)

# This is a simple demo optimizer (nearest-neighbor). Replace this with your TSP solver if needed.
def dummy_optimize_route(locations):
    if not locations:
        return []
    remaining = locations.copy()
    route = [remaining.pop(0)]
    while remaining:
        last = route[-1]
        next_idx = min(
            range(len(remaining)),
            key=lambda i: (remaining[i]["lat"] - last["lat"]) ** 2 + (remaining[i]["lon"] - last["lon"]) ** 2,
        )
        route.append(remaining.pop(next_idx))
    return route


@app.route("/api/optimize-route", methods=["POST"])
def optimize_route():
    data = request.get_json() or {}
    locs = data.get("locations", [])
    if not isinstance(locs, list) or any("lat" not in p or "lon" not in p for p in locs):
        return jsonify({"error": "locations must be a list of {lat, lon} objects"}), 400

    optimized = dummy_optimize_route(locs)
    return jsonify({"optimized": optimized})


@app.route("/api/categories")
def get_categories():
    """Get list of unique categories from one-hot encoded columns."""
    try:
        df = pd.read_csv('../data/attractions_with_coordinates.csv')
        # Get all category columns (categories/0, categories/1, etc.)
        category_cols = [col for col in df.columns if 'categories' in col.lower()]
        
        # Collect all unique categories from these columns
        all_categories = set()
        for col in category_cols:
            # Get non-null values and strip whitespace
            cats = df[col].dropna().unique()
            for cat in cats:
                cat_str = str(cat).strip()
                if cat_str:
                    all_categories.add(cat_str)
        
        # Sort alphabetically
        categories = sorted(list(all_categories))
        return jsonify({'categories': categories})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/category-counts")
def get_category_counts():
    """Get count of attractions for each category type used in constraints."""
    try:
        df = pd.read_csv('../data/attractions_with_coordinates.csv')
        
        # Define the categories we care about for constraints
        category_mappings = {
            'shopping_mall': ['shopping mall'],
            'temple': ['temple', 'buddhist temple'],
            'garden': ['garden', 'national park', 'park'],
            'info_centre': ['information centre', 'information center'],
            'shinjuku': [],  # This is based on city name, not category
            'amusement': ['amusement park', 'amusement center', 'amusement centre', 'theme park'],
            'bar': ['bar']
        }
        
        counts = {}
        
        # Count shopping malls
        shopping_cols = [col for col in df.columns if 'categories' in col.lower()]
        shopping_count = 0
        for col in shopping_cols:
            shopping_count += df[col].astype(str).str.lower().str.contains('shopping mall', na=False).sum()
        counts['shopping_mall'] = int(shopping_count)
        
        # Count temples
        temple_count = 0
        for col in shopping_cols:
            temple_count += df[col].astype(str).str.lower().str.contains('temple', na=False).sum()
        counts['temple'] = int(temple_count)
        
        # Count gardens
        garden_count = 0
        for col in shopping_cols:
            garden_count += (df[col].astype(str).str.lower().str.contains('garden', na=False) | 
                           df[col].astype(str).str.lower().str.contains('national park', na=False) |
                           df[col].astype(str).str.lower().str.contains('park', na=False)).sum()
        counts['garden'] = int(garden_count)
        
        # Count info centres
        info_count = 0
        for col in shopping_cols:
            info_count += (df[col].astype(str).str.lower().str.contains('information centre', na=False) |
                          df[col].astype(str).str.lower().str.contains('information center', na=False)).sum()
        counts['info_centre'] = int(info_count)
        
        # Count Shinjuku attractions (based on city name)
        shinjuku_count = df['city'].astype(str).str.lower().str.contains('shinjuku', na=False).sum()
        counts['shinjuku'] = int(shinjuku_count)
        
        # Count amusement parks
        amusement_count = 0
        for col in shopping_cols:
            amusement_count += (df[col].astype(str).str.lower().str.contains('amusement park', na=False) |
                              df[col].astype(str).str.lower().str.contains('amusement center', na=False) |
                              df[col].astype(str).str.lower().str.contains('amusement centre', na=False) |
                              df[col].astype(str).str.lower().str.contains('theme park', na=False)).sum()
        counts['amusement'] = int(amusement_count)
        
        # Count bars
        bar_count = 0
        for col in shopping_cols:
            bar_count += df[col].astype(str).str.lower().str.contains('bar', na=False).sum()
        counts['bar'] = int(bar_count)
        
        return jsonify({'counts': counts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/attractions")
def get_attractions():
    attractions = []
    try:
        df = pd.read_csv('../data/attractions_with_coordinates.csv')
        for idx, row in df.iterrows():
            attractions.append({
                'title': row['title'],
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'categoryName': row.get('categoryName', ''),
                'city': row.get('city', ''),
                'state': row.get('state', ''),
                'totalScore': float(row['totalScore']) if pd.notna(row['totalScore']) else 0,
                'reviewsCount': int(row['reviewsCount']) if pd.notna(row['reviewsCount']) else 0,
                'prices': float(row['prices']) if pd.notna(row['prices']) else 0,
                'address': row.get('address', '')
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(attractions)


@app.route("/api/optimize-attractions", methods=["POST"])
def optimize_attractions():
    """Optimize attraction selection based on budget and constraints."""
    try:
        data = request.get_json() or {}
        budget = data.get('budget', 10000)
        n_select = data.get('n_select', 9)
        category_filter = data.get('category_filter', '')
        constraints = data.get('constraints', [])
        model_type = data.get('model_type', 'normal')
        preference = data.get('preference', 'Premium Experience')
        start_lat = data.get('start_lat')
        start_lon = data.get('start_lon')
        max_attractions_per_day = int(data.get('num_days', 1))
        
        # Load attractions data
        attractions_df = load_attractions_data('../data/attractions_with_coordinates.csv')
        
        # Filter by category if specified
        if category_filter and category_filter.strip():
            # Get all category columns (categories/0, categories/1, etc.)
            category_cols = [col for col in attractions_df.columns if 'categories' in col.lower()]
            
            # Create a mask for rows that contain the category
            mask = False
            for col in category_cols:
                mask = mask | attractions_df[col].astype(str).str.contains(category_filter, case=False, na=False)
            
            attractions_df = attractions_df[mask]
        
        if len(attractions_df) == 0:
            return jsonify({"error": "No attractions found matching filter"}), 400
        
        # Run optimization with selected model
        if model_type == 'weighted':
            selected_df, model, encoded = solve_attraction_finding_weighted(
                attractions_df,
                budget=budget,
                n_select=n_select,
                constraints=constraints,
                preference=preference,
            )
        else:
            selected_df, model, encoded = solve_attraction_finding_problem(
                attractions_df,
                budget=budget,
                n_select=n_select,
                constraints=constraints
            )
        
        if selected_df is None:
            return jsonify({"error": "Optimization failed - no solution found with given constraints"}), 400
        
        # Convert to JSON-serializable format
        selected_list = []
        for idx, row in selected_df.iterrows():
            # Find the row in original attractions_df to get lat/lon
            orig_row = attractions_df[attractions_df['title'] == row['title']].iloc[0]
            selected_list.append({
                'title': row['title'],
                'latitude': float(orig_row['latitude']),
                'longitude': float(orig_row['longitude']),
                'review_score': float(row['review_score']),
                'prices': float(row['prices']),
                'all_categories': str(row['all_categories']),
                'city': orig_row.get('city', ''),
                'state': orig_row.get('state', '')
            })
        
        response_payload = {
            'selected': selected_list,
            'total_score': float(selected_df['review_score'].sum()),
            'total_price': float(selected_df['prices'].sum()),
            'count': len(selected_list)
        }

        # Optional transportation plan (day-wise route including stations)
        if start_lat is not None and start_lon is not None and max_attractions_per_day >= 1:
            try:
                loc_df = pd.DataFrame(
                    [
                        {
                            "title": a["title"],
                            "review_score": a["review_score"],
                            "city": a["city"],
                            "all_categories": a["all_categories"],
                            "latitude": a["latitude"],
                            "longitude": a["longitude"],
                        }
                        for a in selected_list
                    ]
                )

                transport_plan = build_multi_day_itinerary(
                    df_locations=loc_df,
                    max_attractions_per_day=max_attractions_per_day,
                    start_lat=float(start_lat),
                    start_lon=float(start_lon),
                    line_file="../data/line20240426.csv",
                    station_file="../data/station20240426.csv",
                    join_file="../data/join20240426.csv",
                    company_file="../data/company20240328.csv",
                    start_name="START",
                )
                response_payload["transport_plan"] = transport_plan
            except Exception as transport_error:
                response_payload["transport_error"] = str(transport_error)

        return jsonify(response_payload)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
