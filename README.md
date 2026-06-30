# BC2411 - Tokyo Attractions Optimiser 🇯🇵

A web-based itinerary optimisation system that helps travellers plan their Tokyo trip intelligently using mathematical optimisation and route planning techniques.

The application recommends attractions based on budget, preferences, and constraints while generating efficient multi-day travel routes across Tokyo’s metro system.

---

# ✨ Features

- 🎯 Attraction recommendation using Gurobi Linear Optimisation
- 💰 Budget-aware itinerary planning
- 🧠 Weighted preference models
  - Premium Experience
  - Balanced
  - Budget-Friendly
- 🚆 Multi-day transportation route generation
- 🗺️ Interactive map visualisation with Folium
- 📍 Tokyo Metro integration
- 🏙️ Attraction filtering by categories and districts
- ⚡ Flask REST API backend
- 🌐 React frontend interface

---

## 📦 Prerequisites

- **Python Version:** 3.10.11
- **Gurobi Academic License:** [Request one here](https://www.gurobi.com/academia/academic-program-and-licenses/)
- **Virtual Environment:** Create and activate a Python virtual environment

---

## 📘 Project Overview

Planning a trip around Tokyo can be challenging due to its extensive railway network and large number of attractions. Existing navigation tools can provide directions but do not optimise an itinerary based on a traveller's budget, preferences, or travel efficiency.

This project combines **mathematical optimisation** and **route planning** to automatically generate personalised multi-day itineraries. Given user-defined constraints such as budget, attraction preferences, and trip duration, the system recommends attractions and computes efficient travel routes across Tokyo's metro network.

The system consists of two major optimisation components:

1. **Attraction Recommendation Model**
   - Recommends attractions based on constraints and user preferences
   - Supports weighted and unweighted optimisation approaches

2. **Transportation Optimisation Model**
   - Generates efficient travel routes between attractions
   - Uses TSP with MTZ constraints to minimise travel distance

---

# 📊 Data Sources

## Attraction Recommendation Dataset

The attraction recommendation dataset was extracted from Google Maps using Apify’s Google Maps Scraper. The dataset contains various attributes relevant to itinerary planning, including:

- Attraction names
- Review scores
- Review counts
- Attraction categories
- Address-related information

Additional preprocessing and data transformation steps included:

- Geocoding attractions using the Geopy library
- Manual collection of attraction entrance fees
- Address concatenation for standardised geolocation processing
- Missing value detection and removal

These preprocessing and cleaning procedures were implemented in:

```text
Datasets and Code/LOP Python Code/data_cleaning.ipynb
```

Final dataset size:
- **63 cleaned attractions** after preprocessing

---

## Transportation Optimisation Dataset

Railway datasets sourced from Kaggle include:

- `station.csv`
- `join.csv`
- `line.csv`

These datasets are used to:
- Build Tokyo railway network graphs
- Match nearest stations
- Generate feasible travel paths
- Perform line-aware transportation routing

---

# 📐 Model Formulation

## 1️⃣ Attraction Recommendation Model

Found in:

```text
Datasets and Code/LOP Python Code/finalised_arbitary_constraints.ipynb
Datasets and Code/LOP Python Code/tokyoattraction_defined_constraints.ipynb
```

Included:

1. Interactive Folium map visualisations to display all recommended attractions
2. Formula prompts generated for AI-assisted optimisation code formulation
3. A generalised optimisation framework supporting multiple user-defined attraction constraints
4. Weighted and unweighted attraction recommendation models
5. Budget-aware and preference-aware optimisation
6. Constraint feasibility checking and infeasibility diagnosis functions

## 2️⃣ Transportation Recommendation Model

Found in: 

``` text
Datasets and Code/LOP Python Code/Transportation optimisation.ipynb
```

Included:

1. Attraction clustering using angular grouping techniques to group attractions into multi-day itineraries based on geographical proximity
2. Integration of Tokyo Metro railway datasets sourced from Kaggle
3. Travelling Salesman Problem (TSP) formulation to identify the shortest travel distance within each cluster
4. Miller-Tucker-Zemlin (MTZ) constraints for subtour elimination
5. Railway-aware transportation routing between attractions
6. Interactive route visualisation using Leaflet and OpenStreetMap

---

# 🖥️ How to Run the Application

👨‍💻 1. Run frontend in the terminal:
``` text 
cd "Website Implementation Code/frontend"
start index.html
```

🔧 2. Run backend in the terminal:
``` text
cd ..
cd "backend"
pip install -r requirements.txt
python app.py
```

🧐 3. Go back to the html and start exploring the constraints combination you are looking for!!