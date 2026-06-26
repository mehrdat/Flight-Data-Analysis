"""
geo.py
------
Geographic lookups so we can draw maps.

The flight dataset only has airport codes, city names and state names, but no
latitude / longitude. Maps need coordinates, so this file gives us:

  1. AIRPORTS  -> lat/lon for the busiest US airports (so we can plot dots and
                  draw route lines on a map).
  2. STATE_ABBR -> turns a full state name ("California") into the 2-letter code
                  ("CA") that plotly's choropleth map needs.

I keep the list to the busy airports because those cover most of the flights.
Any airport that is not in the table just gets dropped from the map (the normal
charts still use every airport, only the MAP needs coordinates).
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# AIRPORT COORDINATES  (code -> name, latitude, longitude)
# Top ~90 US airports by passenger traffic. Coords are the airport location.
# ---------------------------------------------------------------------------
AIRPORTS: dict[str, dict] = {
    "ATL": {"name": "Atlanta",            "lat": 33.6407, "lon": -84.4277},
    "DFW": {"name": "Dallas/Fort Worth",  "lat": 32.8998, "lon": -97.0403},
    "DEN": {"name": "Denver",             "lat": 39.8561, "lon": -104.6737},
    "ORD": {"name": "Chicago O'Hare",     "lat": 41.9742, "lon": -87.9073},
    "LAX": {"name": "Los Angeles",        "lat": 33.9416, "lon": -118.4085},
    "JFK": {"name": "New York JFK",       "lat": 40.6413, "lon": -73.7781},
    "LAS": {"name": "Las Vegas",          "lat": 36.0840, "lon": -115.1537},
    "MCO": {"name": "Orlando",            "lat": 28.4312, "lon": -81.3081},
    "MIA": {"name": "Miami",              "lat": 25.7959, "lon": -80.2870},
    "CLT": {"name": "Charlotte",          "lat": 35.2140, "lon": -80.9431},
    "SEA": {"name": "Seattle",            "lat": 47.4502, "lon": -122.3088},
    "PHX": {"name": "Phoenix",            "lat": 33.4342, "lon": -112.0116},
    "EWR": {"name": "Newark",             "lat": 40.6895, "lon": -74.1745},
    "SFO": {"name": "San Francisco",      "lat": 37.6213, "lon": -122.3790},
    "IAH": {"name": "Houston Bush",       "lat": 29.9902, "lon": -95.3368},
    "BOS": {"name": "Boston",             "lat": 42.3656, "lon": -71.0096},
    "FLL": {"name": "Fort Lauderdale",    "lat": 26.0742, "lon": -80.1506},
    "MSP": {"name": "Minneapolis",        "lat": 44.8848, "lon": -93.2223},
    "LGA": {"name": "New York LaGuardia", "lat": 40.7769, "lon": -73.8740},
    "DTW": {"name": "Detroit",            "lat": 42.2162, "lon": -83.3554},
    "PHL": {"name": "Philadelphia",       "lat": 39.8744, "lon": -75.2424},
    "SLC": {"name": "Salt Lake City",     "lat": 40.7899, "lon": -111.9791},
    "BWI": {"name": "Baltimore",          "lat": 39.1754, "lon": -76.6684},
    "DCA": {"name": "Washington Reagan",  "lat": 38.8512, "lon": -77.0402},
    "SAN": {"name": "San Diego",          "lat": 32.7338, "lon": -117.1933},
    "IAD": {"name": "Washington Dulles",  "lat": 38.9531, "lon": -77.4565},
    "TPA": {"name": "Tampa",              "lat": 27.9755, "lon": -82.5332},
    "MDW": {"name": "Chicago Midway",     "lat": 41.7868, "lon": -87.7522},
    "DAL": {"name": "Dallas Love",        "lat": 32.8471, "lon": -96.8518},
    "HNL": {"name": "Honolulu",           "lat": 21.3187, "lon": -157.9224},
    "PDX": {"name": "Portland",           "lat": 45.5898, "lon": -122.5951},
    "AUS": {"name": "Austin",             "lat": 30.1975, "lon": -97.6664},
    "STL": {"name": "St. Louis",          "lat": 38.7487, "lon": -90.3700},
    "HOU": {"name": "Houston Hobby",      "lat": 29.6454, "lon": -95.2789},
    "NSH": {"name": "Nashville",          "lat": 36.1245, "lon": -86.6782},
    "BNA": {"name": "Nashville",          "lat": 36.1245, "lon": -86.6782},
    "MSY": {"name": "New Orleans",        "lat": 29.9934, "lon": -90.2580},
    "RDU": {"name": "Raleigh-Durham",     "lat": 35.8776, "lon": -78.7875},
    "SMF": {"name": "Sacramento",         "lat": 38.6951, "lon": -121.5908},
    "SJC": {"name": "San Jose",           "lat": 37.3639, "lon": -121.9289},
    "SNA": {"name": "Santa Ana",          "lat": 33.6757, "lon": -117.8682},
    "MCI": {"name": "Kansas City",        "lat": 39.2976, "lon": -94.7139},
    "OAK": {"name": "Oakland",            "lat": 37.7126, "lon": -122.2197},
    "RSW": {"name": "Fort Myers",         "lat": 26.5362, "lon": -81.7552},
    "CLE": {"name": "Cleveland",          "lat": 41.4117, "lon": -81.8498},
    "PIT": {"name": "Pittsburgh",         "lat": 40.4915, "lon": -80.2329},
    "CVG": {"name": "Cincinnati",         "lat": 39.0489, "lon": -84.6678},
    "IND": {"name": "Indianapolis",       "lat": 39.7173, "lon": -86.2944},
    "CMH": {"name": "Columbus",           "lat": 39.9980, "lon": -82.8919},
    "PBI": {"name": "West Palm Beach",    "lat": 26.6832, "lon": -80.0956},
    "JAX": {"name": "Jacksonville",       "lat": 30.4941, "lon": -81.6879},
    "MKE": {"name": "Milwaukee",          "lat": 42.9472, "lon": -87.8966},
    "ONT": {"name": "Ontario CA",         "lat": 34.0560, "lon": -117.6012},
    "BUR": {"name": "Burbank",            "lat": 34.2007, "lon": -118.3590},
    "ANC": {"name": "Anchorage",          "lat": 61.1743, "lon": -149.9963},
    "OGG": {"name": "Maui",               "lat": 20.8986, "lon": -156.4305},
    "BDL": {"name": "Hartford",           "lat": 41.9389, "lon": -72.6832},
    "ABQ": {"name": "Albuquerque",        "lat": 35.0402, "lon": -106.6092},
    "OMA": {"name": "Omaha",              "lat": 41.3032, "lon": -95.8941},
    "BUF": {"name": "Buffalo",            "lat": 42.9405, "lon": -78.7322},
    "SAT": {"name": "San Antonio",        "lat": 29.5337, "lon": -98.4698},
    "RNO": {"name": "Reno",               "lat": 39.4991, "lon": -119.7681},
    "BOI": {"name": "Boise",              "lat": 43.5644, "lon": -116.2228},
    "MEM": {"name": "Memphis",            "lat": 35.0424, "lon": -89.9767},
    "OKC": {"name": "Oklahoma City",      "lat": 35.3931, "lon": -97.6007},
    "RIC": {"name": "Richmond",           "lat": 37.5052, "lon": -77.3197},
    "ELP": {"name": "El Paso",            "lat": 31.8072, "lon": -106.3781},
    "TUS": {"name": "Tucson",             "lat": 32.1161, "lon": -110.9410},
    "GEG": {"name": "Spokane",            "lat": 47.6199, "lon": -117.5338},
    "CHS": {"name": "Charleston",         "lat": 32.8986, "lon": -80.0405},
    "GRR": {"name": "Grand Rapids",       "lat": 42.8808, "lon": -85.5228},
    "PVD": {"name": "Providence",         "lat": 41.7240, "lon": -71.4282},
    "ORF": {"name": "Norfolk",            "lat": 36.8946, "lon": -76.2012},
    "ALB": {"name": "Albany",             "lat": 42.7483, "lon": -73.8017},
    "SDF": {"name": "Louisville",         "lat": 38.1744, "lon": -85.7360},
    "BHM": {"name": "Birmingham",         "lat": 33.5629, "lon": -86.7535},
    "GSP": {"name": "Greenville",         "lat": 34.8957, "lon": -82.2189},
    "SAV": {"name": "Savannah",           "lat": 32.1276, "lon": -81.2021},
    "DSM": {"name": "Des Moines",         "lat": 41.5340, "lon": -93.6631},
    "TUL": {"name": "Tulsa",              "lat": 36.1984, "lon": -95.8881},
    "FAT": {"name": "Fresno",             "lat": 36.7762, "lon": -119.7181},
    "ROC": {"name": "Rochester",          "lat": 43.1189, "lon": -77.6724},
    "SYR": {"name": "Syracuse",           "lat": 43.1112, "lon": -76.1063},
    "LIT": {"name": "Little Rock",        "lat": 34.7294, "lon": -92.2243},
    "PSP": {"name": "Palm Springs",       "lat": 33.8297, "lon": -116.5067},
    "MYR": {"name": "Myrtle Beach",       "lat": 33.6797, "lon": -78.9283},
    "PWM": {"name": "Portland ME",        "lat": 43.6462, "lon": -70.3093},
    "KOA": {"name": "Kona",               "lat": 19.7388, "lon": -156.0456},
    "LIH": {"name": "Kauai",              "lat": 21.9760, "lon": -159.3390},
    "ICT": {"name": "Wichita",            "lat": 37.6499, "lon": -97.4331},
    "GUM": {"name": "Guam",               "lat": 13.4834, "lon": 144.7960},
}


# ---------------------------------------------------------------------------
# STATE NAME -> 2-letter abbreviation (for the choropleth map)
# ---------------------------------------------------------------------------
STATE_ABBR: dict[str, str] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
    "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "Puerto Rico": "PR", "Virgin Islands": "VI",
}


def airports_dataframe() -> pd.DataFrame:
    """Return the airport table as a pandas DataFrame (code, name, lat, lon)."""
    rows = [
        {"airport": code, "name": v["name"], "lat": v["lat"], "lon": v["lon"]}
        for code, v in AIRPORTS.items()
    ]
    return pd.DataFrame(rows)


def attach_coords(pdf: pd.DataFrame, code_col: str = "origin") -> pd.DataFrame:
    """
    Add lat/lon columns to a pandas dataframe that has an airport code column.
    Rows whose airport is not in our table are dropped (they cannot be mapped).
    """
    air = airports_dataframe().rename(
        columns={"airport": code_col, "lat": "lat", "lon": "lon", "name": "airport_name"}
    )
    out = pdf.merge(air, on=code_col, how="inner")
    return out


def state_to_abbr(name: str) -> str | None:
    """Full state name -> 2 letter code, or None if we don't know it."""
    return STATE_ABBR.get(name)
