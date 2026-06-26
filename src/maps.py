"""
maps.py
-------
Geographic MAPS with plotly. These are the "wow" visuals for the report.

Three map types:
  1. airport_bubble_map   - a dot for each airport, size = how busy, colour =
                            average delay. Shows WHERE the busy/late airports are.
  2. route_map            - lines drawn between origin and destination for the
                            busiest routes, so you see the network.
  3. state_choropleth     - colours each US state by its average delay.

Maps need latitude/longitude / state codes, which the raw data doesn't have,
so we join on the lookup tables in geo.py.

Each function saves an interactive .html (open in a browser) AND tries to save
a .png (needs the 'kaleido' package: pip install kaleido). If kaleido is
missing it just skips the PNG and keeps the HTML.

All functions take a PANDAS dataframe (already aggregated in Spark).
"""

from src import config
from src import geo

try:
    import plotly.graph_objects as go
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False


def _save(fig, filename: str):
    """Save an interactive HTML and (if possible) a PNG into outputs/figures."""
    base = config.FIG_DIR / filename
    html_path = base.with_suffix(".html")
    fig.write_html(str(html_path))
    print(f"saved -> {html_path}")
    try:
        png_path = base.with_suffix(".png")
        fig.write_image(str(png_path), width=1100, height=700, scale=2)
        print(f"saved -> {png_path}")
    except Exception as e:
        print(f"(PNG skipped - install 'kaleido' to enable static images: {e})")


def _check():
    if not _HAS_PLOTLY:
        raise ImportError("plotly is not installed. Run: pip install plotly kaleido")


# ---------------------------------------------------------------------------
# 1. AIRPORT BUBBLE MAP
# ---------------------------------------------------------------------------
def airport_bubble_map(pdf, code_col="origin", size_col="flights",
                       color_col="avg_arr_delay", filename="map_airport_bubbles"):
    """
    pdf must have an airport-code column + a size column + a colour column.
    Example: analysis.delays_by_origin_airport(df).toPandas()
    """
    _check()
    data = geo.attach_coords(pdf, code_col=code_col)
    fig = px.scatter_geo(
        data, lat="lat", lon="lon",
        size=size_col, color=color_col,
        hover_name="airport_name",
        hover_data={code_col: True, size_col: ":,", color_col: ":.1f",
                    "lat": False, "lon": False},
        color_continuous_scale="RdYlGn_r",
        scope="usa",
        title="US airports - bubble size = flights, colour = avg arrival delay (min)",
        size_max=40,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    _save(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# 2. ROUTE MAP (lines between airports)
# ---------------------------------------------------------------------------
def route_map(pdf, filename="map_routes"):
    """
    pdf must have origin, dest, flights (and ideally avg_arr_delay).
    Example: advanced_analysis.busiest_routes_volume(df, top=40).toPandas()
    Draws a line for each route; thicker/greener = busier.
    """
    _check()
    air = geo.AIRPORTS
    fig = go.Figure()

    # add the route lines
    max_flights = pdf["flights"].max()
    for _, r in pdf.iterrows():
        o, d = r["origin"], r["dest"]
        if o not in air or d not in air:
            continue
        width = 1 + 6 * (r["flights"] / max_flights)
        fig.add_trace(go.Scattergeo(
            lon=[air[o]["lon"], air[d]["lon"]],
            lat=[air[o]["lat"], air[d]["lat"]],
            mode="lines",
            line=dict(width=width, color="#2a6f97"),
            opacity=0.5,
            hoverinfo="text",
            text=f"{o}-{d}: {int(r['flights']):,} flights",
        ))

    # add airport dots on top
    codes = set(pdf["origin"]).union(set(pdf["dest"]))
    lats = [air[c]["lat"] for c in codes if c in air]
    lons = [air[c]["lon"] for c in codes if c in air]
    names = [c for c in codes if c in air]
    fig.add_trace(go.Scattergeo(
        lon=lons, lat=lats, mode="markers+text",
        marker=dict(size=5, color="#bc4749"),
        text=names, textposition="top center",
        textfont=dict(size=8), hoverinfo="text",
    ))

    fig.update_layout(
        title_text="Busiest flight routes (line thickness = number of flights)",
        showlegend=False, geo=dict(scope="usa"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    _save(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# 3. STATE CHOROPLETH
# ---------------------------------------------------------------------------
def state_choropleth(pdf, state_col="origin_state_nm", value_col="avg_arr_delay",
                     filename="map_state_choropleth"):
    """
    Colour each US state by a value (default: average arrival delay).
    pdf example: analysis.delays_by_state(df).toPandas()
    """
    _check()
    data = pdf.copy()
    data["state_code"] = data[state_col].map(geo.STATE_ABBR)
    data = data.dropna(subset=["state_code"])
    fig = px.choropleth(
        data, locations="state_code", locationmode="USA-states",
        color=value_col, scope="usa",
        color_continuous_scale="RdYlGn_r",
        hover_name=state_col,
        title=f"Average arrival delay by state ({value_col})",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    _save(fig, filename)
    return fig
