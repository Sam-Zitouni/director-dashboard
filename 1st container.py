import os
import json
from datetime import datetime, date
from typing import Optional, Tuple

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import streamlit as st
import plotly.express as px

# ==============================
# PAGE / LAYOUT
# ==============================
st.set_page_config(
    page_title="Director Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Director Dashboard")
st.caption("Strategic decision-making • Partner profitability • Ecosystem monitoring")

# ==============================
# CONNECTION
# ==============================
# Provide two options: single DATABASE_URL or discrete fields via st.secrets
# Fill st.secrets.toml yourself as you requested. Examples:
# [db]
# url = "postgresql+psycopg2://user:pass@host:5432/yourdb"
# adminer_url = "http://accelerantlab.com:8080/?pgsql=db&username=readonly_user&db=rawahel_test"

@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    if "db" in st.secrets and "url" in st.secrets["db"]:
        url = st.secrets["db"]["url"]
    else:
        # fallback to env var (useful for local dev)
        url = os.getenv("DATABASE_URL", "")
    if not url:
        st.stop()
    return create_engine(url, pool_pre_ping=True)

engine = get_engine()

# ==============================
# UTILITIES
# ==============================
@st.cache_data(show_spinner=False)
def fetch_df(query: str, params: Optional[dict] = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params or {})

@st.cache_data(show_spinner=False)
def table_exists(schema: str, table: str) -> bool:
    q = """
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = :schema AND table_name = :table
    """
    df = fetch_df(q, {"schema": schema, "table": table})
    return not df.empty

@st.cache_data(show_spinner=False)
def column_exists(schema: str, table: str, column: str) -> bool:
    q = """
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = :schema AND table_name = :table AND column_name = :column
    """
    df = fetch_df(q, {"schema": schema, "table": table, "column": column})
    return not df.empty

# Helper: safe ratio
def ratio(n, d):
    try:
        n = float(n)
        d = float(d)
        return (n / d) if d else np.nan
    except Exception:
        return np.nan

# ==============================
# SIDEBAR CONTROLS
# ==============================
st.sidebar.header("Filters & Assumptions")

as_of = st.sidebar.date_input("As of date", value=date.today())

period = st.sidebar.selectbox(
    "Analysis window",
    ["MTD", "QTD", "YTD", "Last 30 days", "Last 90 days", "Custom"],
    index=2,
)

custom_start = None
if period == "Custom":
    col1, col2 = st.sidebar.columns(2)
    custom_start = col1.date_input("Start date", value=date(date.today().year, 1, 1))
    as_of = col2.date_input("End date", value=date.today())

# ROFA asset base assumption
asset_base = st.sidebar.number_input(
    "Fleet asset base (currency)",
    min_value=0.0,
    value=float(os.getenv("ASSET_BASE", "0") or 0),
    step=1000.0,
    help="Used in ROFA = Net Profit / Fleet Asset Base. Override if you track asset valuation elsewhere.",
)

# Booking source mapping assumption
source_mapping_note = (
    "If your DB lacks an explicit booking source column, we infer as follows: "
    "Invoices linked to an agency => B2B; otherwise 'single' type => Web/Mobile; you can reclassify below."
)
st.sidebar.caption(source_mapping_note)

source_map = {
    "agency": "B2B",
    "single": "Web/Mobile",
    "pos": "POS",
    "web": "Web",
    "mobile": "Mobile",
}

# Allow manual overrides
manual_overrides = st.sidebar.text_area(
    "Source label overrides (JSON)",
    value=json.dumps(source_map, indent=2),
    height=140,
)
try:
    source_map = json.loads(manual_overrides)
except Exception:
    st.sidebar.warning("Invalid JSON. Using defaults.")

# Helper to resolve period clause

def period_clause(col_name: str) -> Tuple[str, dict]:
    params = {"end": as_of}
    if period == "MTD":
        q = f"AND date_trunc('month', {col_name}) = date_trunc('month', :end) AND {col_name} <= :end"
    elif period == "QTD":
        q = f"AND date_trunc('quarter', {col_name}) = date_trunc('quarter', :end) AND {col_name} <= :end"
    elif period == "YTD":
        q = f"AND date_trunc('year', {col_name}) = date_trunc('year', :end) AND {col_name} <= :end"
    elif period == "Last 30 days":
        q = f"AND {col_name} > :end - INTERVAL '30 days' AND {col_name} <= :end"
    elif period == "Last 90 days":
        q = f"AND {col_name} > :end - INTERVAL '90 days' AND {col_name} <= :end"
    else:  # Custom
        params["start"] = custom_start or date(date.today().year, 1, 1)
        q = f"AND {col_name} BETWEEN :start AND :end"
    return q, params

# ==============================
# DATA PULLS (safe, table-aware)
# ==============================
schema = "public"

# Invoices
if table_exists(schema, "invoices"):
    clause, params = period_clause("date")
    invoices_q = f"""
        SELECT id, invoice_number, date, total_amount, agent_id, payment_status, type, corporate_customer_id, private_trip_id
        FROM public.invoices
        WHERE 1=1 {clause}
    """
    invoices = fetch_df(invoices_q, params)
else:
    invoices = pd.DataFrame(columns=["id","invoice_number","date","total_amount","agent_id","payment_status","type","corporate_customer_id","private_trip_id"])  # empty

# Trip costs
if table_exists(schema, "trip_cost_reports"):
    clause, params = period_clause("date")
    tcost_q = f"""
        SELECT route_id, agent_id, trip_id, type, amount, date
        FROM public.trip_cost_reports
        WHERE 1=1 {clause}
    """
    trip_costs = fetch_df(tcost_q, params)
else:
    trip_costs = pd.DataFrame(columns=["route_id","agent_id","trip_id","type","amount","date"])  

# Agencies
if table_exists(schema, "agencies"):
    agencies = fetch_df("SELECT id, name, type, commission_percent, commission_rate, commission_flat FROM public.agencies")
else:
    agencies = pd.DataFrame(columns=["id","name","type","commission_percent","commission_rate","commission_flat"])  

# Agency reports (debit/credit per agency)
if table_exists(schema, "agency_reports"):
    clause, params = period_clause("date")
    arep_q = f"""
        SELECT agency_id, account_id, description, date, debit, credit, currency_id
        FROM public.agency_reports
        WHERE 1=1 {clause}
    """
    agency_reports = fetch_df(arep_q, params)
else:
    agency_reports = pd.DataFrame(columns=["agency_id","account_id","description","date","debit","credit","currency_id"])  

# Management reports (fleet on/off)
if table_exists(schema, "management_reports"):
    clause, params = period_clause("date")
    man_q = f"""
        SELECT date, vehicles_on, vehicles_off, done_trips, all_trips
        FROM public.management_reports
        WHERE 1=1 {clause}
        ORDER BY date
    """
    mgmt = fetch_df(man_q, params)
else:
    mgmt = pd.DataFrame(columns=["date","vehicles_on","vehicles_off","done_trips","all_trips"])  

# Vehicles & types
if table_exists(schema, "vehicles"):
    vehicles = fetch_df("SELECT id, registration_number, fleet_type_id, status, created_at, updated_at FROM public.vehicles")
else:
    vehicles = pd.DataFrame(columns=["id","registration_number","fleet_type_id","status","created_at","updated_at"])  

if table_exists(schema, "fleet_types"):
    fleet_types = fetch_df("SELECT id, name, total_seat FROM public.fleet_types")
else:
    fleet_types = pd.DataFrame(columns=["id","name","total_seat"])  

# Vehicle events (for timeline)
if table_exists(schema, "vehicle_events"):
    clause, params = period_clause("created_at")
    ve_q = f"""
        SELECT id, type, data, driver_id, trip, reporter_id, vehicle_id, created_at
        FROM public.vehicle_events
        WHERE 1=1 {clause}
        ORDER BY created_at
    """
    v_events = fetch_df(ve_q, params)
else:
    v_events = pd.DataFrame(columns=["id","type","data","driver_id","trip","reporter_id","vehicle_id","created_at"])  

# Routes (optional, for heatmap / RASK)
has_routes = table_exists(schema, "routes")
if has_routes:
    routes_cols = fetch_df("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name='routes'
    """)
    route_has_distance = "distance_km" in set(routes_cols["column_name"]) or "distance" in set(routes_cols["column_name"]) 
    distance_col = "distance_km" if "distance_km" in set(routes_cols["column_name"]) else ("distance" if "distance" in set(routes_cols["column_name"]) else None)
    routes = fetch_df("SELECT id, name " + (f", {distance_col} as distance_km" if distance_col else "") + " FROM public.routes")
else:
    routes = pd.DataFrame(columns=["id","name","distance_km"])  
    route_has_distance = False
    distance_col = None

# ==============================
# DERIVED / METRICS
# ==============================

# Revenue (invoices)
if not invoices.empty:
    invoices["date"] = pd.to_datetime(invoices["date"])
    gross_revenue = invoices["total_amount"].fillna(0).sum()
    bookings = len(invoices)
else:
    gross_revenue = 0.0
    bookings = 0

# Costs (trip_cost_reports)
if not trip_costs.empty:
    total_cost = trip_costs["amount"].fillna(0).sum()
else:
    total_cost = 0.0

# Commission cost estimation
commission_cost = 0.0
if not invoices.empty:
    inv = invoices.copy()
    if not agencies.empty:
        inv = inv.merge(agencies[["id","commission_percent","commission_flat","name"]], left_on="agent_id", right_on="id", how="left", suffixes=("","_agency"))
        # Commission = percent * total_amount + flat
        inv["commission_percent"] = inv["commission_percent"].fillna(0) / 100.0
        inv["commission_flat"] = inv["commission_flat"].fillna(0)
        inv["_comm"] = inv["total_amount"].fillna(0) * inv["commission_percent"] + inv["commission_flat"]
        commission_cost = inv["_comm"].sum()
    elif not agency_reports.empty:
        # Fallback: use debits as commission cost proxy when account is commission account
        commission_cost = agency_reports["debit"].fillna(0).sum()

net_profit = gross_revenue - total_cost - commission_cost

# Fleet utilization & efficiency (from management_reports)
util_rate = np.nan
teep = np.nan  # Equipment Efficiency Index proxy (done_trips / all_trips)
if not mgmt.empty:
    vehicles_on = mgmt["vehicles_on"].fillna(0).sum()
    vehicles_off = mgmt["vehicles_off"].fillna(0).sum()
    done_trips = mgmt["done_trips"].fillna(0).sum()
    all_trips = mgmt["all_trips"].fillna(0).sum()
    util_rate = ratio(vehicles_on, vehicles_on + vehicles_off)
    teep = ratio(done_trips, all_trips)

# ROFA
rofa = ratio(net_profit, asset_base) if asset_base else np.nan

# Booking source breakdown
booking_sources = pd.DataFrame()
if not invoices.empty:
    def infer_source(row):
        # Try invoice.type first, then agency-based inference
        t = (row.get("type") or "").lower()
        mapped = source_map.get(t)
        if mapped:
            return mapped
        return "B2B" if not pd.isna(row.get("agent_id")) else source_map.get("single", "Web/Mobile")
    invoices["booking_source"] = invoices.apply(infer_source, axis=1)
    booking_sources = invoices.groupby("booking_source").agg(bookings=("id","count"), revenue=("total_amount","sum")).reset_index()

# Customer retention (simple): % of corporate customers with >=2 bookings in window
retention_rate = np.nan
if not invoices.empty and "corporate_customer_id" in invoices.columns:
    corp = invoices.dropna(subset=["corporate_customer_id"]).copy()
    if not corp.empty:
        counts = corp.groupby("corporate_customer_id")["id"].count()
        retained = (counts >= 2).sum()
        retention_rate = ratio(retained, counts.shape[0])

# RASK (Revenue per Available Seat-Km)
rask = np.nan
if route_has_distance and not invoices.empty and not fleet_types.empty:
    # We do not have a direct join between invoice and route/vehicle in the provided excerpt.
    # Use trip_cost_reports as a bridge on route_id and agent_id counts as proxy of trips in the period.
    if not trip_costs.empty:
        trips_by_route = trip_costs.groupby("route_id")["trip_id"].nunique().reset_index(name="trips")
        # Available seat-km approximation: avg seats * distance * trips
        avg_seats = fleet_types["total_seat"].fillna(0).mean() if not fleet_types.empty else 0
        routes_nonnull = routes.dropna(subset=["id"]) if not routes.empty else pd.DataFrame(columns=["id","distance_km"]) 
        if distance_col:
            routes_nonnull = routes_nonnull.rename(columns={"distance_km":"distance_km"})
        merged = trips_by_route.merge(routes_nonnull[["id","distance_km"]], left_on="route_id", right_on="id", how="left")
        ask = (avg_seats * merged["distance_km"].fillna(0) * merged["trips"].fillna(0)).sum()
        total_rev = gross_revenue
        rask = ratio(total_rev, ask)

# ==============================
# KPI HEADER
# ==============================
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

kpi1.metric("Gross Revenue", f"{gross_revenue:,.0f}")
kpi2.metric("Net Profit", f"{net_profit:,.0f}")
kpi3.metric("Commission Cost", f"{commission_cost:,.0f}")
kpi4.metric("Fleet Utilization", f"{util_rate*100:,.1f}%" if not np.isnan(util_rate) else "—")
kpi5.metric("Efficiency Index", f"{teep*100:,.1f}%" if not np.isnan(teep) else "—")
kpi6.metric("ROFA", f"{rofa*100:,.1f}%" if not np.isnan(rofa) else "—")

# ==============================
# VISUALS
# ==============================

st.subheader("Route Profitability Heatmap")
if not trip_costs.empty:
    # Profit by route = revenue allocated - cost; revenue allocation proportional by trips per route
    route_rev = pd.DataFrame()
    if not trip_costs.empty:
        trips_by_route_month = (
            trip_costs.assign(month=lambda d: pd.to_datetime(d["date"]).dt.to_period("M").dt.to_timestamp())
                      .groupby(["route_id","month"])["trip_id"].nunique()
                      .reset_index(name="trips")
        )
        # allocate revenue by proportion of trips
        if not invoices.empty:
            total_trips = trips_by_route_month.groupby("month")["trips"].sum().rename("total_trips")
            route_rev = trips_by_route_month.merge(total_trips, on="month", how="left")
            route_rev["alloc_revenue"] = route_rev.apply(lambda r: (r["trips"] / r["total_trips"]) * gross_revenue if r["total_trips"] else 0, axis=1)
        # cost by route-month
        route_cost = (
            trip_costs.assign(month=lambda d: pd.to_datetime(d["date"]).dt.to_period("M").dt.to_timestamp())
                      .groupby(["route_id","month"])['amount'].sum().reset_index(name="cost")
        )
        heat = route_cost.merge(route_rev[["route_id","month","alloc_revenue"]], on=["route_id","month"], how="left")
        heat["profit"] = heat["alloc_revenue"].fillna(0) - heat["cost"].fillna(0)
        # add route name if available
        if not routes.empty:
            heat = heat.merge(routes.rename(columns={"id":"route_id"})[["route_id","name"]], on="route_id", how="left")
        heat_display = heat.copy()
        heat_display["route"] = heat_display.get("name", heat_display["route_id"]).fillna(heat_display["route_id"]).astype(str)
        fig = px.density_heatmap(heat_display, x="month", y="route", z="profit", color_continuous_scale="RdBu")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trip cost data to build the heatmap.")
else:
    st.info("trip_cost_reports table not found or empty.")

st.subheader("Top 5 Agencies Leaderboard")
if not invoices.empty:
    aggs = invoices.groupby("agent_id").agg(revenue=("total_amount","sum"), bookings=("id","count")).reset_index()
    # attach agency names
    if not agencies.empty:
        aggs = aggs.merge(agencies[["id","name"]], left_on="agent_id", right_on="id", how="left")
        aggs["agency"] = aggs["name"].fillna(aggs["agent_id"].astype(str))
    else:
        aggs["agency"] = aggs["agent_id"].astype(str)
    # approximate commissions
    if not agencies.empty:
        aggs = aggs.merge(agencies[["id","commission_percent","commission_flat"]].rename(columns={"id":"agent_id"}), on="agent_id", how="left")
        aggs["commission_percent"] = aggs["commission_percent"].fillna(0) / 100
        aggs["commission_flat"] = aggs["commission_flat"].fillna(0)
        # assume flat applies per booking
        aggs["commission_cost"] = aggs["revenue"] * aggs["commission_percent"] + aggs["commission_flat"] * aggs["bookings"]
    else:
        aggs["commission_cost"] = 0.0
    # attach cost share by proportional trips
    if not trip_costs.empty:
        trips_by_agency = trip_costs.groupby("agent_id")["trip_id"].nunique().reset_index(name="trips")
        total_trips = trips_by_agency["trips"].sum()
        aggs = aggs.merge(trips_by_agency, on="agent_id", how="left")
        aggs["trips"] = aggs["trips"].fillna(0)
        total_cost_share = total_cost
        aggs["cost_share"] = aggs.apply(lambda r: (r["trips"] / total_trips) * total_cost_share if total_trips else 0, axis=1)
    else:
        aggs["cost_share"] = 0.0
    aggs["profit"] = aggs["revenue"] - aggs["commission_cost"] - aggs["cost_share"]
    aggs_top = aggs.sort_values("profit", ascending=False).head(5)
    fig = px.bar(aggs_top, x="agency", y=["revenue","commission_cost","cost_share","profit"], barmode="group")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(aggs_top[["agency","bookings","revenue","commission_cost","cost_share","profit"]].round(2))
else:
    st.info("No invoices available to compute agency leaderboard.")

st.subheader("Monthly Financial Trendline (Bookings vs Revenue vs Cost)")
if not invoices.empty:
    rev_month = invoices.assign(month=lambda d: pd.to_datetime(d["date"]).dt.to_period("M").dt.to_timestamp()) \
                          .groupby("month").agg(revenue=("total_amount","sum"), bookings=("id","count")).reset_index()
else:
    rev_month = pd.DataFrame(columns=["month","revenue","bookings"])  

if not trip_costs.empty:
    cost_month = trip_costs.assign(month=lambda d: pd.to_datetime(d["date"]).dt.to_period("M").dt.to_timestamp()) \
                             .groupby("month").agg(cost=("amount","sum")).reset_index()
else:
    cost_month = pd.DataFrame(columns=["month","cost"])  

trend = rev_month.merge(cost_month, on="month", how="outer").fillna(0)
trend = trend.sort_values("month")
fig = px.line(trend, x="month", y=["bookings","revenue","cost"], markers=True)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Fleet Lifecycle Usage Timeline")
if not v_events.empty:
    v_events["created_at"] = pd.to_datetime(v_events["created_at"])
    # Simplify type for timeline label
    v_simple = v_events.copy()
    v_simple["event"] = v_simple["type"].fillna("event")
    if not vehicles.empty:
        v_simple = v_simple.merge(vehicles[["id","registration_number"]], left_on="vehicle_id", right_on="id", how="left")
        v_simple["vehicle"] = v_simple["registration_number"].fillna(v_simple["vehicle_id"].astype(str))
    else:
        v_simple["vehicle"] = v_simple["vehicle_id"].astype(str)
    fig = px.scatter(v_simple, x="created_at", y="vehicle", color="event", hover_data=["trip","reporter_id","driver_id"])
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(v_simple[["created_at","vehicle","event","trip","driver_id","reporter_id"]].sort_values("created_at"))
else:
    st.info("No vehicle events to plot.")

# ==============================
# PARTNER / AGENCY PROFITABILITY MATRIX
# ==============================
st.subheader("Partner / Agency Profitability Matrix")

if not invoices.empty:
    ag_metrics = invoices.groupby("agent_id").agg(
        revenue=("total_amount","sum"),
        bookings=("id","count")
    ).reset_index()
    if not agencies.empty:
        ag_metrics = ag_metrics.merge(agencies[["id","name"]], left_on="agent_id", right_on="id", how="left")
        ag_metrics["agency"] = ag_metrics["name"].fillna(ag_metrics["agent_id"].astype(str))
    else:
        ag_metrics["agency"] = ag_metrics["agent_id"].astype(str)

    # attach commission and cost share (same approach as leaderboard)
    if not agencies.empty:
        ag_metrics = ag_metrics.merge(agencies[["id","commission_percent","commission_flat"]].rename(columns={"id":"agent_id"}), on="agent_id", how="left")
        ag_metrics["commission_percent"] = ag_metrics["commission_percent"].fillna(0) / 100
        ag_metrics["commission_flat"] = ag_metrics["commission_flat"].fillna(0)
        ag_metrics["commission_cost"] = ag_metrics["revenue"] * ag_metrics["commission_percent"] + ag_metrics["commission_flat"] * ag_metrics["bookings"]
    else:
        ag_metrics["commission_cost"] = 0.0

    if not trip_costs.empty:
        trips_by_agency = trip_costs.groupby("agent_id")["trip_id"].nunique().reset_index(name="trips")
        total_trips = trips_by_agency["trips"].sum()
        ag_metrics = ag_metrics.merge(trips_by_agency, on="agent_id", how="left")
        ag_metrics["trips"] = ag_metrics["trips"].fillna(0)
        ag_metrics["cost_share"] = ag_metrics.apply(lambda r: (r["trips"] / total_trips) * total_cost if total_trips else 0, axis=1)
    else:
        ag_metrics["cost_share"] = 0.0

    ag_metrics["profit"] = ag_metrics["revenue"] - ag_metrics["commission_cost"] - ag_metrics["cost_share"]

    fig = px.scatter(ag_metrics, x="revenue", y="profit", size="bookings", hover_name="agency")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(ag_metrics[["agency","bookings","revenue","commission_cost","cost_share","profit"]].round(2))
else:
    st.info("No agency metrics available (invoices empty).")

# ==============================
# BOOKING SOURCE BREAKDOWN & RASK & RETENTION
# ==============================
colA, colB, colC = st.columns(3)

with colA:
    st.markdown("### Booking Source Breakdown")
    if not booking_sources.empty:
        fig = px.pie(booking_sources, names="booking_source", values="bookings", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(booking_sources)
    else:
        st.info("No booking source data available.")

with colB:
    st.markdown("### RASK (Rev/ASK)")
    if not np.isnan(rask):
        st.metric("RASK", f"{rask:,.4f}")
    else:
        st.info("Insufficient route distance / seat data to compute RASK.")

with colC:
    st.markdown("### Customer Retention Rate")
    if not np.isnan(retention_rate):
        st.metric("Retention", f"{retention_rate*100:,.1f}%")
    else:
        st.info("Insufficient customer identifiers in invoices to compute retention.")

# ==============================
# QUICK ACTIONS / SHORTCUTS
# ==============================
st.subheader("Quick Actions / Shortcuts")
a1, a2, a3, a4 = st.columns(4)

with a1:
    if st.button("🔍 View agency performance drilldown"):
        st.session_state["show_agency_drill"] = True
with a2:
    if st.button("📊 Download revenue report"):
        st.session_state["download_revenue"] = True
with a3:
    if st.button("🗺 Explore route analytics"):
        st.session_state["show_route_analytics"] = True
with a4:
    adminer_url = (st.secrets.get("db", {}).get("adminer_url") if "db" in st.secrets else None) or os.getenv("ADMINER_URL")
    st.link_button("✏️ Edit strategic pricing policy", adminer_url or "https://www.adminer.org/")

# Drilldown: Agency performance
if st.session_state.get("show_agency_drill"):
    st.markdown("## Agency Performance Drilldown")
    if not invoices.empty:
        sel_agency = st.selectbox("Select agency", options=sorted(invoices["agent_id"].dropna().unique()))
        detail = invoices[invoices["agent_id"] == sel_agency].copy()
        st.dataframe(detail.sort_values("date", ascending=False))
    else:
        st.info("No invoices.")

# Download: Revenue report (CSV)
if st.session_state.get("download_revenue"):
    if not invoices.empty:
        csv = invoices.to_csv(index=False).encode("utf-8")
        st.download_button("Download Invoices CSV", data=csv, file_name="revenue_report.csv", mime="text/csv")
    else:
        st.info("Nothing to download.")

# Route analytics
if st.session_state.get("show_route_analytics"):
    st.markdown("## Route Analytics")
    if not trip_costs.empty:
        avail_routes = sorted(trip_costs["route_id"].dropna().unique())
        pick = st.selectbox("Route", options=avail_routes)
        subset = trip_costs[trip_costs["route_id"] == pick].copy()
        subset["month"] = pd.to_datetime(subset["date"]).dt.to_period("M").dt.to_timestamp()
        cost_by_month = subset.groupby("month")["amount"].sum().reset_index()
        fig = px.bar(cost_by_month, x="month", y="amount")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trip cost data available.")

st.caption("Built with ❤️ in Streamlit • Data freshness depends on your database.")
