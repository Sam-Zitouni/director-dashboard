# This cell creates a NEW extended Streamlit app with all Director KPIs & visuals,
# designed to work against your live PostgreSQL DB (same one you access via Adminer).
# It detects available columns at runtime and gracefully degrades with clear messages
# when a KPI needs fields that your schema doesn't have.
#
# Run:  streamlit run director_dashboard_app_extended.py

from textwrap import dedent

extended_code = dedent(r'''
import os
import datetime as dt
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import streamlit as st
import matplotlib.pyplot as plt

# =========================
# Utilities
# =========================
def get_env(name: str, default: str = "") -> str:
    v = os.environ.get(name, "").strip()
    return v if v else default

def pg_url(host: str, port: str, db: str, user: str, pwd: str) -> str:
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"

@st.cache_resource(show_spinner=False)
def get_engine(host: str, port: str, db: str, user: str, pwd: str) -> Engine:
    return create_engine(pg_url(host, port, db, user, pwd), pool_pre_ping=True)

def safe_read_sql(engine: Engine, sql: str, params: dict = None) -> pd.DataFrame:
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params or {})
    except Exception as e:
        st.warning(f"Query failed: {e}")
        return pd.DataFrame()

def table_columns(engine: Engine, table: str) -> List[str]:
    q = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = :t
    """
    df = safe_read_sql(engine, q, {"t": table})
    return [c for c in df["column_name"].tolist()] if not df.empty else []

def has_cols(engine: Engine, table: str, cols: List[str]) -> bool:
    tcols = set([c.lower() for c in table_columns(engine, table)])
    return all(c.lower() in tcols for c in cols)

def kpi(label: str, value, help_text: Optional[str] = None):
    st.metric(label, value, help=help_text)

def line_chart_df(df: pd.DataFrame, x: str, y: str, title: str):
    if df.empty or x not in df or y not in df:
        st.info("No data to plot.")
        return
    fig, ax = plt.subplots()
    ax.plot(df[x], df[y])
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    st.pyplot(fig)

def bar_chart_df(df: pd.DataFrame, x: str, y: str, title: str, rotate: bool = True):
    if df.empty or x not in df or y not in df:
        st.info("No data to plot.")
        return
    fig, ax = plt.subplots()
    ax.bar(df[x], df[y])
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if rotate:
        plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

def heatmap_from_pivot(pivot_df: pd.DataFrame, title: str):
    if pivot_df.empty:
        st.info("No data to plot.")
        return
    fig, ax = plt.subplots()
    im = ax.imshow(pivot_df.values, aspect="auto")
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index)
    ax.set_title(title)
    st.pyplot(fig)

def default_dates() -> Tuple[dt.date, dt.date]:
    today = dt.date.today()
    start = today.replace(day=1) - dt.timedelta(days=90)  # last ~3 months
    return start, today

# =========================
# App Setup
# =========================
st.set_page_config(page_title="Director Dashboard — Extended (Veyn BI)", layout="wide")
st.title("👤 Director Dashboard — Extended")
st.caption("Strategic decision-making • Partner profitability • Ecosystem monitoring")

with st.sidebar:
    st.header("Database")
    host = st.text_input("PGHOST", get_env("PGHOST", "localhost"))
    port = st.text_input("PGPORT", get_env("PGPORT", "5432"))
    db   = st.text_input("PGDATABASE", get_env("PGDATABASE", "postgres"))
    user = st.text_input("PGUSER", get_env("PGUSER", "postgres"))
    pwd  = st.text_input("PGPASSWORD", type="password", value=get_env("PGPASSWORD", ""))

    st.header("Filters")
    start_d, end_d = default_dates()
    start_date = st.date_input("Start date", start_d)
    end_date = st.date_input("End date", end_d)

    st.header("Assumptions / Inputs")
    avg_route_distance_km = st.number_input("Avg Route Distance (km) — for RASK", min_value=1.0, value=250.0, step=10.0)
    total_fleet_value = st.number_input("Total Fleet Value (currency) — for ROFA", min_value=0.0, value=1_000_000.0, step=10_000.0)

    st.caption("These inputs are used when the DB lacks explicit fields (e.g., route distance, asset value).")

if not all([host, port, db, user, pwd]):
    st.stop()

engine = get_engine(host, port, db, user, pwd)

# Preload schema knowledge
tbl_cols: Dict[str, List[str]] = {}
for t in ["payments", "wallet_transactions", "invoices", "trip_cost_reports",
          "management_reports", "vehicles", "fleet_types", "vehicle_events",
          "agencies", "agency_reports", "agency_commissions", "customers",
          "loyalty_points", "bookings", "routes", "booking_transactions"]:
    tbl_cols[t] = table_columns(engine, t)

# =========================
# FINANCE: Revenue, Costs, Net Profit
# =========================
st.subheader("💰 Finance")

# Revenue from payments
payments_q = """
SELECT
  DATE(created_at) AS d,
  SUM(CASE WHEN direction = 'in' THEN amount ELSE 0 END) AS revenue_in,
  SUM(CASE WHEN direction = 'out' THEN amount ELSE 0 END) AS payouts_out
FROM payments
WHERE created_at::date BETWEEN :sd AND :ed
GROUP BY 1
ORDER BY 1;
"""
df_pay = safe_read_sql(engine, payments_q, {"sd": start_date, "ed": end_date})
gross_rev = float(df_pay["revenue_in"].sum()) if not df_pay.empty else 0.0
payouts = float(df_pay["payouts_out"].sum()) if not df_pay.empty else 0.0
kcols = st.columns(3)
with kcols[0]:
    kpi("Gross Revenue", f"{gross_rev:,.2f}")
with kcols[1]:
    kpi("Payouts / Refunds", f"{payouts:,.2f}")
# Operating costs
costs_q = """
SELECT date AS d, SUM(amount) AS operating_cost
FROM trip_cost_reports
WHERE date BETWEEN :sd AND :ed
GROUP BY 1 ORDER BY 1;
"""
df_costs = safe_read_sql(engine, costs_q, {"sd": start_date, "ed": end_date})
oper_cost = float(df_costs["operating_cost"].sum()) if not df_costs.empty else 0.0
with kcols[2]:
    kpi("Operating Costs", f"{oper_cost:,.2f}")

net_profit = (gross_rev - payouts) - oper_cost
kpi("Approx. Net Profit", f"{net_profit:,.2f}", help_text="= Net inflow (payments) − operating costs")

st.write("**Revenue Trend**")
tmp = df_pay.rename(columns={"d":"date"})
line_chart_df(tmp, "date", "revenue_in", "Daily Revenue (payments.direction='in')")

st.write("**Operating Cost Trend**")
tmp2 = df_costs.rename(columns={"d":"date"})
line_chart_df(tmp2, "date", "operating_cost", "Daily Operating Costs (trip_cost_reports)")

# =========================
# COMMISSION COST (from agency_reports / commissions)
# =========================
st.subheader("💼 Commission Cost & Agencies")

# Commission cost proxy from agency_reports (debit as commission/costs depending on accounting)
agency_sum_q = """
WITH ar AS (
  SELECT agency_id,
         SUM(COALESCE(credit,0)) AS credit_sum,
         SUM(COALESCE(debit,0))  AS debit_sum
  FROM agency_reports
  WHERE date BETWEEN :sd AND :ed
  GROUP BY 1
)
SELECT a.id, a.name,
       COALESCE(ar.credit_sum,0) AS credit_sum,
       COALESCE(ar.debit_sum,0)  AS debit_sum,
       COALESCE(ar.credit_sum,0) - COALESCE(ar.debit_sum,0) AS net_agency
FROM agencies a
LEFT JOIN ar ON ar.agency_id = a.id
ORDER BY net_agency DESC NULLS LAST;
"""
df_ag = safe_read_sql(engine, agency_sum_q, {"sd": start_date, "ed": end_date})
if not df_ag.empty:
    # Commission cost assumption: use debit_sum (if your accounting marks commissions as debits)
    commission_cost = float(df_ag["debit_sum"].sum())
else:
    commission_cost = 0.0
kpi("Commission Cost (proxy)", f"{commission_cost:,.2f}", "Using debit_sum from agency_reports")

if not df_ag.empty:
    st.write("**Top Agencies by Net (Credit − Debit)**")
    top10 = df_ag.head(10).copy()
    bar_chart_df(top10, "name", "net_agency", "Top 10 Agencies Leaderboard")
st.dataframe(df_ag)

# =========================
# FLEET UTILIZATION & EQUIPMENT EFFICIENCY
# =========================
st.subheader("🚌 Fleet Utilization & Equipment Efficiency")

mgmt_q = """
SELECT date AS d,
       COALESCE(vehicles_on,0) AS vehicles_on,
       COALESCE(vehicles_off,0) AS vehicles_off,
       COALESCE(done_trips,0) AS done_trips,
       COALESCE(all_trips,0) AS all_trips
FROM management_reports
WHERE date BETWEEN :sd AND :ed
ORDER BY 1;
"""
df_mgmt = safe_read_sql(engine, mgmt_q, {"sd": start_date, "ed": end_date})
if not df_mgmt.empty:
    df_mgmt["utilization"] = np.where((df_mgmt["vehicles_on"] + df_mgmt["vehicles_off"])>0,
                                      df_mgmt["vehicles_on"] / (df_mgmt["vehicles_on"] + df_mgmt["vehicles_off"]),
                                      np.nan)
    kpi("Avg Fleet Utilization", f"{df_mgmt['utilization'].mean()*100:,.1f}%")
    line_chart_df(df_mgmt.rename(columns={"d":"date"}), "date", "utilization", "Fleet Utilization Over Time")
else:
    st.info("No data in management_reports for the selected period.")

# Equipment Efficiency Index = 1 - (failures / trips)
veh_event_cols = tbl_cols.get("vehicle_events", [])
fail_q = None
if veh_event_cols:
    # Try to detect failure/breakdown by the type column if present
    if "type" in [c.lower() for c in veh_event_cols]:
        fail_q = """
        SELECT DATE(created_at) AS d, COUNT(*) AS failures
        FROM vehicle_events
        WHERE created_at::date BETWEEN :sd AND :ed
          AND (LOWER(type) LIKE '%%fail%%' OR LOWER(type) LIKE '%%break%%')
        GROUP BY 1 ORDER BY 1;
        """
    else:
        # If there's no "type" column, count all events as potential reliability incidents (coarse)
        fail_q = """
        SELECT DATE(created_at) AS d, COUNT(*) AS failures
        FROM vehicle_events
        WHERE created_at::date BETWEEN :sd AND :ed
        GROUP BY 1 ORDER BY 1;
        """
df_fail = safe_read_sql(engine, fail_q, {"sd": start_date, "ed": end_date}) if fail_q else pd.DataFrame()

if not df_mgmt.empty and not df_fail.empty:
    df_e = pd.merge(df_mgmt[["d","all_trips"]], df_fail, on="d", how="left").fillna({"failures":0})
    df_e["efficiency_index"] = np.where(df_e["all_trips"]>0, 1 - (df_e["failures"]/df_e["all_trips"]), np.nan)
    kpi("Equipment Efficiency (avg)", f"{df_e['efficiency_index'].mean()*100:,.1f}%")
    line_chart_df(df_e.rename(columns={"d":"date"}), "date", "efficiency_index", "Equipment Efficiency Index Over Time")
else:
    st.info("Cannot compute Equipment Efficiency Index (need management_reports.all_trips and vehicle_events).")

# =========================
# RASK (Revenue per Available Seat Kilometer)
# =========================
st.subheader("📏 RASK (Revenue per Available Seat Kilometer)")

# Available seats = sum(fleet_types.total_seat * active vehicles factor)
# We don't have distance per route; we use sidebar avg_route_distance_km and trips count as proxy.
seats_q = """
SELECT ft.id AS fleet_type_id, ft.total_seat, COUNT(v.id) AS vehicles_count
FROM fleet_types ft
LEFT JOIN vehicles v ON v.fleet_type_id = ft.id
GROUP BY 1,2
"""
df_seats = safe_read_sql(engine, seats_q, {})
total_seats = int((df_seats["total_seat"] * df_seats["vehicles_count"]).sum()) if not df_seats.empty else 0

trips_total = int(df_mgmt["done_trips"].sum()) if not df_mgmt.empty else 0
available_seat_km = total_seats * max(trips_total, 1) * float(avg_route_distance_km)
rask = (gross_rev / available_seat_km) if available_seat_km > 0 else np.nan

kpi("Total Seats (fleet)", f"{total_seats:,}")
kpi("Trips (period)", f"{trips_total:,}")
kpi("RASK", f"{rask:,.6f}" if not np.isnan(rask) else "n/a",
    help_text="Revenue / (Seats × Trips × Avg Distance) — proxy due to missing route distance")

# =========================
# BOOKING SOURCE BREAKDOWN
# =========================
st.subheader("🧭 Booking Source Breakdown")

# Try bookings.source or payments.channel/platform/origin
source_df = pd.DataFrame()
if "bookings" in tbl_cols and any(c in [x.lower() for x in tbl_cols["bookings"]] for c in ["source","channel","platform","origin"]):
    # Build query with whichever column exists
    bcols = [c.lower() for c in tbl_cols["bookings"]]
    src_col = "source" if "source" in bcols else ("channel" if "channel" in bcols else ("platform" if "platform" in bcols else "origin"))
    q = f"""
    SELECT COALESCE({src_col}, 'unknown') AS src, COUNT(*) AS n
    FROM bookings
    WHERE COALESCE(created_at, updated_at)::date BETWEEN :sd AND :ed
    GROUP BY 1 ORDER BY 2 DESC
    """
    source_df = safe_read_sql(engine, q, {"sd": start_date, "ed": end_date})
elif "payments" in tbl_cols and any(c in [x.lower() for x in tbl_cols["payments"]] for c in ["channel","source","platform","origin"]):
    pcols = [c.lower() for c in tbl_cols["payments"]]
    src_col = "channel" if "channel" in pcols else ("source" if "source" in pcols else ("platform" if "platform" in pcols else "origin"))
    q = f"""
    SELECT COALESCE({src_col}, 'unknown') AS src, COUNT(*) AS n
    FROM payments
    WHERE created_at::date BETWEEN :sd AND :ed
    GROUP BY 1 ORDER BY 2 DESC
    """
    source_df = safe_read_sql(engine, q, {"sd": start_date, "ed": end_date})

if not source_df.empty:
    bar_chart_df(source_df, "src", "n", "Booking Source Breakdown")
    st.dataframe(source_df)
else:
    st.info("No explicit source/channel column found on bookings or payments — using sidebar assumption.")
    st.caption("Add a source/channel column (bookings.source or payments.channel) to enable this KPI.")

# =========================
# CUSTOMER RETENTION RATE
# =========================
st.subheader("🔁 Customer Retention Rate")

ret_df = pd.DataFrame()
if "bookings" in tbl_cols and has_cols(engine, "bookings", ["customer_id"]):
    q = """
    WITH b AS (
      SELECT customer_id, DATE(COALESCE(created_at, updated_at)) AS d
      FROM bookings
      WHERE COALESCE(created_at, updated_at)::date BETWEEN :sd AND :ed
        AND customer_id IS NOT NULL
    )
    SELECT
      COUNT(DISTINCT customer_id) AS total_customers,
      COUNT(*) FILTER (WHERE cnt >= 2) AS returning_customers
    FROM (
      SELECT customer_id, COUNT(*) AS cnt FROM b GROUP BY 1
    ) t;
    """
    ret_df = safe_read_sql(engine, q, {"sd": start_date, "ed": end_date})

if not ret_df.empty and "total_customers" in ret_df and "returning_customers" in ret_df:
    total_c = int(ret_df.loc[0, "total_customers"])
    returning_c = int(ret_df.loc[0, "returning_customers"])
    retention = (returning_c / total_c) * 100 if total_c > 0 else np.nan
    kpi("Retention Rate", f"{retention:,.1f}%" if not np.isnan(retention) else "n/a",
        help_text="Returning customers / total customers in period")
else:
    st.info("Cannot compute retention — need bookings(customer_id, created_at/updated_at).")

# =========================
# ROUTE PROFITABILITY HEATMAP
# =========================
st.subheader("🗺 Route Profitability Heatmap")

# Try to compute revenue per route via bookings + booking_transactions (amount) or payments linked to bookings
rev_by_route = pd.DataFrame()

# Strategy A: booking_transactions has (booking_id, amount); bookings has (route_id); routes has (name)
if has_cols(engine, "booking_transactions", ["booking_id"]) and has_cols(engine, "bookings", ["id","route_id"]):
    # Try to detect amount column in booking_transactions
    bt_cols = [c.lower() for c in tbl_cols["booking_transactions"]]
    amt_col = "amount" if "amount" in bt_cols else None
    if amt_col:
        q = f"""
        SELECT r.name AS route_name, DATE(b.created_at) AS d, SUM(bt.{amt_col}) AS revenue
        FROM booking_transactions bt
        JOIN bookings b ON b.id = bt.booking_id
        JOIN routes r ON r.id = b.route_id
        WHERE b.created_at::date BETWEEN :sd AND :ed
        GROUP BY 1,2
        ORDER BY 1,2;
        """
        rev_by_route = safe_read_sql(engine, q, {"sd": start_date, "ed": end_date})

# Strategy B: payments has booking_id; if so, use that
if rev_by_route.empty and has_cols(engine, "payments", ["booking_id"]) and has_cols(engine, "bookings", ["id","route_id"]):
    q = """
    SELECT r.name AS route_name, DATE(p.created_at) AS d, SUM(p.amount) AS revenue
    FROM payments p
    JOIN bookings b ON b.id = p.booking_id
    JOIN routes r ON r.id = b.route_id
    WHERE p.created_at::date BETWEEN :sd AND :ed
      AND p.direction = 'in'
    GROUP BY 1,2 ORDER BY 1,2;
    """
    rev_by_route = safe_read_sql(engine, q, {"sd": start_date, "ed": end_date})

# Costs by route (proxy): if bookings has route_id and trip_cost_reports has route_id or cost_center_id mapping
cost_by_route = pd.DataFrame()
if has_cols(engine, "bookings", ["route_id"]) and has_cols(engine, "routes", ["id","name"]):
    # Use per-booking route-level per-row operational fields from routes if exist, else trip_cost_reports by cost_center
    # 1) From routes table fields (fuel_average * fuel_price + meal_price*meals + primes) as static per-route proxy
    route_cost_q = """
    SELECT
      r.name AS route_name,
      COALESCE(r.fuel_average,0) * COALESCE(r.fuel_price, COALESCE(r.fuel_unit_price,0)) +
      COALESCE(r.meal_price,0) * COALESCE(r.meals,0) +
      COALESCE(r.prime_driver,0) + COALESCE(r.prime_codriver,0) +
      COALESCE(r.prime_accomodation_driver_codriver,0) + COALESCE(r.prime_accomodation_codriver,0) AS base_cost
    FROM routes r
    """
    rc = safe_read_sql(engine, route_cost_q, {})
    # Multiply base_cost by number of bookings per route in period as a crude proxy
    book_cnt_q = """
    SELECT r.name AS route_name, COUNT(*) AS bookings_cnt
    FROM bookings b
    JOIN routes r ON r.id = b.route_id
    WHERE COALESCE(b.created_at, b.updated_at)::date BETWEEN :sd AND :ed
    GROUP BY 1
    """
    bc = safe_read_sql(engine, book_cnt_q, {"sd": start_date, "ed": end_date})
    if not rc.empty and not bc.empty:
        cost_by_route = pd.merge(rc, bc, on="route_name", how="right")
        cost_by_route["cost"] = cost_by_route["base_cost"] * cost_by_route["bookings_cnt"]
        cost_by_route = cost_by_route[["route_name","cost"]]

# Build heatmap-friendly pivot: routes x month with net profit (revenue - cost)
heatmap_pivot = pd.DataFrame()
if not rev_by_route.empty:
    # Aggregate monthly
    rev_by_route["month"] = pd.to_datetime(rev_by_route["d"]).dt.to_period("M").astype(str)
    rev_m = rev_by_route.groupby(["route_name","month"], as_index=False)["revenue"].sum()
    if not cost_by_route.empty:
        # Use same monthly cost across each month for simplicity (or divide equally)
        # Join by route name; if multiple months, cost is same for each month.
        heatmap_df = pd.merge(rev_m, cost_by_route, on="route_name", how="left")
        heatmap_df["net"] = heatmap_df["revenue"] - heatmap_df["cost"].fillna(0)
    else:
        heatmap_df = rev_m.rename(columns={"revenue":"net"})
    heatmap_pivot = heatmap_df.pivot(index="route_name", columns="month", values="net").fillna(0)

if not heatmap_pivot.empty:
    heatmap_from_pivot(heatmap_pivot, "Route Profitability Heatmap (Net by Month)")
    st.dataframe(heatmap_pivot)
else:
    st.info("Could not compute Route Profitability Heatmap — need revenue by route (bookings+payments or booking_transactions) and/or route costs.")

# =========================
# MONTHLY FINANCIAL TRENDLINE (Bookings vs Revenue vs Cost)
# =========================
st.subheader("📈 Monthly Financial Trendline")

# Bookings count by month
bookings_by_month = pd.DataFrame()
if "bookings" in tbl_cols and any(c in [x.lower() for x in tbl_cols["bookings"]] for c in ["created_at","updated_at"]):
    ts_col = "created_at" if "created_at" in [x.lower() for x in tbl_cols["bookings"]] else "updated_at"
    q = f"""
    SELECT TO_CHAR(DATE_TRUNC('month', COALESCE({ts_col},{ts_col})), 'YYYY-MM-01') AS month_start,
           COUNT(*) AS bookings_cnt
    FROM bookings
    WHERE COALESCE({ts_col},{ts_col})::date BETWEEN :sd AND :ed
    GROUP BY 1 ORDER BY 1;
    """
    bookings_by_month = safe_read_sql(engine, q, {"sd": start_date, "ed": end_date})

rev_month = pd.DataFrame()
if not df_pay.empty:
    tmp = df_pay.copy()
    tmp["month_start"] = pd.to_datetime(tmp["d"]).dt.to_period("M").dt.to_timestamp()
    rev_month = tmp.groupby("month_start", as_index=False)["revenue_in"].sum()

cost_month = pd.DataFrame()
if not df_costs.empty:
    tmpc = df_costs.copy()
    tmpc["month_start"] = pd.to_datetime(tmpc["d"]).dt.to_period("M").dt.to_timestamp()
    cost_month = tmpc.groupby("month_start", as_index=False)["operating_cost"].sum()

# Display lines one by one
if not bookings_by_month.empty:
    bookings_by_month["month_start"] = pd.to_datetime(bookings_by_month["month_start"])
    line_chart_df(bookings_by_month, "month_start", "bookings_cnt", "Bookings per Month")
else:
    st.info("No bookings in selected range (or missing timestamps).")

if not rev_month.empty:
    line_chart_df(rev_month, "month_start", "revenue_in", "Revenue per Month")
if not cost_month.empty:
    line_chart_df(cost_month, "month_start", "operating_cost", "Operating Cost per Month")

# =========================
# QUICK ACTIONS
# =========================
st.subheader("⚡ Quick Actions / Shortcuts")

# Drilldown links are simulated as filtered tables below; downloads provided
with st.expander("🔍 View agency performance drilldown"):
    st.dataframe(df_ag)

if not df_pay.empty:
    csv_rev = df_pay.to_csv(index=False).encode("utf-8")
    st.download_button("📊 Download revenue report (CSV)", data=csv_rev, file_name="revenue_daily.csv", mime="text/csv")

st.caption("🗺 Explore route analytics & ✏ Edit pricing would require write access and forms — out of scope for read-only BI, but can be added if needed.")
''')

with open('/mnt/data/director_dashboard_app_extended.py', 'w', encoding='utf-8') as f:
    f.write(extended_code)

'/mnt/data/director_dashboard_app_extended.py'
