# director_full_dashboard.py
import os
import datetime as dt
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ------------- CONFIG -------------
load_dotenv()
st.set_page_config(page_title="Director Dashboard — Full", layout="wide")
COLOR_PALETTE = ["#89CFF0", "#FFB6C1"]  # baby blue, baby pink

# ------------- DB helpers -------------
@st.cache_resource
def get_engine():
    user = os.getenv("")
    pwd = os.getenv("")
    host = os.getenv("")
    port = os.getenv("", "")
    db = os.getenv("")

    if not all([user, pwd, host, db]):
        raise RuntimeError("Missing DB credentials in environment variables.")
    from urllib.parse import quote_plus
    pwd_q = quote_plus(pwd)
    url = f"postgresql://{user}:{pwd_q}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True)

@st.cache_data(ttl=300)
def run_query(sql_text: str, params: dict | None = None):
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql_text), conn, params=params or {})

# ------------- SQL templates -------------
SUMMARY_SQL = """
SELECT
  COALESCE(SUM(b.total_amount),0) AS gross_revenue,
  COUNT(*) AS total_bookings,
  COALESCE(AVG(b.total_amount),0) AS avg_ticket,
  COALESCE(SUM(ac.amount),0) AS commission_cost,
  COALESCE(SUM(b.operating_cost),0) AS operating_cost,
  COALESCE(SUM(m.cost),0) AS maintenance_cost
FROM bookings b
LEFT JOIN agency_commissions ac ON ac.booking_id = b.id
LEFT JOIN (
    SELECT booking_id, SUM(cost) AS cost FROM maintenance_tasks GROUP BY booking_id
) m ON m.booking_id = b.id
WHERE b.booking_date BETWEEN :start AND :end;
"""

MONTHLY_SQL = """
SELECT
  DATE_TRUNC('month', b.booking_date)::date AS month,
  COUNT(*) AS bookings_count,
  COALESCE(SUM(b.total_amount),0) AS revenue,
  COALESCE(SUM(b.operating_cost),0) AS cost
FROM bookings b
WHERE b.booking_date BETWEEN :start AND :end
GROUP BY month
ORDER BY month;
"""

ROUTE_SQL = """
SELECT r.id AS route_id, r.name AS route_name,
       COALESCE(SUM(b.total_amount),0) AS revenue,
       COALESCE(SUM(b.operating_cost),0) AS cost,
       COALESCE(SUM(b.total_amount - COALESCE(b.operating_cost,0)),0) AS profit
FROM bookings b
LEFT JOIN routes r ON b.route_id = r.id
WHERE b.booking_date BETWEEN :start AND :end
GROUP BY r.id, r.name
ORDER BY profit DESC;
"""

SOURCE_SQL = """
SELECT COALESCE(b.booking_source,'UNKNOWN') AS source, COUNT(*) AS cnt
FROM bookings b
WHERE b.booking_date BETWEEN :start AND :end
GROUP BY source
ORDER BY cnt DESC;
"""

AGENCY_SQL = """
SELECT a.id AS agency_id, a.name AS agency_name,
  COALESCE(SUM(b.total_amount),0) AS revenue,
  COALESCE(SUM(ac.amount),0) AS commission_cost,
  (COALESCE(SUM(b.total_amount),0) - COALESCE(SUM(b.operating_cost),0) - COALESCE(SUM(ac.amount),0)) AS profit
FROM bookings b
LEFT JOIN agencies a ON b.agency_id = a.id
LEFT JOIN agency_commissions ac ON ac.booking_id = b.id
WHERE b.booking_date BETWEEN :start AND :end
GROUP BY a.id, a.name
ORDER BY profit DESC
LIMIT 20;
"""

FLEET_UTIL_SQL = """
SELECT
  vt.name AS fleet_type,
  COALESCE(SUM(vu.in_use_hours),0) AS in_use_hours,
  COALESCE(SUM(v.total_available_hours),0) AS available_hours
FROM vehicle_usages vu
LEFT JOIN vehicles v ON vu.vehicle_id = v.id
LEFT JOIN fleet_types vt ON v.fleet_type_id = vt.id
WHERE vu.usage_date BETWEEN :start AND :end
GROUP BY vt.name;
"""

RASK_SQL = """
SELECT
  COALESCE(SUM(b.total_amount),0) AS revenue,
  COALESCE(SUM(r.distance_km * v.capacity),0) AS available_seat_km
FROM bookings b
LEFT JOIN routes r ON b.route_id = r.id
LEFT JOIN vehicles v ON b.vehicle_id = v.id
WHERE b.booking_date BETWEEN :start AND :end;
"""

ROFA_SQL = """
SELECT
  SUM(rev) AS fleet_revenue,
  SUM(m.cost) AS maintenance_cost,
  SUM(f.asset_value) AS fleet_value
FROM (
  SELECT v.id AS vehicle_id, COALESCE(SUM(b.total_amount),0) AS rev
  FROM bookings b
  LEFT JOIN vehicles v ON b.vehicle_id = v.id
  WHERE b.booking_date BETWEEN :start AND :end
  GROUP BY v.id
) AS vehicle_revenues
LEFT JOIN (
  SELECT vehicle_id, COALESCE(SUM(cost),0) AS cost FROM maintenance_tasks WHERE task_date BETWEEN :start AND :end GROUP BY vehicle_id
) m ON m.vehicle_id = vehicle_revenues.vehicle_id
LEFT JOIN fleet_assets f ON f.vehicle_id = vehicle_revenues.vehicle_id;
"""

RETENTION_SQL = """
WITH period_current AS (
  SELECT DISTINCT customer_id FROM bookings WHERE booking_date BETWEEN :start AND :end
),
period_prev AS (
  SELECT DISTINCT customer_id FROM bookings WHERE booking_date BETWEEN :prev_start AND :prev_end
)
SELECT
  (SELECT COUNT(*) FROM period_current) AS customers_current,
  (SELECT COUNT(*) FROM period_prev) AS customers_prev,
  (SELECT COUNT(*) FROM period_current WHERE customer_id IN (SELECT customer_id FROM period_prev)) AS returning_customers;
"""

# ------------- UI: Sidebar filters -------------
st.sidebar.header("Filters")
today = dt.date.today()
default_start = today - dt.timedelta(days=90)
date_range = st.sidebar.date_input("Date range", value=(default_start, today))
if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date = date_range
    end_date = date_range

# Connect to DB or enable sample mode when DB not available
try:
    engine = get_engine()
    sample_mode = False
except Exception as e:
    st.sidebar.error("DB connection error: " + str(e))
    sample_mode = st.sidebar.checkbox("Use sample data (dev)", value=True)

# Populate small picklists if DB is present
selected_routes = []
selected_agencies = []
booking_source_filter = []

if not sample_mode:
    try:
        rlist = run_query("SELECT id, name FROM routes ORDER BY name LIMIT 200;")
        selected_routes = st.sidebar.multiselect("Routes", options=rlist['id'].tolist(),
                                                 format_func=lambda v: rlist[rlist['id']==v]['name'].values[0])
    except Exception:
        selected_routes = []

    try:
        alist = run_query("SELECT id, name FROM agencies ORDER BY name LIMIT 200;")
        selected_agencies = st.sidebar.multiselect("Agencies", options=alist['id'].tolist(),
                                                   format_func=lambda v: alist[alist['id']==v]['name'].values[0])
    except Exception:
        selected_agencies = []

    try:
        ssrc = run_query("SELECT DISTINCT COALESCE(booking_source,'UNKNOWN') AS source FROM bookings LIMIT 50;")
        booking_source_filter = st.sidebar.multiselect("Booking Sources", options=ssrc['source'].tolist(), default=ssrc['source'].tolist())
    except Exception:
        booking_source_filter = []
else:
    selected_routes = st.sidebar.multiselect("Routes", options=[])
    selected_agencies = st.sidebar.multiselect("Agencies", options=[])
    booking_source_filter = st.sidebar.multiselect("Booking Sources", options=["Web","POS","Mobile","B2B"], default=["Web","POS","Mobile","B2B"])

params = {"start": start_date, "end": end_date, "prev_start": start_date - dt.timedelta(days=90), "prev_end": start_date - dt.timedelta(days=1)}

# ------------- Data loading -------------
if sample_mode:
    # sample data so UI can be tested offline
    months = pd.date_range(start=start_date, end=end_date, freq="MS")
    monthly_df = pd.DataFrame({"month": months, "bookings_count": [50]*len(months),
                               "revenue": [50000]*len(months), "cost":[20000]*len(months)})
    route_df = pd.DataFrame({"route_id":[1,2,3,4], "route_name":["Tunis→Sfax","Sousse→Tunis","Gafsa→Tunis","Gabes→Sfax"],
                             "revenue":[150000,90000,40000,20000], "cost":[75000,40000,20000,8000]})
    route_df["profit"] = route_df["revenue"] - route_df["cost"]
    source_df = pd.DataFrame({"source":["Web","POS","Mobile"], "cnt":[300,150,200]})
    summary = {"gross_revenue": route_df["revenue"].sum(), "total_bookings": 650, "avg_ticket": 75, "commission_cost": 12000, "operating_cost": route_df["cost"].sum(), "maintenance_cost": 5000}
    agency_df = pd.DataFrame({"agency_id":[1,2,3], "agency_name":["Agency A","Agency B","Agency C"], "revenue":[80000,60000,20000], "commission_cost":[8000,4000,2000], "profit":[60000,50000,16000]})
    fleet_util_df = pd.DataFrame({"fleet_type":["Coach","Minibus"], "in_use_hours":[1200,800], "available_hours":[2000,1200]})
    rask_val = 150000 / (1000 * 50)  # fake value
    rofa_df = pd.DataFrame({"fleet_revenue":[150000], "maintenance_cost":[5000], "fleet_value":[500000]})
    retention = {"customers_current":300, "customers_prev":320, "returning_customers":200}
else:
    summary_df = run_query(SUMMARY_SQL, params)
    summary = summary_df.iloc[0].to_dict()
    monthly_df = run_query(MONTHLY_SQL, params)
    route_df = run_query(ROUTE_SQL, params)
    source_df = run_query(SOURCE_SQL, params)
    agency_df = run_query(AGENCY_SQL, params)
    try:
        fleet_util_df = run_query(FLEET_UTIL_SQL, params)
    except Exception:
        fleet_util_df = pd.DataFrame()
    try:
        rask_df = run_query(RASK_SQL, params)
        if not rask_df.empty and 'available_seat_km' in rask_df.columns and rask_df.at[0,'available_seat_km'] and rask_df.at[0,'available_seat_km']>0:
            rask_val = float(rask_df.at[0,'revenue'])/float(rask_df.at[0,'available_seat_km'])
        else:
            rask_val = None
    except Exception:
        rask_val = None
    try:
        rofa_df = run_query(ROFA_SQL, params)
    except Exception:
        rofa_df = pd.DataFrame()
    try:
        ret_df = run_query(RETENTION_SQL, params)
        retention = {"customers_current": int(ret_df.at[0,'customers_current']), "customers_prev": int(ret_df.at[0,'customers_prev']), "returning_customers": int(ret_df.at[0,'returning_customers'])}
    except Exception:
        retention = {"customers_current":0,"customers_prev":0,"returning_customers":0}

# ------------- Client-side filtering for small result sets -------------
if selected_routes and not route_df.empty and "route_id" in route_df.columns:
    route_df = route_df[route_df["route_id"].isin(selected_routes)]

# ------------- Top KPIs -------------
st.title("🚍 Director Dashboard — Full Feature Set")
k1,k2,k3,k4,k5 = st.columns([1.5,1,1,1,1])
k1.metric("💰 Gross Revenue", f"{summary.get('gross_revenue',0):,.0f}")
k2.metric("📉 Net Profit (est)", f"{(summary.get('gross_revenue',0) - summary.get('operating_cost',0) - summary.get('commission_cost',0) - summary.get('maintenance_cost',0)) :,.0f}")
k3.metric("🧾 Commission Cost", f"{summary.get('commission_cost',0):,.0f}")
k4.metric("🎟 Avg Ticket", f"{summary.get('avg_ticket',0):.2f}")
k5.metric("📊 Retention", f"{(retention['returning_customers']/retention['customers_prev']*100) if retention['customers_prev']>0 else 0:.1f}%")

if rask_val:
    st.subheader(f"RASK: {rask_val:.6f} (revenue per available seat-km)")
else:
    st.subheader("RASK: N/A (requires routes.distance_km and vehicles.capacity)")

# ------------- Visuals -------------
left, mid, right = st.columns([2,2,1])

with left:
    st.subheader("Monthly Financial Trend (Bookings / Revenue / Cost)")
    if monthly_df.empty:
        st.info("No monthly data.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly_df['month'], y=monthly_df['bookings_count'], name='Bookings', yaxis='y2', marker_color=COLOR_PALETTE[0]))
        fig.add_trace(go.Scatter(x=monthly_df['month'], y=monthly_df['revenue'], name='Revenue', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=monthly_df['month'], y=monthly_df['cost'], name='Cost', mode='lines+markers'))
        fig.update_layout(xaxis_title='Month', yaxis_title='Revenue/Cost', legend_title_text='Metric',
                          yaxis2=dict(overlaying='y', side='right', title='Bookings'))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Route Profitability Heatmap")
    if route_df.empty:
        st.info("No route data.")
    else:
        route_df_sorted = route_df.sort_values("profit", ascending=False)
        fig2 = px.bar(route_df_sorted, x='route_name', y='profit', color='profit', title='Profit by Route', color_continuous_scale=px.colors.sequential.ice)
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

with mid:
    st.subheader("Top Agencies Leaderboard")
    if agency_df.empty:
        st.info("No agency data.")
    else:
        agency_df['rank'] = agency_df['profit'].rank(method='first', ascending=False).astype(int)
        top5 = agency_df.sort_values('profit', ascending=False).head(10)
        st.dataframe(top5[['rank','agency_name','revenue','commission_cost','profit']].rename(columns={'agency_name':'Agency'}))
        st.download_button("Download agencies CSV", top5.to_csv(index=False), file_name="top_agencies.csv")

    st.subheader("Fleet Utilization & Efficiency")
    if fleet_util_df.empty:
        st.info("No fleet usage data available in schema.")
    else:
        fleet_util_df['utilization_rate'] = fleet_util_df['in_use_hours'] / fleet_util_df['available_hours']
        st.dataframe(fleet_util_df[['fleet_type','in_use_hours','available_hours','utilization_rate']])

with right:
    st.subheader("Booking Sources")
    if source_df.empty:
        st.info("No source data.")
    else:
        fig3 = px.pie(source_df, names='source', values='cnt', title='Booking Sources', color_discrete_sequence=COLOR_PALETTE)
        fig3.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Quick Actions")
    col1,col2 = st.columns(2)
    if col1.button("🔍 Agency performance drilldown"):
        st.experimental_set_query_params(action="agency_drilldown")
    if col2.download_button("📊 Download revenue report", monthly_df.to_csv(index=False), file_name="monthly_revenue.csv"):
        pass
    if st.button("🗺 Explore route analytics (open)"):
        st.info("This would open route analytics page (link to another app). Placeholder.")

# ------------- Drilldown bookings -------------
st.markdown("---")
st.subheader("Drilldown: Bookings (latest 200 rows by date)")

if sample_mode:
    bookings_df = pd.DataFrame({
        "booking_id":[1001,1002,1003],
        "booking_date":[start_date, start_date, end_date],
        "route_name":["Tunis→Sfax","Sousse→Tunis","Tunis→Sfax"],
        "customer_id":[11,12,13],
        "seats":[2,1,3],
        "total_amount":[45.0,30.0,55.0],
        "booking_source":["Web","POS","Mobile"]
    })
else:
    q = """
    SELECT b.id AS booking_id, b.booking_date, r.name AS route_name, b.customer_id,
           b.seats_reserved AS seats, b.total_amount, b.booking_source, b.operating_cost
    FROM bookings b
    LEFT JOIN routes r ON b.route_id = r.id
    WHERE b.booking_date BETWEEN :start AND :end
    ORDER BY b.booking_date DESC
    LIMIT 200;
    """
    bookings_df = run_query(q, params)

st.dataframe(bookings_df)
st.download_button("Export bookings CSV", bookings_df.to_csv(index=False), file_name="bookings_drilldown.csv")

st.markdown("""
**Notes & next steps**
- Column/table names used in queries assume common names (`bookings.total_amount`, `routes.distance_km`, `vehicles.capacity`, `agency_commissions.amount`). Adjust SQL if your schema uses different names.
- For heavy workloads, create materialized views in DB and query those.
- Fleet lifecycle timeline requires `fleet_assets` with purchase_date and maintenance logs; we included ROFA query as example if such tables exist.
- Security: run behind VPN or reverse proxy; never commit `.env`.
""")
