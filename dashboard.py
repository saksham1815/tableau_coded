"""
Security Analytics Dashboard
=============================
A Tableau-style interactive analytics dashboard for CSV-based web traffic /
security logs, built with Streamlit + Plotly.

Run with:
    streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ======================================================================
# PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="Security Analytics Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# COLOR CONSTANTS
# ======================================================================
COLOR_BLOCKED = "#E53935"     # red
COLOR_ALLOWED = "#43A047"     # green
COLOR_CHALLENGE = "#FB8C00"   # orange

COLOR_RISK_HIGH = "#E53935"   # red
COLOR_RISK_MEDIUM = "#FDD835" # yellow
COLOR_RISK_LOW = "#43A047"    # green

COLOR_SLOW = "#E53935"        # red
COLOR_NORMAL = "#43A047"      # green

ACTION_COLOR_MAP = {
    "block": COLOR_BLOCKED, "blocked": COLOR_BLOCKED, "deny": COLOR_BLOCKED,
    "allow": COLOR_ALLOWED, "allowed": COLOR_ALLOWED, "pass": COLOR_ALLOWED,
    "challenge": COLOR_CHALLENGE, "captcha": COLOR_CHALLENGE,
}

CHART_TEMPLATE = "plotly_white"

# Columns expected in the dataset (used for graceful-degradation checks)
EXPECTED_COLUMNS = [
    "ip", "tls_fingerprint", "client_host", "client_host_state", "access_time",
    "request_type", "request_method", "request_path", "request_path_decoded",
    "request_query", "request_length", "client_platform", "headers_user_agent",
    "headers_host", "headers_referer", "ua_name", "ua_category", "ua_os",
    "ua_browser_type", "geo_country_code", "geo_org", "geo_asn", "api_key_id",
    "tenant_id", "account_name", "domain_name", "site_name", "policy_name",
    "selector", "category", "request_id", "action", "monitor_action",
    "deciding_condition_names", "deciding_tags", "triggered_condition_names",
    "triggered_tags", "requests_per_minute", "requests_per_session",
    "session_length_seconds", "requests_with_expired_token",
    "requests_with_no_token", "captcha_provider", "captcha_solve_duration_seconds",
    "aws_region", "tcp_rtt_ms", "tls_rtt_ms", "cwaf_tags", "account_id", "ds",
]

NUMERIC_HINT_COLUMNS = [
    "request_length", "requests_per_minute", "requests_per_session",
    "session_length_seconds", "requests_with_expired_token",
    "requests_with_no_token", "captcha_solve_duration_seconds",
    "tcp_rtt_ms", "tls_rtt_ms",
]


# ======================================================================
# HELPER UTILITIES
# ======================================================================
def col_exists(df: pd.DataFrame, col: str) -> bool:
    """Check whether a column exists and is not entirely null."""
    return col in df.columns and df[col].notna().any()


def safe_unique(df: pd.DataFrame, col: str):
    """Return sorted unique non-null values for a column, or [] if absent."""
    if not col_exists(df, col):
        return []
    try:
        vals = df[col].dropna().unique().tolist()
        vals = [v for v in vals if str(v).strip() != ""]
        return sorted(vals, key=lambda x: str(x))
    except Exception:
        return []


def fmt_int(n) -> str:
    try:
        return f"{int(n):,}"
    except Exception:
        return "N/A"


def fmt_float(n, decimals=2) -> str:
    try:
        return f"{float(n):,.{decimals}f}"
    except Exception:
        return "N/A"


# ======================================================================
# DATA PIPELINE
# ======================================================================
@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    """Load CSV file into a DataFrame. Cached for performance."""
    try:
        df = pd.read_csv(uploaded_file, low_memory=False)
        return df
    except Exception:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, low_memory=False, encoding="latin-1")
            return df
        except Exception as e:
            st.error(f"Failed to read CSV file: {e}")
            return pd.DataFrame()


@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data: dedupe, handle nulls, coerce numeric/date types."""
    if df.empty:
        return df

    df = df.copy()

    # Remove fully duplicate rows
    df = df.drop_duplicates()

    # Strip whitespace from string/object columns
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        try:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].replace({"nan": np.nan, "None": np.nan, "": np.nan})
        except Exception:
            pass

    # Coerce known numeric columns
    for c in NUMERIC_HINT_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Convert access_time to datetime
    if "access_time" in df.columns:
        df["access_time"] = pd.to_datetime(df["access_time"], errors="coerce", utc=True)
    elif "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce", utc=True)

    # Normalize action / monitor_action text casing for consistent grouping
    for c in ["action", "monitor_action"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df


@st.cache_data(show_spinner=False)
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive year/month/day/hour/weekday columns from access_time."""
    if df.empty:
        return df

    df = df.copy()
    time_col = "access_time" if "access_time" in df.columns else (
        "ds" if "ds" in df.columns else None
    )

    if time_col is not None and pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df["year"] = df[time_col].dt.year
        df["month"] = df[time_col].dt.month
        df["day"] = df[time_col].dt.day
        df["hour"] = df[time_col].dt.hour
        df["weekday"] = df[time_col].dt.day_name()
        df["date"] = df[time_col].dt.date
    else:
        df["year"] = np.nan
        df["month"] = np.nan
        df["day"] = np.nan
        df["hour"] = np.nan
        df["weekday"] = np.nan
        df["date"] = np.nan

    return df


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply sidebar multi-select filters dynamically based on available columns."""
    if df.empty:
        return df

    filtered = df.copy()

    for col, selected_vals in filters.items():
        if selected_vals and col in filtered.columns:
            filtered = filtered[filtered[col].isin(selected_vals)]

    return filtered


def aggregate_data(df: pd.DataFrame, group_column: str, metric_column: str, operation: str) -> pd.DataFrame:
    """
    Reusable aggregation engine.

    operation: one of COUNT, DISTINCT COUNT, SUM, AVERAGE, MEDIAN, MIN, MAX
    """
    if df.empty or group_column not in df.columns:
        return pd.DataFrame()

    work = df.copy()

    try:
        if operation == "COUNT":
            result = work.groupby(group_column).size().reset_index(name="value")
        elif operation == "DISTINCT COUNT":
            if metric_column not in work.columns:
                return pd.DataFrame()
            result = work.groupby(group_column)[metric_column].nunique().reset_index(name="value")
        else:
            if metric_column not in work.columns:
                return pd.DataFrame()
            work[metric_column] = pd.to_numeric(work[metric_column], errors="coerce")
            agg_map = {
                "SUM": "sum",
                "AVERAGE": "mean",
                "MEDIAN": "median",
                "MIN": "min",
                "MAX": "max",
            }
            func = agg_map.get(operation, "sum")
            result = work.groupby(group_column)[metric_column].agg(func).reset_index(name="value")

        result = result.sort_values("value", ascending=False)
        return result
    except Exception as e:
        st.warning(f"Aggregation failed: {e}")
        return pd.DataFrame()


def calculate_security_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a per-IP risk score:
        risk_score = blocked_requests + expired_token_requests
                     + captcha_events + high_rate_requests
    Returns a dataframe sorted by descending risk score.
    """
    if df.empty or "ip" not in df.columns:
        return pd.DataFrame()

    work = df.copy()

    grouped = work.groupby("ip").agg(
        requests=("ip", "size")
    ).reset_index()

    # Blocked requests per IP
    if "action" in work.columns:
        blocked = (
            work[work["action"].astype(str).str.lower().isin(["block", "blocked", "deny"])]
            .groupby("ip").size().reset_index(name="blocked_requests")
        )
        grouped = grouped.merge(blocked, on="ip", how="left")
    else:
        grouped["blocked_requests"] = 0

    # Expired token requests per IP
    if "requests_with_expired_token" in work.columns:
        token = work.groupby("ip")["requests_with_expired_token"].sum().reset_index(
            name="expired_token_requests"
        )
        grouped = grouped.merge(token, on="ip", how="left")
    else:
        grouped["expired_token_requests"] = 0

    # Captcha events per IP
    if "captcha_provider" in work.columns:
        captcha = (
            work[work["captcha_provider"].notna()]
            .groupby("ip").size().reset_index(name="captcha_events")
        )
        grouped = grouped.merge(captcha, on="ip", how="left")
    else:
        grouped["captcha_events"] = 0

    # High request-rate events per IP (requests_per_minute above 75th percentile)
    if "requests_per_minute" in work.columns and work["requests_per_minute"].notna().any():
        threshold = work["requests_per_minute"].quantile(0.75)
        high_rate = (
            work[work["requests_per_minute"] > threshold]
            .groupby("ip").size().reset_index(name="high_rate_requests")
        )
        grouped = grouped.merge(high_rate, on="ip", how="left")
    else:
        grouped["high_rate_requests"] = 0

    grouped = grouped.fillna(0)
    grouped["risk_score"] = (
        grouped["blocked_requests"]
        + grouped["expired_token_requests"]
        + grouped["captcha_events"]
        + grouped["high_rate_requests"]
    )

    # Attach country / action / category (most frequent value per IP)
    def most_common(series):
        try:
            return series.dropna().mode().iloc[0]
        except Exception:
            return np.nan

    extras = {}
    if "geo_country_code" in work.columns:
        extras["country"] = work.groupby("ip")["geo_country_code"].agg(most_common)
    if "action" in work.columns:
        extras["action"] = work.groupby("ip")["action"].agg(most_common)
    if "category" in work.columns:
        extras["category"] = work.groupby("ip")["category"].agg(most_common)

    for name, series in extras.items():
        grouped = grouped.merge(series.reset_index(name=name), on="ip", how="left")

    grouped = grouped.sort_values("risk_score", ascending=False)
    return grouped


def risk_bucket(score: float, max_score: float) -> str:
    """Bucket a risk score into High / Medium / Low."""
    if max_score <= 0:
        return "Low"
    ratio = score / max_score
    if ratio >= 0.66:
        return "High"
    elif ratio >= 0.33:
        return "Medium"
    return "Low"


RISK_COLOR_MAP = {"High": COLOR_RISK_HIGH, "Medium": COLOR_RISK_MEDIUM, "Low": COLOR_RISK_LOW}


# ======================================================================
# KPI HELPERS
# ======================================================================
def render_kpi(col, label, value, help_text=""):
    with col:
        st.metric(label=label, value=value, help=help_text)


def build_kpi_section(df: pd.DataFrame):
    st.subheader("📊 Key Performance Indicators")

    total_requests = len(df)
    unique_ips = df["ip"].nunique() if "ip" in df.columns else None
    unique_domains = df["domain_name"].nunique() if "domain_name" in df.columns else None
    unique_countries = df["geo_country_code"].nunique() if "geo_country_code" in df.columns else None
    avg_req_size = df["request_length"].mean() if "request_length" in df.columns else None
    max_req_size = df["request_length"].max() if "request_length" in df.columns else None
    avg_rpm = df["requests_per_minute"].mean() if "requests_per_minute" in df.columns else None

    blocked = allowed = challenge = None
    if "action" in df.columns:
        actions_lower = df["action"].astype(str).str.lower()
        blocked = actions_lower.isin(["block", "blocked", "deny"]).sum()
        allowed = actions_lower.isin(["allow", "allowed", "pass"]).sum()
        challenge = actions_lower.isin(["challenge", "captcha"]).sum()

    avg_tcp_rtt = df["tcp_rtt_ms"].mean() if "tcp_rtt_ms" in df.columns else None
    avg_tls_rtt = df["tls_rtt_ms"].mean() if "tls_rtt_ms" in df.columns else None

    row1 = st.columns(4)
    render_kpi(row1[0], "Total Requests", fmt_int(total_requests), "Total number of log rows after filtering")
    render_kpi(row1[1], "Unique IPs", fmt_int(unique_ips) if unique_ips is not None else "N/A", "Distinct source IP addresses")
    render_kpi(row1[2], "Unique Domains", fmt_int(unique_domains) if unique_domains is not None else "N/A", "Distinct domains observed")
    render_kpi(row1[3], "Unique Countries", fmt_int(unique_countries) if unique_countries is not None else "N/A", "Distinct geo countries")

    row2 = st.columns(4)
    render_kpi(row2[0], "Avg Request Size", fmt_float(avg_req_size) if avg_req_size is not None else "N/A", "Average request_length")
    render_kpi(row2[1], "Max Request Size", fmt_int(max_req_size) if max_req_size is not None else "N/A", "Largest single request")
    render_kpi(row2[2], "Avg Requests/Min", fmt_float(avg_rpm) if avg_rpm is not None else "N/A", "Average requests_per_minute")
    render_kpi(row2[3], "Avg TCP RTT (ms)", fmt_float(avg_tcp_rtt) if avg_tcp_rtt is not None else "N/A", "Average TCP round-trip time")

    row3 = st.columns(4)
    render_kpi(row3[0], "Blocked Requests", fmt_int(blocked) if blocked is not None else "N/A", "Requests with a blocking action")
    render_kpi(row3[1], "Allowed Requests", fmt_int(allowed) if allowed is not None else "N/A", "Requests that were allowed")
    render_kpi(row3[2], "Challenge Requests", fmt_int(challenge) if challenge is not None else "N/A", "Requests served a challenge/captcha")
    render_kpi(row3[3], "Avg TLS RTT (ms)", fmt_float(avg_tls_rtt) if avg_tls_rtt is not None else "N/A", "Average TLS round-trip time")


# ======================================================================
# CHART BUILDERS
# ======================================================================
def chart_traffic_overview(df: pd.DataFrame):
    st.subheader("📈 Traffic Overview")

    if "access_time" not in df.columns or df["access_time"].isna().all():
        st.info("No valid 'access_time' data available for traffic-over-time charts.")
        return

    c1, c2 = st.columns(2)

    with c1:
        ts = df.set_index("access_time").resample("1h").size().reset_index(name="requests")
        fig = px.line(
            ts, x="access_time", y="requests",
            title="Request Volume Over Time (Hourly)",
            template=CHART_TEMPLATE,
        )
        fig.update_traces(hovertemplate="%{x}<br>Requests: %{y}")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "hour" in df.columns and "weekday" in df.columns:
            heat = df.dropna(subset=["hour", "weekday"]).groupby(["weekday", "hour"]).size().reset_index(name="requests")
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            heat["weekday"] = pd.Categorical(heat["weekday"], categories=weekday_order, ordered=True)
            pivot = heat.pivot(index="weekday", columns="hour", values="requests").fillna(0)
            fig = px.imshow(
                pivot, aspect="auto", color_continuous_scale="Reds",
                title="Hourly Traffic Heatmap", labels=dict(color="Requests"),
                template=CHART_TEMPLATE,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Hour/weekday data unavailable for heatmap.")

    if "date" in df.columns:
        daily = df.dropna(subset=["date"]).groupby("date").size().reset_index(name="requests")
        if not daily.empty:
            fig = px.bar(
                daily, x="date", y="requests",
                title="Daily Request Trend", template=CHART_TEMPLATE,
            )
            st.plotly_chart(fig, use_container_width=True)


def chart_geography(df: pd.DataFrame):
    st.subheader("🌍 Geography")
    c1, c2 = st.columns(2)

    with c1:
        if col_exists(df, "geo_country_code"):
            top_countries = df["geo_country_code"].value_counts().reset_index()
            top_countries.columns = ["country", "requests"]
            fig = px.choropleth(
                top_countries, locations="country", locationmode="ISO-3",
                color="requests", color_continuous_scale="Reds",
                title="Country Traffic (Top requests by country)",
                template=CHART_TEMPLATE,
            )
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.bar(
                top_countries.head(15), x="country", y="requests",
                title="Top 15 Countries by Requests", template=CHART_TEMPLATE,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Column 'geo_country_code' not available.")

    with c2:
        if col_exists(df, "geo_asn"):
            asn_counts = df["geo_asn"].value_counts().reset_index().head(15)
            asn_counts.columns = ["asn", "requests"]
            fig = px.bar(
                asn_counts, x="asn", y="requests", orientation="v",
                title="ASN Distribution (Top 15)", template=CHART_TEMPLATE,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'geo_asn' not available.")


def chart_traffic_analysis(df: pd.DataFrame):
    st.subheader("🔍 Traffic Analysis")
    c1, c2 = st.columns(2)

    with c1:
        if col_exists(df, "request_method"):
            counts = df["request_method"].value_counts().reset_index()
            counts.columns = ["method", "requests"]
            fig = px.pie(
                counts, names="method", values="requests",
                title="HTTP Method Distribution", template=CHART_TEMPLATE,
                hole=0.4,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'request_method' not available.")

    with c2:
        if col_exists(df, "request_type"):
            counts = df["request_type"].value_counts().reset_index()
            counts.columns = ["type", "requests"]
            fig = px.pie(
                counts, names="type", values="requests",
                title="Request Type Distribution", template=CHART_TEMPLATE,
                hole=0.4,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'request_type' not available.")

    c3, c4 = st.columns(2)
    with c3:
        if col_exists(df, "request_path"):
            counts = df["request_path"].value_counts().reset_index().head(15)
            counts.columns = ["path", "requests"]
            fig = px.bar(
                counts, x="requests", y="path", orientation="h",
                title="Top 15 URLs", template=CHART_TEMPLATE,
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'request_path' not available.")

    with c4:
        if col_exists(df, "domain_name"):
            counts = df["domain_name"].value_counts().reset_index().head(15)
            counts.columns = ["domain", "requests"]
            fig = px.bar(
                counts, x="requests", y="domain", orientation="h",
                title="Top 15 Domains", template=CHART_TEMPLATE,
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'domain_name' not available.")


def chart_security_analytics(df: pd.DataFrame):
    st.subheader("🛡️ Security Analytics")
    c1, c2 = st.columns(2)

    with c1:
        if col_exists(df, "action"):
            counts = df["action"].value_counts().reset_index()
            counts.columns = ["action", "requests"]
            colors = [ACTION_COLOR_MAP.get(str(a).lower(), "#9E9E9E") for a in counts["action"]]
            fig = go.Figure(go.Bar(
                x=counts["action"], y=counts["requests"], marker_color=colors,
                hovertemplate="Action: %{x}<br>Requests: %{y}<extra></extra>",
            ))
            fig.update_layout(title="Allowed vs Blocked Traffic", template=CHART_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'action' not available.")

    with c2:
        if col_exists(df, "action") and "access_time" in df.columns and df["access_time"].notna().any():
            trend = df.dropna(subset=["access_time"]).set_index("access_time").groupby(
                [pd.Grouper(freq="1D"), "action"]
            ).size().reset_index(name="requests")
            fig = px.line(
                trend, x="access_time", y="requests", color="action",
                color_discrete_map=ACTION_COLOR_MAP,
                title="Security Action Trend (Daily)", template=CHART_TEMPLATE,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for security action trend.")

    c3, c4 = st.columns(2)
    with c3:
        if col_exists(df, "triggered_condition_names"):
            counts = df["triggered_condition_names"].value_counts().reset_index().head(15)
            counts.columns = ["rule", "count"]
            fig = px.bar(
                counts, x="count", y="rule", orientation="h",
                title="Top Triggered Rules", template=CHART_TEMPLATE,
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'triggered_condition_names' not available.")

    with c4:
        if col_exists(df, "category"):
            counts = df["category"].value_counts().reset_index().head(15)
            counts.columns = ["category", "count"]
            fig = px.bar(
                counts, x="count", y="category", orientation="h",
                title="Top Categories", template=CHART_TEMPLATE,
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'category' not available.")


def chart_performance_analytics(df: pd.DataFrame):
    st.subheader("⚡ Performance Analytics")
    c1, c2 = st.columns(2)

    with c1:
        if col_exists(df, "tcp_rtt_ms"):
            fig = px.histogram(
                df, x="tcp_rtt_ms", nbins=40,
                title="TCP Latency Distribution", template=CHART_TEMPLATE,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'tcp_rtt_ms' not available.")

    with c2:
        if col_exists(df, "tls_rtt_ms"):
            fig = px.histogram(
                df, x="tls_rtt_ms", nbins=40,
                title="TLS Latency Distribution", template=CHART_TEMPLATE,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'tls_rtt_ms' not available.")

    c3, c4 = st.columns(2)
    with c3:
        if col_exists(df, "tcp_rtt_ms"):
            work = df.dropna(subset=["tcp_rtt_ms"]).copy()
            if not work.empty:
                threshold = work["tcp_rtt_ms"].quantile(0.90)
                work["speed_class"] = np.where(work["tcp_rtt_ms"] > threshold, "Slow", "Normal")
                counts = work["speed_class"].value_counts().reset_index()
                counts.columns = ["class", "requests"]
                colors = [COLOR_SLOW if c == "Slow" else COLOR_NORMAL for c in counts["class"]]
                fig = go.Figure(go.Bar(
                    x=counts["class"], y=counts["requests"], marker_color=colors,
                ))
                fig.update_layout(
                    title=f"Slow Request Detection (>{threshold:.1f}ms TCP RTT, 90th pct)",
                    template=CHART_TEMPLATE,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Cannot detect slow requests without 'tcp_rtt_ms'.")

    with c4:
        if col_exists(df, "request_length"):
            fig = px.box(
                df, y="request_length",
                title="Request Size Analysis", template=CHART_TEMPLATE,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'request_length' not available.")


def render_risk_table(df: pd.DataFrame):
    st.subheader("🚨 Risk & Anomaly Analysis")

    risk_df = calculate_security_score(df)
    if risk_df.empty:
        st.info("Unable to calculate risk scores — missing 'ip' column or insufficient data.")
        return

    max_score = risk_df["risk_score"].max()
    risk_df["risk_level"] = risk_df["risk_score"].apply(lambda s: risk_bucket(s, max_score))

    # Anomaly callouts
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("High Traffic IPs", fmt_int((risk_df["requests"] > risk_df["requests"].quantile(0.9)).sum()))
    with c2:
        st.metric("Token Abuse IPs", fmt_int((risk_df["expired_token_requests"] > 0).sum()))
    with c3:
        st.metric("High Risk IPs", fmt_int((risk_df["risk_level"] == "High").sum()))

    display_cols = ["ip"]
    rename_map = {"ip": "IP"}
    if "country" in risk_df.columns:
        display_cols.append("country")
        rename_map["country"] = "Country"
    display_cols.append("requests")
    rename_map["requests"] = "Requests"
    if "action" in risk_df.columns:
        display_cols.append("action")
        rename_map["action"] = "Action"
    display_cols.append("risk_score")
    rename_map["risk_score"] = "Risk Score"
    if "category" in risk_df.columns:
        display_cols.append("category")
        rename_map["category"] = "Category"
    display_cols.append("risk_level")
    rename_map["risk_level"] = "Risk Level"

    table = risk_df[display_cols].rename(columns=rename_map).head(100)

    def highlight_risk(row):
        color = RISK_COLOR_MAP.get(row["Risk Level"], "#FFFFFF")
        return [f"background-color: {color}33"] * len(row)

    try:
        styled = table.style.apply(highlight_risk, axis=1)
        st.dataframe(styled, use_container_width=True, height=400)
    except Exception:
        st.dataframe(table, use_container_width=True, height=400)

    csv_bytes = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Risk Table (CSV)", data=csv_bytes,
        file_name="risk_analysis.csv", mime="text/csv",
    )


# ======================================================================
# LIVE INVESTIGATION / PIVOT BUILDER
# ======================================================================
CHART_TYPES = ["Line", "Bar", "Stacked Bar", "Pie", "Scatter", "Area", "Box", "Heatmap", "Table Only"]

AGG_FUNCS = {
    "COUNT": "count",
    "DISTINCT COUNT": pd.Series.nunique,
    "SUM": "sum",
    "AVERAGE": "mean",
    "MEDIAN": "median",
    "MIN": "min",
    "MAX": "max",
}

INVESTIGATION_DIMENSION_FIELDS = {
    "IP": "ip",
    "Country": "geo_country_code",
    "ASN": "geo_asn",
    "Domain": "domain_name",
    "Site": "site_name",
    "Tenant": "tenant_id",
    "Account": "account_name",
    "Policy": "policy_name",
    "Category": "category",
    "Action": "action",
    "Monitor Action": "monitor_action",
    "Request Type": "request_type",
    "HTTP Method": "request_method",
    "Browser": "ua_browser_type",
    "OS": "ua_os",
    "User Agent": "ua_name",
    "Cloud Region": "aws_region",
    "Captcha Provider": "captcha_provider",
    "TLS Fingerprint": "tls_fingerprint",
    "Hour": "hour",
    "Weekday": "weekday",
    "Date": "date",
    "None": None,
}

INVESTIGATION_METRIC_FIELDS = {
    "Request Count": None,  # row count, used with COUNT
    "Request Size": "request_length",
    "Requests/Min": "requests_per_minute",
    "Requests/Session": "requests_per_session",
    "Session Length (s)": "session_length_seconds",
    "Expired Token Requests": "requests_with_expired_token",
    "No-Token Requests": "requests_with_no_token",
    "Captcha Solve Time (s)": "captcha_solve_duration_seconds",
    "TCP RTT (ms)": "tcp_rtt_ms",
    "TLS RTT (ms)": "tls_rtt_ms",
    "IP (for distinct count)": "ip",
}


def run_pivot_query(df: pd.DataFrame, rows_col, cols_col, metric_col, operation: str) -> pd.DataFrame:
    """
    Flexible pivot engine used by the Live Investigation tab.
    rows_col / cols_col: dimension columns to group by (cols_col optional).
    metric_col: numeric/identifier column to aggregate, or None for plain COUNT.
    operation: key from AGG_FUNCS.
    """
    if df.empty or rows_col is None or rows_col not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    group_cols = [rows_col] + ([cols_col] if cols_col and cols_col in work.columns and cols_col != rows_col else [])

    func = AGG_FUNCS.get(operation, "count")

    try:
        if operation == "COUNT" or metric_col is None:
            result = work.groupby(group_cols).size().reset_index(name="value")
        else:
            if metric_col not in work.columns:
                return pd.DataFrame()
            if operation != "DISTINCT COUNT":
                work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
            result = work.groupby(group_cols)[metric_col].agg(func).reset_index(name="value")
        return result
    except Exception as e:
        st.warning(f"Pivot query failed: {e}")
        return pd.DataFrame()


def render_pivot_chart(result: pd.DataFrame, chart_type: str, rows_col: str, cols_col, metric_label: str, operation: str):
    """Render the selected chart type from a pivoted result dataframe."""
    if result.empty:
        st.info("No data for the current selection.")
        return

    title = f"{operation} of {metric_label} by {rows_col}" + (f" / {cols_col}" if cols_col else "")
    color_arg = cols_col if (cols_col and cols_col in result.columns) else None

    try:
        if chart_type == "Table Only":
            return

        elif chart_type == "Line":
            fig = px.line(result, x=rows_col, y="value", color=color_arg, markers=True,
                          title=title, template=CHART_TEMPLATE)

        elif chart_type == "Bar":
            fig = px.bar(result, x=rows_col, y="value", color=color_arg,
                        title=title, template=CHART_TEMPLATE, barmode="group")

        elif chart_type == "Stacked Bar":
            fig = px.bar(result, x=rows_col, y="value", color=color_arg,
                        title=title, template=CHART_TEMPLATE, barmode="stack")

        elif chart_type == "Pie":
            pie_data = result.groupby(rows_col)["value"].sum().reset_index().sort_values("value", ascending=False).head(15)
            fig = px.pie(pie_data, names=rows_col, values="value", hole=0.4,
                        title=title, template=CHART_TEMPLATE)

        elif chart_type == "Scatter":
            fig = px.scatter(result, x=rows_col, y="value", color=color_arg, size="value",
                             title=title, template=CHART_TEMPLATE)

        elif chart_type == "Area":
            fig = px.area(result, x=rows_col, y="value", color=color_arg,
                          title=title, template=CHART_TEMPLATE)

        elif chart_type == "Box":
            fig = px.box(result, x=rows_col, y="value", color=color_arg,
                        title=title, template=CHART_TEMPLATE)

        elif chart_type == "Heatmap":
            if cols_col and cols_col in result.columns:
                pivot = result.pivot_table(index=rows_col, columns=cols_col, values="value", aggfunc="sum", fill_value=0)
                fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Reds",
                                title=title, template=CHART_TEMPLATE, labels=dict(color=metric_label))
            else:
                st.info("Heatmap needs both a Row and a Column dimension selected.")
                return
        else:
            fig = px.bar(result, x=rows_col, y="value", title=title, template=CHART_TEMPLATE)

        fig.update_traces(hovertemplate=f"{rows_col}: %{{x}}<br>{metric_label}: %{{y}}<extra></extra>" if chart_type != "Pie" else None)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not render {chart_type} chart: {e}")


def render_live_investigation(df: pd.DataFrame):
    """
    Real-time, self-serve investigation tab:
    upload -> pick chart type -> pick row/column dimensions -> pick metric & aggregation
    -> apply local filters -> get instant chart + pivot table.
    Operates on the data already uploaded/filtered by the sidebar.
    """
    st.subheader("🔬 Live Investigation")
    st.caption(
        "Build your own analysis on the fly: choose a chart type, drag fields into Rows/Columns, "
        "pick a metric, and filter — results update instantly, like a pivot table + chart builder."
    )

    if df.empty:
        st.info("No data available. Upload a CSV and adjust filters above.")
        return

    # ---- Step 1: local (in-tab) filters, independent of sidebar, for rapid drilling ----
    with st.expander("🔎 Step 1 — Quick Filters (applied only within this investigation)", expanded=False):
        local_filters = {}
        quick_filter_fields = ["geo_country_code", "action", "category", "domain_name", "ua_browser_type", "policy_name"]
        available_quick = [c for c in quick_filter_fields if col_exists(df, c)]
        if available_quick:
            f_cols = st.columns(min(3, len(available_quick)))
            for i, field in enumerate(available_quick):
                with f_cols[i % len(f_cols)]:
                    opts = safe_unique(df, field)
                    sel = st.multiselect(field, opts, key=f"inv_filter_{field}")
                    if sel:
                        local_filters[field] = sel
        else:
            st.caption("No common filterable columns found in this dataset.")

    work_df = apply_filters(df, local_filters) if local_filters else df

    # ---- Step 2: chart type + dimensions + metric + aggregation ----
    st.markdown("**Step 2 — Configure Analysis**")
    c1, c2, c3, c4, c5 = st.columns([1.1, 1, 1, 1.2, 1])

    with c1:
        chart_type = st.selectbox("Chart Type", CHART_TYPES, key="inv_chart_type")

    available_rows = {k: v for k, v in INVESTIGATION_DIMENSION_FIELDS.items() if v is None or v in work_df.columns}
    available_cols = dict(available_rows)  # same universe, "None" allowed

    with c2:
        rows_label = st.selectbox("Rows (Group By)", [k for k in available_rows if k != "None"], key="inv_rows")
    with c3:
        cols_label = st.selectbox("Columns (Split By)", list(available_cols.keys()), index=list(available_cols.keys()).index("None") if "None" in available_cols else 0, key="inv_cols")
    with c4:
        metric_label = st.selectbox("Metric / Value", list(INVESTIGATION_METRIC_FIELDS.keys()), key="inv_metric")
    with c5:
        operation = st.selectbox("Aggregation", list(AGG_FUNCS.keys()), key="inv_agg")

    rows_col = available_rows.get(rows_label)
    cols_col = available_cols.get(cols_label)
    metric_col = INVESTIGATION_METRIC_FIELDS.get(metric_label)

    if operation == "COUNT":
        metric_col_used = None
    else:
        metric_col_used = metric_col if metric_col else "ip"
        if metric_col_used not in work_df.columns:
            st.warning(f"Selected metric column '{metric_col_used}' isn't present in this dataset — switch metric or aggregation.")
            return

    top_n = st.slider("Limit to Top N rows (by value)", min_value=5, max_value=100, value=20, step=5, key="inv_topn")

    # ---- Step 3: run pivot + render ----
    result = run_pivot_query(work_df, rows_col, cols_col, metric_col_used, operation)

    if result.empty:
        st.info("No results for this combination — try a different field or aggregation.")
        return

    # Trim to top N by total value per row dimension, keep columns split intact
    totals = result.groupby(rows_col)["value"].sum().sort_values(ascending=False).head(top_n).index
    result_trimmed = result[result[rows_col].isin(totals)].sort_values("value", ascending=False)

    st.markdown("---")
    chart_tab, table_tab, summary_tab = st.tabs(["📊 Chart", "📋 Pivot Table", "📝 Summary"])

    with chart_tab:
        render_pivot_chart(result_trimmed, chart_type, rows_col, cols_col if cols_col else None, metric_label, operation)

    with table_tab:
        pivot_display = result_trimmed.copy()
        if cols_col and cols_col in pivot_display.columns:
            try:
                pivot_display = pivot_display.pivot_table(index=rows_col, columns=cols_col, values="value", aggfunc="sum", fill_value=0)
                pivot_display = pivot_display.reset_index()
            except Exception:
                pass
        st.dataframe(pivot_display, use_container_width=True, height=420)
        csv_bytes = pivot_display.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Pivot (CSV)", data=csv_bytes, file_name="live_investigation.csv", mime="text/csv", key="inv_dl")

    with summary_tab:
        st.write(f"**Dimension (Rows):** {rows_label}" + (f"  |  **Split (Columns):** {cols_label}" if cols_col else ""))
        st.write(f"**Metric:** {metric_label}  |  **Aggregation:** {operation}")
        st.write(f"**Total groups shown:** {fmt_int(len(totals))}")
        st.write(f"**Top entry:** {result_trimmed.iloc[0][rows_col]} → {fmt_float(result_trimmed.iloc[0]['value'])}")
        st.write(f"**Sum of values:** {fmt_float(result_trimmed['value'].sum())}")
        st.write(f"**Average value:** {fmt_float(result_trimmed['value'].mean())}")
        if local_filters:
            st.write("**Active quick filters:**")
            for k, v in local_filters.items():
                st.write(f"- {k}: {', '.join(map(str, v))}")


# ======================================================================
# ADVANCED ANALYSIS BUILDER
# ======================================================================
def render_advanced_analysis(df: pd.DataFrame):
    st.subheader("🧮 Advanced Analytics Builder")

    group_options = {
        "Country": "geo_country_code",
        "IP": "ip",
        "Domain": "domain_name",
        "Policy": "policy_name",
        "Category": "category",
        "User Agent": "headers_user_agent",
    }
    metric_options = {
        "Requests": None,  # COUNT doesn't need a metric column
        "Size": "request_length",
        "Latency (TCP)": "tcp_rtt_ms",
        "Latency (TLS)": "tls_rtt_ms",
    }
    agg_options = ["COUNT", "DISTINCT COUNT", "SUM", "AVERAGE", "MEDIAN", "MIN", "MAX"]

    available_group_options = {k: v for k, v in group_options.items() if v in df.columns}
    if not available_group_options:
        st.info("No groupable columns available in this dataset.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        group_label = st.selectbox("Group By", list(available_group_options.keys()), key="adv_group")
    with c2:
        metric_label = st.selectbox("Metric", list(metric_options.keys()), key="adv_metric")
    with c3:
        operation = st.selectbox("Aggregation", agg_options, key="adv_agg")

    group_col = available_group_options[group_label]
    metric_col = metric_options[metric_label]

    if operation == "COUNT":
        metric_col_used = group_col  # placeholder, unused in COUNT branch
    elif operation == "DISTINCT COUNT":
        metric_col_used = metric_col if metric_col else "ip"
    else:
        metric_col_used = metric_col

    if operation not in ("COUNT",) and metric_col_used is None:
        st.warning("Please choose a numeric metric (Size/Latency) for this aggregation type.")
        return

    result = aggregate_data(df, group_col, metric_col_used, operation)

    if result.empty:
        st.info("No data available for the selected combination.")
        return

    result = result.rename(columns={group_col: group_label, "value": f"{operation}({metric_label})"})

    tab_table, tab_chart, tab_summary = st.tabs(["📋 Table", "📊 Chart", "📝 Summary"])

    with tab_table:
        st.dataframe(result.head(200), use_container_width=True, height=400)
        csv_bytes = result.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Table (CSV)", data=csv_bytes, file_name="advanced_analysis.csv", mime="text/csv", key="adv_dl")

    with tab_chart:
        value_col = result.columns[-1]
        fig = px.bar(
            result.head(20), x=group_label, y=value_col,
            title=f"{operation} of {metric_label} by {group_label}",
            template=CHART_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_summary:
        value_col = result.columns[-1]
        st.write(f"**Total groups:** {fmt_int(len(result))}")
        st.write(f"**Top group:** {result.iloc[0][group_label]} ({fmt_float(result.iloc[0][value_col])})")
        st.write(f"**Sum of values:** {fmt_float(result[value_col].sum())}")
        st.write(f"**Average value:** {fmt_float(result[value_col].mean())}")
        st.write(f"**Median value:** {fmt_float(result[value_col].median())}")


# ======================================================================
# RAW DATA EXPLORER
# ======================================================================
def render_raw_data_explorer(df: pd.DataFrame):
    st.subheader("🗂️ Raw Data Explorer")

    all_cols = list(df.columns)
    default_cols = all_cols[:12] if len(all_cols) > 12 else all_cols

    selected_cols = st.multiselect("Select columns to display", all_cols, default=default_cols)
    search_term = st.text_input("Search (across all selected columns)", "")

    view_df = df[selected_cols].copy() if selected_cols else df.copy()

    if search_term:
        mask = pd.Series(False, index=view_df.index)
        for c in view_df.columns:
            try:
                mask = mask | view_df[c].astype(str).str.contains(search_term, case=False, na=False)
            except Exception:
                continue
        view_df = view_df[mask]

    st.dataframe(view_df, use_container_width=True, height=450)
    st.caption(f"Showing {fmt_int(len(view_df))} of {fmt_int(len(df))} rows")

    c1, c2 = st.columns(2)
    with c1:
        csv_bytes = view_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download as CSV", data=csv_bytes, file_name="filtered_data.csv", mime="text/csv")
    with c2:
        try:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                view_df.to_excel(writer, index=False, sheet_name="data")
            st.download_button(
                "⬇️ Download as Excel", data=buffer.getvalue(),
                file_name="filtered_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception:
            st.caption("Excel export unavailable (xlsxwriter not installed).")


# ======================================================================
# SIDEBAR FILTER ENGINE
# ======================================================================
def build_sidebar_filters(df: pd.DataFrame) -> dict:
    st.sidebar.title("🛡️ Filters")

    if st.sidebar.button("🔄 Reset Filters"):
        for key in list(st.session_state.keys()):
            if key.startswith("filter_"):
                del st.session_state[key]
        st.rerun()

    filter_field_map = {
        "Country": "geo_country_code",
        "ASN": "geo_asn",
        "Domain": "domain_name",
        "Site": "site_name",
        "Tenant": "tenant_id",
        "Account": "account_name",
        "Request Type": "request_type",
        "HTTP Method": "request_method",
        "Action": "action",
        "Category": "category",
        "Browser": "ua_browser_type",
        "OS": "ua_os",
        "User Agent": "ua_name",
        "Policy": "policy_name",
        "Cloud Region": "aws_region",
    }

    filters = {}
    for label, col in filter_field_map.items():
        options = safe_unique(df, col)
        if options:
            key = f"filter_{col}"
            selected = st.sidebar.multiselect(label, options, key=key)
            filters[col] = selected

    return filters


# ======================================================================
# MAIN DASHBOARD BUILDER
# ======================================================================
def build_dashboard():
    st.title("🛡️ Security Analytics Dashboard")
    st.caption("Tableau-style interactive analytics for web traffic & security logs")

    uploaded_file = st.file_uploader("Upload a CSV log file", type=["csv"])

    if uploaded_file is None:
        st.info("👆 Upload a CSV file to begin analysis.")
        with st.expander("Expected columns (not all required)"):
            st.code(", ".join(EXPECTED_COLUMNS))
        return

    with st.spinner("Loading data..."):
        raw_df = load_csv(uploaded_file)

    if raw_df.empty:
        st.error("The uploaded file could not be parsed or is empty.")
        return

    try:
        df = clean_data(raw_df)
        df = create_time_features(df)
    except Exception as e:
        st.error(f"Error during data processing: {e}")
        return

    filters = build_sidebar_filters(df)
    filtered_df = apply_filters(df, filters)

    if filtered_df.empty:
        st.warning("No data matches the current filter selection.")
        return

    st.sidebar.markdown("---")
    st.sidebar.metric("Rows after filtering", fmt_int(len(filtered_df)))
    st.sidebar.metric("Total rows loaded", fmt_int(len(df)))

    build_kpi_section(filtered_df)
    st.markdown("---")

    tabs = st.tabs([
        "📈 Traffic Overview",
        "🌍 Geography",
        "🔍 Traffic Analysis",
        "🛡️ Security Analytics",
        "⚡ Performance",
        "🚨 Risk Analysis",
        "🔬 Live Investigation",
        "🧮 Advanced Analysis",
        "🗂️ Raw Data Explorer",
    ])

    with tabs[0]:
        try:
            chart_traffic_overview(filtered_df)
        except Exception as e:
            st.error(f"Error rendering Traffic Overview: {e}")

    with tabs[1]:
        try:
            chart_geography(filtered_df)
        except Exception as e:
            st.error(f"Error rendering Geography: {e}")

    with tabs[2]:
        try:
            chart_traffic_analysis(filtered_df)
        except Exception as e:
            st.error(f"Error rendering Traffic Analysis: {e}")

    with tabs[3]:
        try:
            chart_security_analytics(filtered_df)
        except Exception as e:
            st.error(f"Error rendering Security Analytics: {e}")

    with tabs[4]:
        try:
            chart_performance_analytics(filtered_df)
        except Exception as e:
            st.error(f"Error rendering Performance Analytics: {e}")

    with tabs[5]:
        try:
            render_risk_table(filtered_df)
        except Exception as e:
            st.error(f"Error rendering Risk Analysis: {e}")

    with tabs[6]:
        try:
            render_live_investigation(filtered_df)
        except Exception as e:
            st.error(f"Error rendering Live Investigation: {e}")

    with tabs[7]:
        try:
            render_advanced_analysis(filtered_df)
        except Exception as e:
            st.error(f"Error rendering Advanced Analysis: {e}")

    with tabs[8]:
        try:
            render_raw_data_explorer(filtered_df)
        except Exception as e:
            st.error(f"Error rendering Raw Data Explorer: {e}")


# ======================================================================
# ENTRY POINT
# ======================================================================
if __name__ == "__main__":
    build_dashboard()
