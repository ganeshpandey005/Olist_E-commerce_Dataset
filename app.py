# app.py
# Olist Streamlit App - Login + Dashboard + Prediction section
# CSV datasets must be in the clean_dataset folder (or beside this file).
# The hierarchical prediction PKL is downloaded automatically from Google Drive.
# Run: C:\Users\HP\anaconda3\python.exe -m streamlit run "Model Prediction & Python files/app.py"

from pathlib import Path
import hashlib
import hmac

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go



import joblib
from pathlib import Path


# ----------------------------- APP CONFIG -----------------------------
st.set_page_config(
    page_title="Olist Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
# Google Drive folder containing the Olist CSV files.
DATASET_DRIVE_FOLDER_ID = "14YgQja5BpHDakudYrkMqqE9DdMAhUste"
DATA_DIR = ROOT / "clean_dataset"

# Demo login. Change these two values if you want your own login credentials.
APP_USERNAME = "ganesh pandey"
APP_PASSWORD = "GP@123"


# ----------------------------- STYLING -----------------------------
st.markdown("""
<style>
    .block-container {padding-top: 2.8rem !important; padding-bottom: 2rem; max-width: 1500px;}
    .main-title {font-size: 30px; font-weight: 750; margin-bottom: 0.15rem; line-height: 1.25;}
    .sub-title {color: #82909f; margin-bottom: 1rem;}
    div[data-testid="stExpander"] {
        border: 1px solid rgba(150, 170, 190, 0.28);
        border-radius: 12px;
        background: rgba(100, 120, 145, 0.06);
        margin: 0.4rem 0 1rem 0;
    }
    div[data-testid="stExpander"] details summary {font-weight: 700; padding: 0.55rem 0.8rem;}
    div[data-testid="stExpander"] [data-testid="stVerticalBlock"] {gap: 0.55rem;}
    div[data-testid="stMultiSelect"] [data-baseweb="tag"] {font-size: 0.78rem;}
    div[data-testid="stMultiSelect"] [data-baseweb="select"] > div {min-height: 2.55rem;}
    div[data-testid="stMetric"] {
        background: #111c2b;
        border: 1px solid #26374b;
        padding: 15px 16px;
        border-radius: 12px;
    }
    div[data-testid="stMetricLabel"] p {font-size: 14px;}
    div[data-testid="stMetricValue"] {font-size: clamp(18px, 1.8vw, 27px); overflow-wrap:anywhere;}
    section[data-testid="stSidebar"] {border-right: 1px solid #26374b;}
</style>
""", unsafe_allow_html=True)


# ----------------------------- HELPERS -----------------------------
def ensure_dataset_downloaded():
    """Download the public Google Drive dataset folder when CSVs are not present."""
    required_names = (
        ("customer.csv", "olist_customers_dataset.csv"),
        ("orders.csv", "olist_orders_dataset.csv"),
        ("order_items.csv", "olist_order_items_dataset.csv"),
        ("products.csv", "olist_products_dataset.csv"),
    )
    if all(any((DATA_DIR / name).is_file() for name in group) for group in required_names):
        return

    import gdown

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    folder_url = f"https://drive.google.com/drive/folders/{DATASET_DRIVE_FOLDER_ID}"
    try:
        downloaded = gdown.download_folder(
            url=folder_url,
            output=str(DATA_DIR),
            quiet=False,
            use_cookies=False,
            remaining_ok=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not download the clean_dataset folder from Google Drive. "
            "Set the folder's General access to 'Anyone with the link' (Viewer), "
            "and make sure the CSV files are directly inside that folder."
        ) from exc

    if not downloaded and not any(DATA_DIR.iterdir()):
        raise RuntimeError(
            "Google Drive returned no dataset files. Check the folder sharing "
            "permission: General access -> Anyone with the link -> Viewer."
        )


def find_csv(*names):
    for name in names:
        for folder in (DATA_DIR, ROOT):
            path = folder / name
            if path.exists():
                return path
    return None


@st.cache_data(show_spinner=False)
def load_data():
    ensure_dataset_downloaded()
    required = {
        "customers": find_csv("customer.csv", "olist_customers_dataset.csv"),
        "orders": find_csv("orders.csv", "olist_orders_dataset.csv"),
        "items": find_csv("order_items.csv", "olist_order_items_dataset.csv"),
        "products": find_csv("products.csv", "olist_products_dataset.csv"),
    }
    missing = [key for key, path in required.items() if path is None]
    if missing:
        raise FileNotFoundError(
            "Required CSV file(s) not found: " + ", ".join(missing)
            + ". The app downloads the clean_dataset folder from Google Drive; "
              "check the folder sharing permission and file names."
        )

    customers = pd.read_csv(required["customers"])
    orders = pd.read_csv(required["orders"])
    items = pd.read_csv(required["items"])
    products = pd.read_csv(required["products"])

    for frame in (customers, orders, items, products):
        frame.columns = frame.columns.str.strip()

    # Dates
    date_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col], errors="coerce")

    # Numeric sales columns
    for col in ("price", "freight_value"):
        if col in items.columns:
            items[col] = pd.to_numeric(items[col], errors="coerce").fillna(0)

    # Merge order items -> orders -> product -> customer
    df = items.merge(orders, on="order_id", how="left", suffixes=("", "_order"))

    product_fields = ["product_id"]
    if "product_category_name" in products.columns:
        product_fields.append("product_category_name")
    if "product_weight_g" in products.columns:
        product_fields.append("product_weight_g")
    df = df.merge(
        products[product_fields].drop_duplicates("product_id"),
        on="product_id", how="left"
    )

    customer_fields = ["customer_id"]
    for col in (
        "customer_state", "customer_unique_id", "customer_city",
        "customer_zip_code_prefix",
    ):
        if col in customers.columns:
            customer_fields.append(col)
    df = df.merge(
        customers[customer_fields].drop_duplicates("customer_id"),
        on="customer_id", how="left", suffixes=("", "_customer")
    )

    # Seller details (seller_state) for the Seller-Level dashboard.
    sellers_path = find_csv("sellers.csv", "olist_sellers_dataset.csv")
    if sellers_path:
        sellers = pd.read_csv(sellers_path)
        sellers.columns = sellers.columns.str.strip()
        seller_fields = ["seller_id"]
        if "seller_state" in sellers.columns:
            seller_fields.append("seller_state")
        df = df.merge(
            sellers[seller_fields].drop_duplicates("seller_id"),
            on="seller_id", how="left"
        )

    # Average review score per order, avoiding row multiplication.
    reviews_path = find_csv("order_reviews.csv", "olist_order_reviews_dataset.csv")
    if reviews_path:
        reviews = pd.read_csv(reviews_path)
        reviews.columns = reviews.columns.str.strip()
        if {"order_id", "review_score"}.issubset(reviews.columns):
            reviews["review_score"] = pd.to_numeric(
                reviews["review_score"], errors="coerce"
            )
            review_summary = (
                reviews.groupby("order_id", as_index=False)
                .agg(review_score=("review_score", "mean"))
            )
            df = df.merge(review_summary, on="order_id", how="left")

    # English product category labels, if translation CSV exists
    translation_path = find_csv("product_category_name_translation.csv")
    if translation_path and "product_category_name" in df.columns:
        translation = pd.read_csv(translation_path)
        translation.columns = translation.columns.str.strip()
        if {"product_category_name", "product_category_name_english"}.issubset(translation.columns):
            translation = translation.drop_duplicates("product_category_name")
            df = df.merge(translation, on="product_category_name", how="left")
            df["category"] = df["product_category_name_english"].fillna(
                df["product_category_name"]
            )
        else:
            df["category"] = df["product_category_name"]
    elif "product_category_name" in df.columns:
        df["category"] = df["product_category_name"]
    else:
        df["category"] = "Unknown"
    df["category"] = df["category"].fillna("Unknown").astype(str)

    # Payment data - aggregate to one row per order to avoid multiplying item rows.
    payment_path = find_csv("order_payments.csv", "olist_order_payments_dataset.csv")
    if payment_path:
        payments = pd.read_csv(payment_path)
        payments.columns = payments.columns.str.strip()
        if "order_id" in payments.columns:
            pay_parts = []
            if "payment_type" in payments.columns:
                payment_type = (
                    payments.groupby("order_id")["payment_type"]
                    .agg(lambda values: values.mode().iloc[0]
                         if not values.mode().empty else "unknown")
                    .rename("payment_type")
                )
                pay_parts.append(payment_type)
            if "payment_value" in payments.columns:
                payments["payment_value"] = pd.to_numeric(
                    payments["payment_value"], errors="coerce"
                ).fillna(0)
                pay_parts.append(
                    payments.groupby("order_id")["payment_value"].sum()
                )
            if "payment_installments" in payments.columns:
                payments["payment_installments"] = pd.to_numeric(
                    payments["payment_installments"], errors="coerce"
                )
                pay_parts.append(
                    payments.groupby("order_id")["payment_installments"].max()
                )
            if pay_parts:
                pay = pd.concat(pay_parts, axis=1).reset_index()
                df = df.merge(pay, on="order_id", how="left")
        else:
            df["payment_type"] = "Unknown"
    else:
        df["payment_type"] = "Payment data unavailable"

    # Month fields
    if "order_purchase_timestamp" in df.columns:
        df["Month_Year"] = df["order_purchase_timestamp"].dt.strftime("%b %Y")
        df["Month_Sort"] = (
            df["order_purchase_timestamp"].dt.year * 100
            + df["order_purchase_timestamp"].dt.month
        )
    else:
        df["Month_Year"] = "Unknown"
        df["Month_Sort"] = 0

    # Delivery flag: 1 if delivered on/before estimated date
    if {"order_delivered_customer_date", "order_estimated_delivery_date"}.issubset(df.columns):
        df["on_time"] = (
            df["order_delivered_customer_date"].notna()
            & df["order_estimated_delivery_date"].notna()
            & (df["order_delivered_customer_date"] <= df["order_estimated_delivery_date"])
        )
    else:
        df["on_time"] = False

    return df


def money(value):
    return f"R$ {value:,.2f}"


def money_compact(value):
    """Short KPI currency format; full amount remains available in the tooltip."""
    value = float(value or 0)
    absolute = abs(value)
    if absolute >= 1_000_000_000:
        return f"R$ {value / 1_000_000_000:.2f}B"
    if absolute >= 1_000_000:
        return f"R$ {value / 1_000_000:.2f}M"
    if absolute >= 1_000:
        return f"R$ {value / 1_000:.2f}K"
    return f"R$ {value:,.2f}"


LOGO_PATH = ROOT / "images.png"


# ----------------------------- LOGIN -----------------------------
def login_screen():
    st.markdown(
        """
        <style>
        .login-title {font-size:2rem; font-weight:800; text-align:center; margin-bottom:.2rem;}
        .login-sub {text-align:center; opacity:.72; margin-bottom:1.2rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    left, center, right = st.columns([1, 1.2, 1])
    with center:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=115)
        st.markdown(
            "<div class='login-title'>Olist Analytics Portal</div>"
            "<div class='login-sub'>Please sign in to continue</div>",
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            with st.form("login_form"):
                st.subheader("Login")
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                submitted = st.form_submit_button("Login", use_container_width=True)
                if submitted:
                    valid_user = username.strip().lower() == APP_USERNAME.strip().lower()
                    valid_pass = password.strip() == APP_PASSWORD.strip()
                    if valid_user and valid_pass:
                        st.session_state["authenticated"] = True
                        st.session_state["olist_data_ready"] = False
                        st.rerun()
                    else:
                        st.error("Incorrect username or password.")
        st.caption("Use your assigned Olist Analytics login credentials.")


# ----------------------------- EXECUTIVE SUMMARY -----------------------------
def _sync_select_all(checkbox_key, multiselect_key, options):
    # Runs before Streamlit redraws the widgets after the checkbox changes.
    st.session_state[multiselect_key] = list(options) if st.session_state[checkbox_key] else []


def executive_summary(df):
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='main-title'>Executive Summary Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Olist Brazilian E-Commerce Performance Overview</div>", unsafe_allow_html=True)

    # Filters / slicers
    with st.expander("Filters", expanded=False):
        st.markdown("**Filters & slicers**")
        f1, f2, f3 = st.columns(3)

        months_df = df[["Month_Year", "Month_Sort"]].drop_duplicates().sort_values("Month_Sort")
        months = months_df["Month_Year"].dropna().tolist()
        categories = sorted(df["category"].dropna().unique().tolist())
        states = sorted(df["customer_state"].dropna().astype(str).unique().tolist()) if "customer_state" in df.columns else []

        # Streamlit multiselect has no native Select All row inside its dropdown.
        # These checkboxes provide one-click Select All for each slicer.
        with f1:
            all_months = st.checkbox(
                "Select all months", value=True, key="sum_all_months",
                on_change=_sync_select_all,
                args=("sum_all_months", "sum_months", months),
            )
            chosen_months = st.multiselect(
                "Month & Year", months, default=months if all_months else [],
                key="sum_months", placeholder="Choose month(s)..."
            )
        with f2:
            all_categories = st.checkbox(
                "Select all categories", value=True, key="sum_all_categories",
                on_change=_sync_select_all,
                args=("sum_all_categories", "sum_categories", categories),
            )
            chosen_categories = st.multiselect(
                "Product Category", categories, default=categories if all_categories else [],
                key="sum_categories", placeholder="Choose category(s)..."
            )
        with f3:
            if states:
                all_states = st.checkbox(
                    "Select all states", value=True, key="sum_all_states",
                    on_change=_sync_select_all,
                    args=("sum_all_states", "sum_states", states),
                )
                chosen_states = st.multiselect(
                    "States", states, default=states if all_states else [],
                    key="sum_states", placeholder="Choose state(s)..."
                )
            else:
                chosen_states = []

    data = df.copy()
    data = data[data["Month_Year"].isin(chosen_months)]
    data = data[data["category"].isin(chosen_categories)]
    if "customer_state" in data.columns:
        data = data[data["customer_state"].astype(str).isin(chosen_states)]

    if data.empty:
        st.warning("No records match the selected filters. Please change the slicers.")
        return

    st.subheader("KPI")
    # KPI cards
    orders_count = data["order_id"].nunique()
    revenue = data["price"].sum()
    aov = revenue / orders_count if orders_count else 0
    customers_count = (
        data["customer_unique_id"].nunique()
        if "customer_unique_id" in data.columns
        else data["customer_id"].nunique()
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Orders", f"{orders_count:,.0f}")
    k2.metric("Total Revenue", money_compact(revenue), help=f"Exact total revenue: {money(revenue)}")
    k3.metric("Average Order Value", money(aov))
    k4.metric("Total Customers", f"{customers_count:,.0f}")

    st.divider()

    # Sales and order trend
    st.subheader("Sales & Orders Trend")
    trend_base = data.dropna(subset=["order_purchase_timestamp"]).copy()
    trend_base["month_date"] = trend_base["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    trend = trend_base.groupby("month_date", as_index=False).agg(
        Sales=("price", "sum"), Orders=("order_id", "nunique")
    ).sort_values("month_date")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=trend["month_date"], y=trend["Sales"], name="Sales"))
    fig.add_trace(go.Scatter(
        x=trend["month_date"], y=trend["Orders"], name="Count of Orders",
        mode="lines+markers", yaxis="y2"
    ))
    fig.update_layout(
        height=360, xaxis_title="Month & Year",
        yaxis=dict(title="Sales"),
        yaxis2=dict(title="Count of Orders", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=15, r=15, t=35, b=15),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Category and status charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top Product Categories by Revenue")
        cat = data.groupby("category", as_index=False)["price"].sum().nlargest(5, "price")
        cat = cat.sort_values("price")
        fig_cat = px.bar(
            cat, x="price", y="category", orientation="h", text_auto=".2s",
            labels={"price": "Revenue", "category": "Product Category"}
        )
        fig_cat.update_layout(height=350, margin=dict(l=10, r=10, t=25, b=15))
        st.plotly_chart(fig_cat, use_container_width=True)

    with c2:
        st.subheader("Order Status Breakdown")
        if "order_status" in data.columns:
            status = data.groupby("order_status")["order_id"].nunique().reset_index(name="Orders")
            fig_status = px.bar(
                status.sort_values("Orders", ascending=False),
                x="order_status", y="Orders", text="Orders",
                labels={"order_status": "Order Status"}
            )
            fig_status.update_layout(height=350, xaxis_tickangle=-35, margin=dict(l=10, r=10, t=25, b=15))
            st.plotly_chart(fig_status, use_container_width=True)
        else:
            st.info("order_status column not found.")

    # Payment treemap and delivery gauge
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Payment Method Share")
        payment_data = data.dropna(subset=["payment_type"])
        if not payment_data.empty and not payment_data["payment_type"].eq("Payment data unavailable").all():
            payment = payment_data.groupby("payment_type")["order_id"].nunique().reset_index(name="Orders")
            fig_pay = px.treemap(payment, path=["payment_type"], values="Orders", color="Orders")
            fig_pay.update_layout(height=330, margin=dict(l=5, r=5, t=25, b=5))
            st.plotly_chart(fig_pay, use_container_width=True)
        else:
            st.warning("order_payments.csv is required for this chart.")

    with c4:
        st.subheader("Delivery Performance Status")
        delivered = data[data["order_delivered_customer_date"].notna()] if "order_delivered_customer_date" in data.columns else data.iloc[0:0]
        score = float(delivered["on_time"].mean()) if not delivered.empty else 0
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            number={"valueformat": ".0%"},
            title={"text": "On-time Delivery Rate"},
            gauge={"axis": {"range": [0, 1]}, "bar": {"color": "#159DB5"}}
        ))
        gauge.update_layout(height=330, margin=dict(l=15, r=15, t=45, b=10))
        st.plotly_chart(gauge, use_container_width=True)

    st.caption(
        "Revenue is calculated as the sum of item price. Olist raw data does not contain "
        "a direct profit column, so the trend displays Sales and Order Count."
    )


# ----------------------------- OTHER DASHBOARD PAGES -----------------------------
def _sync_customer_filter(checkbox_key, multiselect_key, options):
    st.session_state[multiselect_key] = (
        list(options) if st.session_state[checkbox_key] else []
    )


def customer_level_dashboard(df):
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='main-title'>Customer-Level Analysis | Who Buys, Where & How</div>",
        unsafe_allow_html=True,
    )

    # Build one row per order so item-level rows do not inflate order/customer counts.
    required_cols = {"order_id", "customer_unique_id", "order_purchase_timestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error("Customer dashboard ke liye required columns missing hain: " + ", ".join(sorted(missing)))
        return

    order_cols = [
        "order_id", "customer_unique_id", "customer_id",
        "order_purchase_timestamp", "customer_state", "customer_city",
        "customer_zip_code_prefix", "payment_type", "order_status",
    ]
    order_cols = [c for c in order_cols if c in df.columns]
    orders_view = df[order_cols].drop_duplicates("order_id").copy()

    # Filters / slicers
    with st.expander("Filters", expanded=False):
        st.markdown("**Filters & slicers**")
        f1, f2, f3, f4 = st.columns(4)

        def options_for(col):
            if col not in orders_view.columns:
                return []
            return sorted(orders_view[col].dropna().astype(str).unique().tolist())

        state_options = options_for("customer_state")
        payment_options = options_for("payment_type")
        status_options = options_for("order_status")
        zip_options = options_for("customer_zip_code_prefix")

        with f1:
            all_states = st.checkbox(
                "Select all states", value=True, key="cust_all_states",
                on_change=_sync_customer_filter,
                args=("cust_all_states", "cust_states", state_options),
            )
            selected_states = st.multiselect(
                "States", state_options,
                default=state_options if all_states else [],
                key="cust_states", placeholder="Select state(s)",
            )
        with f2:
            all_payments = st.checkbox(
                "Select all payment modes", value=True, key="cust_all_payments",
                on_change=_sync_customer_filter,
                args=("cust_all_payments", "cust_payments", payment_options),
            )
            selected_payments = st.multiselect(
                "Mode of Payment", payment_options,
                default=payment_options if all_payments else [],
                key="cust_payments", placeholder="Select payment mode(s)",
            )
        with f3:
            all_statuses = st.checkbox(
                "Select all status", value=True, key="cust_all_statuses",
                on_change=_sync_customer_filter,
                args=("cust_all_statuses", "cust_statuses", status_options),
            )
            selected_statuses = st.multiselect(
                "Status of Order", status_options,
                default=status_options if all_statuses else [],
                key="cust_statuses", placeholder="Select order status(es)",
            )
        with f4:
            st.markdown("ZIP Code")

            all_zips = st.checkbox(
                "Select All ZIPs",
                value=True,
                key="cust_all_zips"
            )

            zip_search = st.text_input(
                "Search ZIP Code",
                placeholder="Enter ZIP Code prefix",
                key="cust_zip_search"
            )

    if all_zips:
        selected_zips = zip_options
    elif zip_search.strip():
        selected_zips = [
            z for z in zip_options
            if z.startswith(zip_search.strip())
        ]
    else:
        selected_zips = []

    filtered_orders = orders_view.copy()
    if selected_states and "customer_state" in filtered_orders:
        filtered_orders = filtered_orders[
            filtered_orders["customer_state"].astype(str).isin(selected_states)
        ]
    else:
        filtered_orders = filtered_orders.iloc[0:0]
    if selected_payments and "payment_type" in filtered_orders:
        filtered_orders = filtered_orders[
            filtered_orders["payment_type"].astype(str).isin(selected_payments)
        ]
    else:
        filtered_orders = filtered_orders.iloc[0:0]
    if selected_statuses and "order_status" in filtered_orders:
        filtered_orders = filtered_orders[
            filtered_orders["order_status"].astype(str).isin(selected_statuses)
        ]
    else:
        filtered_orders = filtered_orders.iloc[0:0]
    if not all_zips and zip_search.strip():
            filtered_orders = filtered_orders[
        filtered_orders["customer_zip_code_prefix"].astype(str).str.startswith(zip_search.strip())
        ]

    if filtered_orders.empty:
        st.warning("Selected filters ke liye data available nahi hai.")
        return

    # Calculate item revenue for only the filtered order IDs.
    selected_order_ids = set(filtered_orders["order_id"])
    filtered_items = df[df["order_id"].isin(selected_order_ids)].copy()
    unique_customers = filtered_orders["customer_unique_id"].nunique()
    total_customer_orders = filtered_orders["order_id"].nunique()
    total_spend = pd.to_numeric(filtered_items["price"], errors="coerce").fillna(0).sum()
    average_spend = total_spend / unique_customers if unique_customers else 0

    order_frequency = (
        filtered_orders.groupby("customer_unique_id")["order_id"]
        .nunique()
    )
    repeat_customers = int((order_frequency > 1).sum())
    repeat_rate = repeat_customers / unique_customers if unique_customers else 0

    # KPI heading and exactly four KPI cards.
    st.subheader("KPI")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Unique Customers", f"{unique_customers:,.2f}")
    k2.metric("Total Customer Orders", f"{total_customer_orders:,.2f}")
    k3.metric("Average Spend Per Customer", f"{average_spend:,.2f}")
    k4.metric("Customer Repeat %", f"{repeat_rate:.2%}")

    st.divider()

    # Customer count by state
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Total Customers Per State")
        state_data = (
            filtered_orders.groupby("customer_state")["customer_unique_id"]
            .nunique().reset_index(name="Customers")
            .sort_values("Customers", ascending=False).head(10)
        )
        fig_state = px.bar(
            state_data, x="customer_state", y="Customers", text="Customers",
            labels={"customer_state": "Customer State"},
        )
        fig_state.update_layout(height=330, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_state, use_container_width=True)

    # Customer count by city
    with c2:
        selected_state_label = ", ".join(selected_states) if selected_states else "Selected State"
        st.subheader(f"Total Orders by City")
        if "customer_city" in filtered_orders.columns:
            city_data = (
                filtered_orders.groupby("customer_city")["order_id"]
                .nunique().reset_index(name="Orders")
                .sort_values("Orders", ascending=False).head(15)
                .sort_values("Orders", ascending=True)
            )
            fig_city = px.bar(
                city_data, x="Orders", y="customer_city",
                orientation="h", text="Orders",
                labels={"customer_city": "Customer City", "Orders": "Total Orders"},
            )
            fig_city.update_traces(textposition="outside", cliponaxis=False)
            fig_city.update_layout(
                height=390, margin=dict(l=15, r=35, t=25, b=15),
                yaxis_title="City", xaxis_title="Number of Orders",
            )
            st.plotly_chart(fig_city, use_container_width=True)
        else:
            st.info("customer_city column not found.")

    # Payment method share
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Payment Mode Distribution %")
        pay_data = (
            filtered_orders.groupby("payment_type")["order_id"]
            .nunique().reset_index(name="Orders")
        )
        fig_pay = px.pie(
            pay_data, names="payment_type", values="Orders", hole=0.58,
        )
        fig_pay.update_traces(
            textposition="inside",
            texttemplate="%{label}<br>%{percent:.1%}",
            insidetextorientation="radial",
            insidetextfont=dict(size=12),
            hovertemplate="%{label}<br>Orders: %{value:,}<br>Share: %{percent:.1%}<extra></extra>",
        )
        fig_pay.update_layout(
            height=390, margin=dict(l=15, r=15, t=25, b=15),
            legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
            uniformtext_minsize=10, uniformtext_mode="hide",
        )
        st.plotly_chart(fig_pay, use_container_width=True)

    # Monthly new customers based on each customer's first purchase in filtered data.
    with c4:
        st.subheader("Customer Purchase Frequency")
        frequency = order_frequency.value_counts().sort_index()
        frequency_rows = []
        one_order = int(frequency.get(1, 0))
        two_orders = int(frequency.get(2, 0))
        three_plus = int(frequency[frequency.index >= 3].sum())
        frequency_rows = [
            {"Purchase Group": "1 Order", "Customers": one_order},
            {"Purchase Group": "2 Orders", "Customers": two_orders},
            {"Purchase Group": "3+ Orders", "Customers": three_plus},
        ]
        freq_df = pd.DataFrame(frequency_rows)
        fig_freq = px.bar(
            freq_df, x="Purchase Group", y="Customers", text="Customers",
        )
        fig_freq.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_freq, use_container_width=True)

    st.subheader("Monthly New Customer")
    first_purchase = (
        filtered_orders.groupby("customer_unique_id", as_index=False)
        ["order_purchase_timestamp"].min()
    )
    first_purchase["Month"] = (
        first_purchase["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    )
    monthly_new = (
        first_purchase.groupby("Month").size().reset_index(name="New Customers")
        .sort_values("Month")
    )
    fig_month = px.line(
        monthly_new, x="Month", y="New Customers", markers=True,
        labels={"Month": "Month & Year"},
    )
    fig_month.update_layout(height=330, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_month, use_container_width=True)



def _sync_seller_filter(checkbox_key, multiselect_key, options):
    st.session_state[multiselect_key] = (
        list(options) if st.session_state[checkbox_key] else []
    )


def seller_level_dashboard(df):
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='main-title'>Seller-Level Analysis | Seller Ranking & Performance</div>",
        unsafe_allow_html=True,
    )

    needed = {"seller_id", "order_id", "price"}
    missing = needed - set(df.columns)
    if missing:
        st.error("Seller dashboard ke liye required columns missing hain: " + ", ".join(sorted(missing)))
        return

    # One row per seller-order for order counts and review/shipping metrics.
    order_seller_cols = [
        "seller_id", "order_id", "seller_state", "order_purchase_timestamp",
        "order_estimated_delivery_date", "order_delivered_customer_date",
        "review_score",
    ]
    order_seller_cols = [c for c in order_seller_cols if c in df.columns]
    seller_orders = df[order_seller_cols].drop_duplicates(
        ["seller_id", "order_id"]
    ).copy()

    # Estimated shipping days: purchase timestamp to estimated delivery date.
    if {
        "order_purchase_timestamp", "order_estimated_delivery_date"
    }.issubset(seller_orders.columns):
        seller_orders["estimated_days_to_ship"] = (
            seller_orders["order_estimated_delivery_date"]
            - seller_orders["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400
        seller_orders.loc[
            seller_orders["estimated_days_to_ship"] < 0,
            "estimated_days_to_ship"
        ] = pd.NA
    else:
        seller_orders["estimated_days_to_ship"] = pd.NA

    # Filters
    with st.expander("Filters", expanded=False):
        st.markdown("**Filters & slicers**")
        f1, f2 = st.columns(2)
        states = (
            sorted(seller_orders["seller_state"].dropna().astype(str).unique().tolist())
            if "seller_state" in seller_orders.columns else []
        )
        seller_ids = sorted(seller_orders["seller_id"].dropna().astype(str).unique().tolist())

        with f1:
            all_seller_states = st.checkbox(
                "Select all states", value=True, key="seller_all_states",
                on_change=_sync_seller_filter,
                args=("seller_all_states", "seller_states", states),
            )
            selected_seller_states = st.multiselect(
                "States", states,
                default=states if all_seller_states else [],
                key="seller_states", placeholder="Select state(s)",
            )
        with f2:
            all_seller_ids = st.checkbox(
                "Select all seller IDs", value=True, key="seller_all_ids",
                on_change=_sync_seller_filter,
                args=("seller_all_ids", "seller_ids", seller_ids),
            )
            selected_seller_ids = st.multiselect(
                "ID of Sellers", seller_ids,
                default=seller_ids if all_seller_ids else [],
                key="seller_ids", placeholder="Select seller ID(s)",
            )

    filtered_orders = seller_orders.copy()
    if "seller_state" in filtered_orders.columns:
        if selected_seller_states:
            filtered_orders = filtered_orders[
                filtered_orders["seller_state"].astype(str).isin(selected_seller_states)
            ]
        else:
            filtered_orders = filtered_orders.iloc[0:0]
    if selected_seller_ids:
        filtered_orders = filtered_orders[
            filtered_orders["seller_id"].astype(str).isin(selected_seller_ids)
        ]
    else:
        filtered_orders = filtered_orders.iloc[0:0]

    if filtered_orders.empty:
        st.warning("Selected filters ke liye seller data available nahi hai.")
        return

    selected_seller_order_pairs = filtered_orders[["seller_id", "order_id"]].drop_duplicates()
    filtered_items = df.merge(
        selected_seller_order_pairs, on=["seller_id", "order_id"], how="inner"
    )

    seller_count = filtered_orders["seller_id"].nunique()
    total_sales = pd.to_numeric(filtered_items["price"], errors="coerce").fillna(0).sum()
    avg_ship_days = filtered_orders["estimated_days_to_ship"].mean()
    avg_review = (
        filtered_orders["review_score"].mean()
        if "review_score" in filtered_orders.columns else float("nan")
    )

    st.subheader("KPI")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Sellers", f"{seller_count:,.0f}")
    k2.metric("Total Sales", money_compact(total_sales), help=f"Exact total sales: {money(total_sales)}")
    k3.metric(
        "Average Estimated Days to Ship",
        f"{avg_ship_days:.2f}" if pd.notna(avg_ship_days) else "N/A",
    )
    k4.metric(
        "Average Review Score",
        f"{avg_review:.2f}" if pd.notna(avg_review) else "N/A",
    )

    st.divider()

    # Aggregate sales and product count per seller.
    seller_sales = (
        filtered_items.groupby("seller_id", as_index=False)
        .agg(
            Total_Sales=("price", "sum"),
            Products_Sold=("product_id", "nunique"),
            Order_Count=("order_id", "nunique"),
        )
    )
    seller_sales["seller_short"] = seller_sales["seller_id"].astype(str).str.slice(0, 8) + "…"
    seller_ship = (
        filtered_orders.groupby("seller_id", as_index=False)
        .agg(
            Average_Shipping_Days=("estimated_days_to_ship", "mean"),
            Average_Review_Score=("review_score", "mean")
            if "review_score" in filtered_orders.columns
            else ("estimated_days_to_ship", "count"),
        )
    )
    seller_ship["seller_short"] = seller_ship["seller_id"].astype(str).str.slice(0, 8) + "…"

    # First row: Top sellers by revenue + sellers by state.
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top 10 Sellers By Revenue")
        top_revenue = seller_sales.nlargest(10, "Total_Sales").sort_values("Total_Sales")
        fig_revenue = px.bar(
            top_revenue.assign(seller_short=top_revenue["seller_id"].astype(str).str.slice(0, 8) + "…"),
            x="Total_Sales", y="seller_short", orientation="h",
            text_auto=".2s",
            labels={"Total_Sales": "Total Sales", "seller_short": "Seller ID"},
        )
        fig_revenue.update_layout(height=350, margin=dict(l=8, r=8, t=20, b=10))
        st.plotly_chart(fig_revenue, use_container_width=True)

    with c2:
        st.subheader("No. of Sellers Per States")
        if "seller_state" in filtered_orders.columns:
            state_counts = (
                filtered_orders.groupby("seller_state")["seller_id"]
                .nunique().reset_index(name="Sellers")
                .sort_values("Sellers", ascending=False)
            )
            fig_states = px.bar(
                state_counts, x="seller_state", y="Sellers", text="Sellers",
                labels={"seller_state": "Seller State"},
            )
            fig_states.update_layout(
                height=350, xaxis_tickangle=-35,
                margin=dict(l=8, r=8, t=20, b=10),
            )
            st.plotly_chart(fig_states, use_container_width=True)
        else:
            st.info("sellers.csv mein seller_state column nahi mila.")

    # More spacious lower section: two charts on the first row, one wide chart below.
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("No. of Orders Per Sellers")
        orders_per_seller = (
            seller_sales.nlargest(10, "Order_Count")
            .sort_values("Order_Count")
            .copy()
        )
        orders_per_seller["seller_short"] = orders_per_seller["seller_id"].astype(str).str.slice(0, 8) + "…"
        fig_orders = px.bar(
            orders_per_seller, x="Order_Count", y="seller_short",
            orientation="h", text="Order_Count",
            labels={"seller_short": "Seller ID", "Order_Count": "Orders"},
        )
        fig_orders.update_traces(textposition="outside", cliponaxis=False)
        fig_orders.update_layout(
            height=390, margin=dict(l=10, r=35, t=25, b=15),
            yaxis_title="Seller ID", xaxis_title="Number of Orders",
        )
        st.plotly_chart(fig_orders, use_container_width=True)

    with c4:
        st.subheader("Average Days to Shipping by Sellers")
        ship_top = (
            seller_ship.dropna(subset=["Average_Shipping_Days"])
            .nlargest(10, "Average_Shipping_Days")
            .sort_values("Average_Shipping_Days")
            .copy()
        )
        fig_ship = px.line(
            ship_top, x="seller_short", y="Average_Shipping_Days",
            markers=True, text=ship_top["Average_Shipping_Days"].round(1),
            labels={
                "seller_short": "Seller ID",
                "Average_Shipping_Days": "Average Estimated Days",
            },
        )
        fig_ship.update_traces(line=dict(width=3), marker=dict(size=9), textposition="top center")
        fig_ship.update_layout(
            height=390, xaxis_tickangle=-35,
            margin=dict(l=10, r=10, t=35, b=55),
            yaxis_title="Estimated Days",
        )
        st.plotly_chart(fig_ship, use_container_width=True)

    st.subheader("No. of Products Selling by Each Sellers")
    product_top = (
        seller_sales.nlargest(10, "Products_Sold")
        .sort_values("Products_Sold")
        .copy()
    )
    product_top["seller_short"] = product_top["seller_id"].astype(str).str.slice(0, 8) + "…"
    fig_products = px.bar(
        product_top, x="seller_short", y="Products_Sold",
        text="Products_Sold",
        labels={"seller_short": "Seller ID", "Products_Sold": "Products"},
    )
    fig_products.update_traces(textposition="outside", cliponaxis=False)
    fig_products.update_layout(
        height=410, xaxis_tickangle=-35,
        margin=dict(l=15, r=15, t=25, b=75),
        yaxis_title="Distinct Products Sold",
    )
    st.plotly_chart(fig_products, use_container_width=True)

    st.caption(
        "Shipping days are calculated as order purchase timestamp to estimated delivery date. "
        "Average review score is based on available order review scores. "
        "Products sold means distinct product IDs sold by each seller."
    )



def _sync_order_filter(checkbox_key, multiselect_key, options):
    st.session_state[multiselect_key] = (
        list(options) if st.session_state[checkbox_key] else []
    )


def orders_level_dashboard(df):
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='main-title'>Orders & Payments-Level Analysis | Order Journey</div>",
        unsafe_allow_html=True,
    )

    required = {"order_id", "price", "order_status", "order_purchase_timestamp"}
    missing = required - set(df.columns)
    if missing:
        st.error("Orders dashboard ke liye required columns missing hain: " + ", ".join(sorted(missing)))
        return

    # Build order-level table to prevent item rows from inflating order counts.
    order_columns = [
        "order_id", "order_status", "order_purchase_timestamp",
        "payment_type", "payment_value", "payment_installments",
    ]
    order_columns = [c for c in order_columns if c in df.columns]
    order_view = df[order_columns].drop_duplicates("order_id").copy()

    # Order-level value = sum of item prices for each order.
    order_values = (
        df.groupby("order_id", as_index=False)["price"].sum()
        .rename(columns={"price": "order_value"})
    )
    order_view = order_view.merge(order_values, on="order_id", how="left")
    order_view["Month_Year"] = order_view["order_purchase_timestamp"].dt.strftime("%b %Y")
    order_view["Month_Sort"] = (
        order_view["order_purchase_timestamp"].dt.year * 100
        + order_view["order_purchase_timestamp"].dt.month
    )
    order_view["Weekday"] = order_view["order_purchase_timestamp"].dt.day_name()

    with st.expander("Filters", expanded=False):
        st.markdown("**Filters & slicers**")
        f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.4, 1.8])

        statuses = sorted(order_view["order_status"].dropna().astype(str).unique().tolist())
        payment_modes = (
            sorted(order_view["payment_type"].dropna().astype(str).unique().tolist())
            if "payment_type" in order_view.columns else []
        )
        month_options = (
            order_view[["Month_Year", "Month_Sort"]].drop_duplicates()
            .sort_values("Month_Sort")["Month_Year"].dropna().tolist()
        )
        min_price = float(order_view["order_value"].min()) if not order_view.empty else 0.0
        max_price = float(order_view["order_value"].max()) if not order_view.empty else 0.0

        with f1:
            all_statuses = st.checkbox(
                "Select all statuses", value=True, key="orders_all_statuses",
                on_change=_sync_order_filter,
                args=("orders_all_statuses", "orders_statuses", statuses),
            )
            selected_statuses = st.multiselect(
                "Status of Order", statuses,
                default=statuses if all_statuses else [],
                key="orders_statuses", placeholder="Select status",
            )
        with f2:
            all_payment_modes = st.checkbox(
                "Select all payment modes", value=True, key="orders_all_payments",
                on_change=_sync_order_filter,
                args=("orders_all_payments", "orders_payments", payment_modes),
            )
            selected_payment_modes = st.multiselect(
                "Mode of Payment", payment_modes,
                default=payment_modes if all_payment_modes else [],
                key="orders_payments", placeholder="Select payment mode",
            )
        with f3:
            all_months = st.checkbox(
                "Select all months", value=True, key="orders_all_months",
                on_change=_sync_order_filter,
                args=("orders_all_months", "orders_months", month_options),
            )
            selected_months = st.multiselect(
                "Month & Year", month_options,
                default=month_options if all_months else [],
                key="orders_months", placeholder="Select month(s)",
            )
        with f4:
            st.caption("Price Range (order value)")
            if min_price < max_price:
                selected_price_range = st.slider(
                    "Price Range", min_value=min_price, max_value=max_price,
                    value=(min_price, max_price), key="orders_price_range",
                    label_visibility="collapsed",
                )
            else:
                selected_price_range = (min_price, max_price)
                st.caption(f"Only one price value: {min_price:,.2f}")

    filtered = order_view[
        order_view["order_status"].astype(str).isin(selected_statuses)
        & order_view["Month_Year"].isin(selected_months)
        & order_view["order_value"].between(
            selected_price_range[0], selected_price_range[1], inclusive="both"
        )
    ].copy()
    if "payment_type" in filtered.columns:
        if selected_payment_modes:
            filtered = filtered[filtered["payment_type"].astype(str).isin(selected_payment_modes)]
        else:
            filtered = filtered.iloc[0:0]

    if filtered.empty:
        st.warning("Selected filters ke liye data available nahi hai.")
        return

    total_orders = filtered["order_id"].nunique()
    delivered_orders = int((filtered["order_status"] == "delivered").sum())
    cancelled_orders = int((filtered["order_status"] == "canceled").sum())
    average_order_value = filtered["order_value"].mean() if total_orders else 0

    st.subheader("KPI")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Orders", f"{total_orders:,.0f}")
    k2.metric("Total Delivered Orders", f"{delivered_orders:,.0f}")
    k3.metric("Total Cancelled Orders", f"{cancelled_orders:,.0f}")
    k4.metric("Average Order Value", f"{average_order_value:,.2f}")

    st.divider()

    # Heatmap: Purchase Hour vs Day-wise Order Analysis
    st.subheader("Purchase Hour vs Day-wise Order Analysis")

    heatmap_data = filtered.copy()

    if {"order_id", "order_purchase_timestamp"}.issubset(heatmap_data.columns):

        heatmap_data["order_purchase_timestamp"] = pd.to_datetime(
                heatmap_data["order_purchase_timestamp"],
                errors="coerce"
            )

        heatmap_data = heatmap_data.dropna(
                subset=["order_id", "order_purchase_timestamp"]
            ).copy()

        if not heatmap_data.empty:

            heatmap_data["Purchase Hour"] = (
                    heatmap_data["order_purchase_timestamp"].dt.hour
                )

            heatmap_data["Day Name"] = (
                    heatmap_data["order_purchase_timestamp"].dt.day_name()
                )

            weekday_order = [
                    "Friday", "Monday", "Saturday",
                    "Sunday", "Thursday", "Tuesday", "Wednesday"
                ]

            heatmap = (
                    heatmap_data.groupby(
                        ["Purchase Hour", "Day Name"]
                    )["order_id"]
                    .nunique()
                    .unstack(fill_value=0)
                )

            heatmap = heatmap.reindex(
                    index=range(24),
                    columns=weekday_order,
                    fill_value=0
                )

            fig_heat = px.imshow(
                    heatmap,
                    aspect="auto",
                    text_auto=True,
                    labels={
                        "x": "Day Name",
                        "y": "Purchase Hour",
                        "color": "Total Orders"
                    },
                    color_continuous_scale="Blues"
                )

            fig_heat.update_layout(
                    height=440,
                    margin=dict(l=10, r=10, t=20, b=10)
                )

            st.plotly_chart(
                    fig_heat,
                    use_container_width=True
                )

        else:
            st.info("No order data available for the selected filters.")

    else:
        st.info("Required order columns are missing.")
            
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Transactions By Different Payment Method")
        if {"payment_type", "payment_value"}.issubset(filtered.columns):
            payment_summary = (
                filtered.groupby("payment_type", as_index=False)["payment_value"]
                .sum().sort_values("payment_value", ascending=False)
            )
            fig_payment = px.bar(
                payment_summary, x="payment_value", y="payment_type",
                orientation="h", text_auto=".2s",
                labels={"payment_value": "Payment Value", "payment_type": "Payment Method"},
            )
            fig_payment.update_layout(height=350, yaxis={"categoryorder": "total ascending"}, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_payment, use_container_width=True)
        else:
            st.warning("Payment value data unavailable.")

    with c2:
        st.subheader("Status of Orders")
        status_counts = (
            filtered.groupby("order_status")["order_id"].nunique()
            .reset_index(name="Orders")
        )
        status_counts["Share"] = status_counts["Orders"] / status_counts["Orders"].sum()
        fig_status = px.bar(
            status_counts.sort_values("Share", ascending=True),
            x="Share", y="order_status", orientation="h",
            text=status_counts.sort_values("Share", ascending=True)["Share"].map(lambda x: f"{x:.2%}"),
            labels={"Share": "% Count of Orders", "order_status": "Order Status"},
        )
        fig_status.update_layout(
            height=350, xaxis_tickformat=".0%",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_status, use_container_width=True)

    st.subheader("Payment Installment Analysis")
    if "payment_installments" in filtered.columns:
        installment = (
            filtered.dropna(subset=["payment_installments"])
            .groupby("payment_installments")["order_id"].nunique()
            .reset_index(name="Orders")
            .sort_values("payment_installments")
        )
        fig_install = px.bar(
            installment, x="payment_installments", y="Orders", text="Orders",
            labels={"payment_installments": "Payment Installments", "Orders": "Count of Orders"},
        )
        fig_install.update_layout(height=330, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_install, use_container_width=True)
    else:
        st.warning("Payment installments data unavailable.")

    st.caption(
        "Average Order Value is calculated as total item price per order. "
        "Payment transactions use payment_value from order_payments.csv. "
        "Installment heatmap counts distinct orders by purchase weekday and installment count."
    )


def product_level_dashboard(df):
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='main-title'>Product-Level Analysis | Categories & Revenue</div>",
        unsafe_allow_html=True,
    )

    required = {"product_id", "order_id", "price", "category"}
    missing = required - set(df.columns)
    if missing:
        st.error("Product dashboard ke liye required columns missing hain: " + ", ".join(sorted(missing)))
        return

    with st.expander("Filters", expanded=False):
        st.markdown("**Filters & slicers**")
        f1, f2, f3 = st.columns(3)
        categories = sorted(df["category"].dropna().astype(str).unique().tolist())
        states = sorted(df["customer_state"].dropna().astype(str).unique().tolist()) if "customer_state" in df.columns else []
        month_options = (
            df[["Month_Year", "Month_Sort"]].drop_duplicates()
            .sort_values("Month_Sort")["Month_Year"].dropna().tolist()
        )

        with f1:
            all_categories = st.checkbox(
                "Select all categories", value=True, key="product_all_categories",
                on_change=_sync_order_filter,
                args=("product_all_categories", "product_categories", categories),
            )
            selected_categories = st.multiselect(
                "Product Category", categories,
                default=categories if all_categories else [],
                key="product_categories", placeholder="Select category(s)",
            )
        with f2:
            all_states = st.checkbox(
                "Select all states", value=True, key="product_all_states",
                on_change=_sync_order_filter,
                args=("product_all_states", "product_states", states),
            )
            selected_states = st.multiselect(
                "States", states,
                default=states if all_states else [],
                key="product_states", placeholder="Select state(s)",
            )
        with f3:
            all_months = st.checkbox(
                "Select all months", value=True, key="product_all_months",
                on_change=_sync_order_filter,
                args=("product_all_months", "product_months", month_options),
            )
            selected_months = st.multiselect(
                "Month & Year", month_options,
                default=month_options if all_months else [],
                key="product_months", placeholder="Select month(s)",
            )

    filtered = df[
        df["category"].astype(str).isin(selected_categories)
        & df["Month_Year"].isin(selected_months)
    ].copy()
    if "customer_state" in filtered.columns:
        if selected_states:
            filtered = filtered[filtered["customer_state"].astype(str).isin(selected_states)]
        else:
            filtered = filtered.iloc[0:0]

    if filtered.empty:
        st.warning("Selected filters ke liye product data available nahi hai.")
        return

    unique_products = filtered["product_id"].nunique()
    category_count = filtered["category"].nunique()
    average_price = pd.to_numeric(filtered["price"], errors="coerce").mean()
    average_weight = (
        pd.to_numeric(filtered["product_weight_g"], errors="coerce").mean()
        if "product_weight_g" in filtered.columns else float("nan")
    )

    st.subheader("KPI")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Products", f"{unique_products:,.0f}")
    k2.metric("Total Product Categories", f"{category_count:,.0f}")
    k3.metric("Average Price", f"{average_price:,.2f}")
    k4.metric("Average Product Weight (g)", f"{average_weight:,.2f}" if pd.notna(average_weight) else "N/A")

    st.divider()

    # Aggregate product-category and state metrics.
    category_orders = (
        filtered.groupby("category")["order_id"].nunique()
        .reset_index(name="Orders")
    )
    category_revenue = (
        filtered.groupby("category")["price"].sum()
        .reset_index(name="Revenue")
    )

    # Top category charts: two wide visuals with distinct orientations.
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top Product Category by No. of Orders")
        top_orders = category_orders.nlargest(10, "Orders").sort_values("Orders")
        fig_orders = px.bar(
            top_orders, x="Orders", y="category", orientation="h",
            text="Orders", labels={"category": "Product Category"},
        )
        fig_orders.update_traces(textposition="outside", cliponaxis=False)
        fig_orders.update_layout(
            height=420, margin=dict(l=10, r=35, t=25, b=15),
            xaxis_title="Number of Orders",
        )
        st.plotly_chart(fig_orders, use_container_width=True)

    with c2:
        st.subheader("Top Product Categories by Revenue")
        top_revenue = category_revenue.nlargest(10, "Revenue").sort_values("Revenue")
        fig_revenue = px.bar(
            top_revenue, x="Revenue", y="category", orientation="h",
            text_auto=".2s", labels={"category": "Product Category"},
        )
        fig_revenue.update_traces(textposition="outside", cliponaxis=False)
        fig_revenue.update_layout(
            height=420, margin=dict(l=10, r=40, t=25, b=15),
            xaxis_title="Revenue (R$)",
        )
        st.plotly_chart(fig_revenue, use_container_width=True)

    # Full-width 100% stacked comparison. Percentages are calculated explicitly;
    # Plotly Express px.bar does not accept barnorm as an argument.
    st.subheader("Top Selling Product Categories Across States")
    if "customer_state" in filtered.columns:
        state_source = filtered.copy()
        # Normalize state labels so whitespace/case variants cannot create duplicate bars.
        state_source["customer_state"] = (
            state_source["customer_state"].astype(str).str.strip().str.upper()
        )
        state_product = (
            state_source.groupby(["customer_state", "category"])["order_id"]
            .nunique().reset_index(name="Orders")
        )
        top_state_categories = (
            state_product.groupby("category")["Orders"].sum()
            .nlargest(3).index.tolist()
        )
        state_product = state_product[
            state_product["category"].isin(top_state_categories)
        ].copy()
        state_totals = state_product.groupby("customer_state")["Orders"].transform("sum")
        state_product["Share"] = (
            state_product["Orders"].div(state_totals.where(state_totals.ne(0), 1)) * 100
        )
        fig_state = px.bar(
            state_product, x="Share", y="customer_state",
            color="category", orientation="h", barmode="stack",
            hover_data={"Orders": True, "Share": ":.1f", "customer_state": True},
            labels={
                "customer_state": "Customer State",
                "category": "Product Category",
                "Share": "Share of Orders (%)",
            },
        )
        fig_state.update_layout(
            height=440, xaxis_range=[0, 100],
            xaxis_ticksuffix="%",
            margin=dict(l=15, r=20, t=25, b=20),
            legend_title_text="Product Category",
        )
        st.plotly_chart(fig_state, use_container_width=True)
    else:
        st.info("Customer state column not found.")

    # Monthly trend and average price share a spacious two-column row.
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Monthly Trend of Product Category by Revenue")
        monthly = filtered.dropna(subset=["order_purchase_timestamp"]).copy()
        monthly["Month"] = monthly["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
        top_five = category_revenue.nlargest(5, "Revenue")["category"].tolist()
        monthly = monthly[monthly["category"].isin(top_five)]
        monthly_revenue = (
            monthly.groupby(["Month", "category"], as_index=False)["price"].sum()
            .rename(columns={"price": "Revenue"})
        )
        fig_monthly = px.line(
            monthly_revenue, x="Month", y="Revenue", color="category", markers=True,
            labels={"Month": "Month & Year", "category": "Product Category"},
        )
        fig_monthly.update_layout(
            height=410, margin=dict(l=10, r=10, t=25, b=20),
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    with c4:
        st.subheader("Average Price of Product Category")
        avg_price = (
            filtered.groupby("category")["price"].mean()
            .reset_index(name="Average Price")
            .nlargest(10, "Average Price")
            .sort_values("Average Price")
        )
        fig_avg = px.bar(
            avg_price, x="Average Price", y="category", orientation="h",
            text_auto=".2f", labels={"category": "Product Category"},
        )
        fig_avg.update_traces(textposition="outside", cliponaxis=False)
        fig_avg.update_layout(
            height=410, margin=dict(l=10, r=40, t=25, b=15),
            xaxis_title="Average Price (R$)",
        )
        st.plotly_chart(fig_avg, use_container_width=True)

    st.caption(
        "Product weight is the average product_weight_g from products.csv. "
        "Revenue is the sum of order item price. State comparison uses customer_state."
    )


def simple_dashboard_page(title, description):
    st.markdown(f"<div class='main-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sub-title'>{description}</div>", unsafe_allow_html=True)
    st.info("This dashboard section is ready as a navigation page. Its charts will be added in the next step.")


# ----------------------------- PREDICTION PAGE -----------------------------


# ----------------------------- PREDICTION PAGE -----------------------------
# Google Drive file ID from the shared link provided.
# The Drive file must be the single review_model_bundle.pkl file and must be
# shared as: General access -> Anyone with the link -> Viewer.
MODEL_DRIVE_FILE_ID = "1rSmKltAWYiWuYvgwte4TOrEjktD_QBM-"
MODEL_PATH = ROOT / "review_model_bundle.pkl"


@st.cache_resource(show_spinner="Downloading and loading review prediction models...")
def load_review_models():
    """Download the compact hierarchical PKL bundle from Google Drive and load it."""
    import gdown

    if not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size == 0:
        try:
            downloaded_path = gdown.download(
                id=MODEL_DRIVE_FILE_ID,
                output=str(MODEL_PATH),
                quiet=False,
                fuzzy=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not download the prediction model from Google Drive. "
                "Check that the file is shared as 'Anyone with the link – Viewer'."
            ) from exc

        if not downloaded_path or not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size == 0:
            raise RuntimeError(
                "Google Drive did not return the PKL file. Check the Drive sharing "
                "permission and confirm the link points directly to review_model_bundle.pkl."
            )

    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            "The downloaded file could not be opened as a Joblib PKL bundle. "
            "The Drive link must point to the single review_model_bundle.pkl exported "
            "by the final hierarchical-model export cell, not a folder or ZIP file."
        ) from exc
    required = [
        "binary_model", "happy_model", "unhappy_model", "feature_columns",
        "category_mapping", "payment_mapping",
        "customer_state_mapping", "seller_state_mapping"
    ]
    missing = [key for key in required if key not in bundle]
    if missing:
        raise ValueError(f"Model bundle is incomplete. Missing: {missing}")

    features = list(bundle["feature_columns"])
    expected_features = [
        "has_delivery_conflict", "has_carrier_conflict", "customer_state",
        "price", "freight_value", "payment_type", "payment_installments",
        "payment_value", "product_name_length", "product_description_length",
        "product_photos_qty", "product_weight_g", "seller_state",
        "delivery_delay", "days_to_deliver",
        "shipping_gap_days_after_purschase", "approval_delay_hours",
        "product_category_encoded"
    ]
    if features != expected_features:
        raise ValueError(
            "Feature columns in review_model_bundle.pkl do not match the app inputs. "
            "Re-export the bundle using the 18 leakage-free features from the notebook."
        )

    for stage in ("binary_model", "happy_model", "unhappy_model"):
        model = bundle[stage]
        expected = getattr(model, "n_features_in_", len(features))
        if int(expected) != len(features):
            raise ValueError(
                f"{stage} was trained with {expected} features, but this app supplies "
                f"{len(features)}. The notebook models must be trained once using the "
                "18 features (excluding review_answer_gap), then exported again."
            )

    return {
        "binary": bundle["binary_model"],
        "happy": bundle["happy_model"],
        "unhappy": bundle["unhappy_model"],
        "features": features,
        "categories": bundle["category_mapping"],
        "payments": bundle["payment_mapping"],
        "customer_states": bundle["customer_state_mapping"],
        "seller_states": bundle["seller_state_mapping"],
        "hierarchy": bundle.get("hierarchy", {})
    }


def prediction_page():
    st.markdown("<div class='main-title' style='margin-top:0.65rem;'>Review Score Prediction</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Estimate the customer's review score (1–5 stars) "
        "using order, delivery, payment, product and location details.</div>",
        unsafe_allow_html=True,
    )

    try:
        models = load_review_models()
    except Exception as exc:
        st.error(f"Prediction model loading error: {exc}")
        st.info(
            "Keep review_model_bundle.pkl in the same folder as app.py. "
            "The bundle must contain the mappings created from le1, le2, le3 and le4 "
            "in the notebook's basic LabelEncoder cell."
        )
        return

    # Keep the prediction result above the input form so it remains visible after submission.
    result_area = st.empty()

    def mapping_options(mapping):
        return sorted(mapping.keys(), key=lambda value: str(value))

    with st.container():
        st.subheader("Delivery")
        a, b, c = st.columns(3)
        with a:
            delay = st.number_input(
                "Delay vs estimated delivery (days)", -90, 180, 0,
                help="Positive = late delivery; negative = early delivery."
            )
            total_days = st.number_input("Total delivery time (days)", 0, 180, 7)
        with b:
            shipping_gap = st.number_input("Purchase to shipping (days)", 0, 90, 5)
            approval_delay = st.number_input("Order approval time (hours)", 0.0, 720.0, 1.0)
        with c:
            delivery_conflict = st.selectbox("Delivery date conflict", ["No", "Yes"])
            carrier_conflict = st.selectbox("Carrier date conflict", ["No", "Yes"])

        st.divider()
        st.subheader("Payment")
        a, b, c = st.columns(3)
        with a:
            payment_label = st.selectbox(
                "Payment method", mapping_options(models["payments"])
            )
        with b:
            installments = st.number_input("Number of installments", 1, 30, 1)
        with c:
            payment_value = st.number_input("Payment amount (R$)", 0.0, 100000.0, 100.0)

        st.divider()
        st.subheader("Product")
        a, b, c = st.columns(3)
        with a:
            category_label = st.selectbox(
                "Product category", mapping_options(models["categories"])
            )
            price = st.number_input("Item price (R$)", 0.0, 100000.0, 100.0)
        with b:
            freight = st.number_input("Freight amount (R$)", 0.0, 10000.0, 15.0)
            weight = st.number_input("Product weight (grams)", 0, 100000, 1000)
        with c:
            name_len = st.number_input("Product name length", 0, 300, 40)
            desc_len = st.number_input("Product description length", 0, 5000, 500)
            photos = st.number_input("Number of product photos", 0, 50, 2)

        st.divider()
        st.subheader("Customer and Seller Location")
        a, b = st.columns(2)
        with a:
            customer_state = st.selectbox(
                "Customer state", mapping_options(models["customer_states"])
            )
        with b:
            seller_state = st.selectbox(
                "Seller state", mapping_options(models["seller_states"])
            )

        predict_clicked = st.button(
            "Predict Review Score", type="primary", use_container_width=True,
            key="predict_review_score_button"
        )

    if not predict_clicked:
        return

    # Keep the original model input names and provide spelling/name aliases used by
    # different notebook exports. This fixes the purchase_to_shipping_gap mismatch.
    values = {
        "has_delivery_conflict": int(delivery_conflict == "Yes"),
        "has_carrier_conflict": int(carrier_conflict == "Yes"),
        "customer_state": models["customer_states"][customer_state],
        "price": float(price),
        "freight_value": float(freight),
        "payment_type": models["payments"][payment_label],
        "payment_installments": int(installments),
        "payment_value": float(payment_value),
        "product_name_length": int(name_len),
        "product_description_length": int(desc_len),
        "product_photos_qty": int(photos),
        "product_weight_g": int(weight),
        "seller_state": models["seller_states"][seller_state],
        "delivery_delay": int(delay),
        "days_to_deliver": int(total_days),
        "shipping_gap_days_after_purschase": int(shipping_gap),
        "shipping_gap_days_after_purchase": int(shipping_gap),
        "purchase_to_shipping_gap": int(shipping_gap),
        "purchase_to_shipping_gap_days": int(shipping_gap),
        "approval_delay_hours": float(approval_delay),
        "product_category_encoded": models["categories"][category_label],
    }

    try:
        def make_model_input(feature_names, stage_name):
            missing_features = [name for name in feature_names if name not in values]
            if missing_features:
                raise ValueError(
                    f"{stage_name} model expects feature(s) not supplied by the form: "
                    f"{missing_features}. The model's original feature list and the "
                    "app's input fields must come from the same training notebook export."
                )
            return pd.DataFrame(
                [[values[name] for name in feature_names]],
                columns=feature_names,
            ).astype(float)

        x_input = make_model_input(models["features"], "Hierarchical")

        # Stage 1 chooses one branch; only that branch predicts the final rating.
        binary_prediction = int(models["binary"].predict(x_input)[0])
        if binary_prediction == 1:
            predicted_rating = int(models["happy"].predict(x_input)[0])
            branch_label = "Happy branch (4-5 stars)"
        else:
            predicted_rating = int(models["unhappy"].predict(x_input)[0])
            branch_label = "Unhappy branch (1-3 stars)"

        if predicted_rating not in (1, 2, 3, 4, 5):
            raise ValueError(f"Model returned an invalid review score: {predicted_rating}")
        stars = "★" * predicted_rating + "☆" * (5 - predicted_rating)

        with result_area.container():
            st.divider()
            st.subheader("Predicted Review Score")
            left, right = st.columns([1, 1.6])
            with left:
                st.markdown(
                    f"""
                    <div style="border:1px solid rgba(120,150,170,.35);
                                border-radius:16px;padding:22px;text-align:center;">
                      <div style="font-size:13px;opacity:.72;">PREDICTED RATING</div>
                      <div style="font-size:48px;font-weight:850;">{predicted_rating} / 5</div>
                      <div style="font-size:28px;color:#e6a700;letter-spacing:3px;">{stars}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with right:
                st.markdown("**Prediction hierarchy**")
                st.success(f"Selected: {branch_label}")
                st.caption(
                    "Stage 1 classifies the review as happy or unhappy. "
                    "The selected branch then predicts the final star rating."
                )

            if delay >= 10:
                st.warning(
                    f"The entered delivery delay is {delay} days. "
                    "The predicted score is an estimate based on historical patterns."
                )

    except Exception as exc:
        with result_area.container():
            st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
            st.info(
                "The app now checks each model's own feature list. If a feature is still "
                "missing, export the matching feature list and model together from the "
                "same notebook run."
            )


# ----------------------------- MAIN APP -----------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "olist_data_ready" not in st.session_state:
    st.session_state["olist_data_ready"] = False
if "main_section" not in st.session_state:
    st.session_state["main_section"] = "Dashboard"
if "dashboard_page" not in st.session_state:
    st.session_state["dashboard_page"] = "Executive Summary"


def _on_main_section_change():
    if st.session_state.get("main_section") == "Dashboard":
        st.session_state["dashboard_page"] = "Executive Summary"


if not st.session_state["authenticated"]:
    login_screen()
    st.stop()

# Centered loading view; login form is not rendered while data loads.
try:
    if not st.session_state["olist_data_ready"]:
        loading_area = st.empty()
        with loading_area.container():
            if LOGO_PATH.exists():
                logo_left, logo_center, logo_right = st.columns([1, 1.2, 1])
                with logo_center:
                    st.image(str(LOGO_PATH), width=125)
            st.markdown(
                """
                <style>
                .loading-card {
                    max-width:650px; margin:2.6rem auto 1rem auto; padding:2.4rem 1.5rem;
                    text-align:center; border:1px solid rgba(170,190,210,.30);
                    border-radius:22px; background:rgba(20,32,48,.78);
                    box-shadow:0 14px 40px rgba(0,0,0,.22);
                }
                .olist-spinner {
                    width:66px; height:66px; margin:0 auto 1.3rem auto;
                    border:6px solid rgba(150,190,205,.20);
                    border-top:6px solid #35c5b0; border-right:6px solid #35c5b0;
                    border-radius:50%; animation:olist-spin 1s linear infinite;
                }
                @keyframes olist-spin {to {transform:rotate(360deg);}}
                .loading-heading {font-size:1.65rem; font-weight:800; margin-bottom:.35rem;}
                .loading-copy {opacity:.72; font-size:.98rem;}
                </style>
                <div class="loading-card">
                    <div class="olist-spinner"></div>
                    <div class="loading-heading">Preparing Your Dashboard</div>
                    <div class="loading-copy">
                        Loading and connecting customers, orders, products and payments.
                        <br>First-time loading may take a little longer.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.spinner("Loading Olist data..."):
                dataset = load_data()
        loading_area.empty()
        st.session_state["olist_data_ready"] = True
    else:
        dataset = load_data()

except FileNotFoundError as e:
    st.error(f"Dataset file missing: {e}")
    st.info("The app downloads CSV files from the shared Google Drive clean_dataset folder. Check that the folder is shared as Anyone with the link – Viewer.")
    st.stop()
except Exception as e:
    st.error(f"Data loading failed: {type(e).__name__}: {e}")
    st.warning("The dashboard was not opened because the data could not be loaded completely. Check the VS Code Terminal for the full error.")
    st.stop()

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
    st.title("Olist Analytics")
    st.caption("Executive Dashboard")
    st.radio(
        "Main Section",
        ["Dashboard", "Prediction"],
        key="main_section",
        on_change=_on_main_section_change,
    )
    if st.button("Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["olist_data_ready"] = False
        st.rerun()

if st.session_state["main_section"] == "Dashboard":
    st.sidebar.radio(
        "Dashboard Pages",
        ["Executive Summary", "Customer level", "Seller level", "Orders level", "Product level"],
        key="dashboard_page",
    )
    dashboard_page = st.session_state["dashboard_page"]

    if dashboard_page == "Executive Summary":
        executive_summary(dataset)
    elif dashboard_page == "Customer level":
        customer_level_dashboard(dataset)
    elif dashboard_page == "Seller level":
        seller_level_dashboard(dataset)
    elif dashboard_page == "Orders level":
        orders_level_dashboard(dataset)
    elif dashboard_page == "Product level":
        product_level_dashboard(dataset)
else:
    prediction_page()
