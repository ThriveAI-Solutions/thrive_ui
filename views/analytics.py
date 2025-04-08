from pathlib import Path

import altair as alt
import pandas as pd
import polars as pl
import streamlit as st

TEST_RESULTS_PATH = Path("analytics")

# Read and validate data
try:
    df = pl.read_parquet(TEST_RESULTS_PATH / "*.parquet", include_file_paths="run")
except pl.exceptions.ComputeError:
    st.error("Error reading parquet files. Make sure files are in the correct directory.")
    st.stop()
except FileNotFoundError:
    st.error("No parquet files found")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()

# Get filter options
domain_options = df.select(pl.col("additionalMetadata.domain_classification").unique()).to_series().to_list()
metric_options = df.select(pl.col("evaluation_metric").unique()).to_series().to_list()
run_options = df["run"].unique().to_list()

# Initialize session state
if "selected_domains" not in st.session_state:
    st.session_state["selected_domains"] = domain_options
if "selected_metrics" not in st.session_state:
    st.session_state["selected_metrics"] = ["Correctness (GEval)"]
if "selected_runs" not in st.session_state:
    st.session_state["selected_runs"] = [max(run_options)]

# Create filter widgets
domains = st.multiselect(
    label="Show Domains:", options=domain_options, default=st.session_state["selected_domains"], key="selected_domains"
)

metrics = st.multiselect(
    label="Show Metrics:", options=metric_options, default=st.session_state["selected_metrics"], key="selected_metrics"
)

runs = st.multiselect(
    label="Select Run(s):", options=run_options, default=st.session_state["selected_runs"], key="selected_runs"
)

# Prepare chart data
chart_data = (
    df.filter(
        pl.col("additionalMetadata.domain_classification").is_in(domains),
        pl.col("evaluation_metric").is_in(metrics),
        pl.col("run").is_in(runs),
    )
    .group_by(
        [
            "additionalMetadata.domain_classification",
            "evaluation_metric",
            "evaluation_success",
            "run",
        ]
    )
    .len(name="cnt")
    .rename({"additionalMetadata.domain_classification": "domain"})
)

# Calculate percentage correct for each domain and metric
percentage_data = (
    chart_data.group_by(["domain", "evaluation_metric"])
    .agg(
        pl.col("cnt").sum().alias("total"),
        pl.col("cnt").filter(pl.col("evaluation_success")).sum().alias("correct"),
    )
    .with_columns((pl.col("correct") / pl.col("total") * 100).round(1).alias("percentage_correct"))
)

# Create percentage lookup dictionary
percentage_dict = {
    (row["domain"], row["evaluation_metric"]): row["percentage_correct"]
    for row in percentage_data.iter_rows(named=True)
}

# Prepare data for Altair chart
chart_df = []
for metric in metrics:
    for domain in domains:
        # Filter data for this domain and metric
        domain_metric_data = chart_data.filter((pl.col("domain") == domain) & (pl.col("evaluation_metric") == metric))

        # Get the success and failure counts
        success_data = {row["evaluation_success"]: row["cnt"] for row in domain_metric_data.to_dicts()}
        success_count = success_data.get(True, 0)
        failure_count = success_data.get(False, 0)

        # Get percentage
        percentage = percentage_dict.get((domain, metric), 0)

        # Add the data points
        chart_df.extend(
            [
                {
                    "domain": domain,
                    "metric": metric,
                    "category": "Success",
                    "value": success_count,
                    "percentage": percentage,
                    "color": "#3cb371",
                },
                {
                    "domain": domain,
                    "metric": metric,
                    "category": "Failure",
                    "value": failure_count,
                    "percentage": percentage,
                    "color": "#ff4500",
                },
            ]
        )

# Create and display chart
if chart_df:
    chart_df = pd.DataFrame(chart_df)

    # Create base chart
    base = alt.Chart(chart_df)

    # Create donut layer
    donut = (
        base.mark_arc(innerRadius=70)
        .encode(
            theta=alt.Theta(field="value", type="quantitative", stack=True),
            color=alt.Color(
                "category:N",
                scale=alt.Scale(domain=["Success", "Failure"], range=["#3cb371", "#ff4500"]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(width=300, height=200)
    )

    # Create text layer
    text = (
        base.mark_text(align="center", baseline="middle", fontSize=24)
        .transform_aggregate(percentage="mean(percentage)", groupby=["domain", "metric"])
        .transform_calculate(text_display="format(datum.percentage, '.1f') + '%'")
        .encode(text="text_display:N")
    )

    # Combine layers and create faceted chart
    chart = alt.layer(donut, text).facet(row="metric:N", column="domain:N").resolve_scale(theta="independent")

    # Display chart
    st.altair_chart(chart, use_container_width=True)

# Display filtered dataframe
df_filtered = df.filter(
    pl.col("run").is_in(runs),
    pl.col("evaluation_metric").is_in(metrics),
    pl.col("additionalMetadata.domain_classification").is_in(domains),
)

event = st.dataframe(
    data=df_filtered,
    hide_index=True,
    selection_mode="single-row",
    column_order=(
        "outcome",
        "run",
        "name",
        "evaluation_metric",
        "input",
        "additionalMetadata.domain_classification",
        "evaluation_score",
    ),
    column_config={
        "evaluation_score": st.column_config.ProgressColumn(
            "Evaluation Score",
            help="How the model performed on this record",
            format="%f",
            min_value=0,
            max_value=1,
        )
    },
    on_select="rerun",
)


@st.dialog(title="Metric Details:", width="large")
def metric_details(selected_rows):
    if selected_rows is not None:
        selected_data = df_filtered.slice(selected_rows[0], 1).to_pandas().T
        st.write(selected_data)
    else:
        st.write("No row selected")


selected_rows = event.selection.rows
if selected_rows:
    metric_details(selected_rows)
