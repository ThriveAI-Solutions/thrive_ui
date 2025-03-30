from pathlib import Path

import plotly.express as px
import polars as pl
import streamlit as st

TEST_RESULTS_PATH = Path("analytics")
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

chart_data = (
    df.group_by(
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

# Convert options to native Python lists
domain_options = df.select(pl.col("additionalMetadata.domain_classification").unique()).to_series().to_list()
metric_options = df.select(pl.col("evaluation_metric").unique()).to_series().to_list()
run_options = df["run"].unique().to_list()

# Initialize session state only once
if "selected_domains" not in st.session_state:
    st.session_state["selected_domains"] = domain_options
if "selected_metrics" not in st.session_state:
    st.session_state["selected_metrics"] = ["Correctness (GEval)"]
if "selected_runs" not in st.session_state:
    st.session_state["selected_runs"] = [max(run_options)]

# Create multiselect widgets using session state keys
domains = st.multiselect(
    label="Show Domains:", options=domain_options, default=st.session_state["selected_domains"], key="selected_domains"
)

metrics = st.multiselect(
    label="Show Metrics:", options=metric_options, default=st.session_state["selected_metrics"], key="selected_metrics"
)

runs = st.multiselect(
    label="Select Run(s):", options=run_options, default=st.session_state["selected_runs"], key="selected_runs"
)

# Use the selections to filter your data
chart_data_filtered = chart_data.filter(
    pl.col("domain").is_in(domains),
    pl.col("evaluation_metric").is_in(metrics),
    pl.col("run").is_in(runs),
)

fig = px.pie(
    chart_data_filtered,
    values="cnt",
    height=500 if (len(metrics) * 250) < 500 else (len(metrics) * 250),
    color="evaluation_success",
    names="evaluation_success",
    facet_row="evaluation_metric",
    facet_col="domain",
    hole=0.5,
    title="Donut Charts for Evaluation Metrics by Domain",
    color_discrete_map={
        True: "#3cb371",
        False: "#ff4500",
    },
)

fig.update_traces(textposition="inside", textinfo="percent+label")
for annotation in fig.layout.annotations:
    annotation.text = annotation.text.split("=")[-1]
    annotation.textangle = 0

if domains:
    st.plotly_chart(fig)

df_filtered = df.filter(pl.col("run").is_in(runs), pl.col("evaluation_metric").is_in(metrics))

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
