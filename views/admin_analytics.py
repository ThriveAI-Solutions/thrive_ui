import math

import pandas as pd
import streamlit as st
from sqlalchemy import case, func

from orm.models import Message, SessionLocal, User
from utils.enums import MessageType, RoleType


def _kpi_card(label: str, value, help_text: str | None = None):
    c = st.container(border=True)
    with c:
        st.markdown(f"**{label}**")
        st.markdown(f"<h3 style='margin-top:0'>{value}</h3>", unsafe_allow_html=True)
        if help_text:
            st.caption(help_text)


def _read_metrics(days: int = 30):
    chart_types = [
        MessageType.PLOTLY_CHART.value,
        MessageType.ST_LINE_CHART.value,
        MessageType.ST_BAR_CHART.value,
        MessageType.ST_AREA_CHART.value,
        MessageType.ST_SCATTER_CHART.value,
    ]
    import datetime as _dt

    since = _dt.datetime.now() - _dt.timedelta(days=days)

    with SessionLocal() as session:
        users_count = session.query(func.count()).select_from(User).scalar() or 0
        active_users = (
            session.query(func.count(func.distinct(Message.user_id)))
            .filter(Message.role == RoleType.USER.value, Message.created_at >= since)
            .scalar()
            or 0
        )

        # Over-time counts
        date_expr = func.strftime("%Y-%m-%d", Message.created_at)
        over_time = (
            session.query(
                date_expr.label("d"),
                func.sum(case((Message.role == RoleType.USER.value, 1), else_=0)).label("questions"),
                func.sum(case((Message.type.in_(chart_types), 1), else_=0)).label("charts"),
                func.sum(case((Message.type == MessageType.SUMMARY.value, 1), else_=0)).label("summaries"),
                func.sum(case((Message.type == MessageType.SQL.value, 1), else_=0)).label("sql"),
                func.sum(case((Message.type == MessageType.ERROR.value, 1), else_=0)).label("errors"),
            )
            .filter(Message.created_at >= since)
            .group_by("d")
            .order_by("d")
            .all()
        )

        # Over-time conversion: results (charts+dataframes+summaries) / questions
        result_over_time = (
            session.query(
                date_expr.label("d"),
                func.sum(case((Message.role == RoleType.USER.value, 1), else_=0)).label("questions"),
                func.sum(
                    case(
                        (
                            Message.type.in_([MessageType.DATAFRAME.value, MessageType.SUMMARY.value] + chart_types),
                            1,
                        ),
                        else_=0,
                    )
                ).label("results"),
            )
            .filter(Message.created_at >= since)
            .group_by("d")
            .order_by("d")
            .all()
        )

        # Elapsed stats (seconds) for assistant messages
        # Pull raw elapsed times and compute stats in Python for SQLite compatibility
        def _compute_stats_for_query(q):
            import numpy as np

            vals = [float(v or 0.0) for (v,) in q if v is not None]
            n = len(vals)
            if n == 0:
                return {"avg": 0.0, "min": 0.0, "max": 0.0, "stddev": 0.0, "median": 0.0, "n": 0}
            arr = np.array(vals, dtype=float)
            return {
                "avg": float(arr.mean()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "stddev": float(arr.std(ddof=0)),
                "median": float(np.median(arr)),
                "n": int(n),
            }

        overall_q = (
            session.query(Message.elapsed_time)
            .filter(Message.role == RoleType.ASSISTANT.value, Message.created_at >= since)
            .all()
        )
        overall_stats = _compute_stats_for_query(overall_q)

        def _perf_for_type(msg_type: str):
            q = (
                session.query(Message.elapsed_time)
                .filter(
                    Message.role == RoleType.ASSISTANT.value,
                    Message.type == msg_type,
                    Message.created_at >= since,
                )
                .all()
            )
            return _compute_stats_for_query(q)

        perf_types = {
            "sql": _perf_for_type(MessageType.SQL.value),
            "summary": _perf_for_type(MessageType.SUMMARY.value),
            "chart": _perf_for_type(MessageType.PLOTLY_CHART.value),
            "dataframe": _perf_for_type(MessageType.DATAFRAME.value),
        }

        return (
            users_count,
            active_users,
            over_time,
            result_over_time,
            overall_stats,
            perf_types,
        )


def _to_dense_days(rows, days: int):
    import datetime as _dt

    today = _dt.date.today()
    start = today - _dt.timedelta(days=days - 1)
    by_date = {r.d: r for r in rows}
    out = []
    for i in range(days):
        d = start + _dt.timedelta(days=i)
        k = d.strftime("%Y-%m-%d")
        r = by_date.get(k)
        out.append(
            {
                "date": k,
                "questions": int((r.questions if r else 0) or 0),
                "charts": int((r.charts if r else 0) or 0),
                "summaries": int((r.summaries if r else 0) or 0),
                "sql": int((r.sql if r else 0) or 0),
                "errors": int((r.errors if r else 0) or 0),
            }
        )
    return pd.DataFrame(out)


def _guard_admin():
    from orm.models import RoleTypeEnum

    if st.session_state.get("user_role") != RoleTypeEnum.ADMIN.value:
        st.error("You don't have permission to view this page.")
        st.stop()


def main():
    _guard_admin()

    st.title("Admin Analytics")
    st.caption("Usage and performance insights for administrators")

    days = st.segmented_control("Range", options=["7 days", "30 days"], selection_mode="single", default="30 days")
    days_int = 30 if (days or "30 days").startswith("30") else 7

    (
        users_count,
        active_users,
        over_time,
        result_over_time,
        overall_stats,
        perf_types,
    ) = _read_metrics(days=days_int)

    # KPIs row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _kpi_card("Users", users_count)
    # Totals over range
    df = _to_dense_days(over_time, days_int)
    with k2:
        _kpi_card("Questions", int(df["questions"].sum()))
    with k3:
        _kpi_card("Errors", int(df["errors"].sum()))
    with k4:
        _kpi_card("Active Users", active_users)

    st.divider()

    # Over time chart
    if not df.empty:
        import plotly.express as px

        mdf = df.melt(
            id_vars=["date"],
            value_vars=["questions", "sql", "summaries", "charts", "errors"],
            var_name="metric",
            value_name="count",
        )
        mdf["date"] = pd.to_datetime(mdf["date"])
        fig = px.line(mdf, x="date", y="count", color="metric", markers=True)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

        # Conversion over time: build dense frame for questions/results
        import datetime as _dt

        start = (pd.Timestamp.today().normalize() - pd.Timedelta(days=days_int - 1)).date()
        by_date = {r.d: {"questions": int(r.questions or 0), "results": int(r.results or 0)} for r in result_over_time}
        conv_rows = []
        for i in range(days_int):
            d = (pd.Timestamp(start) + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            vals = by_date.get(d, {"questions": 0, "results": 0})
            q = vals["questions"]
            res = vals["results"]
            conv_rows.append({"date": d, "questions": q, "results": res, "conversion": (res / q) if q else 0})
        conv_df = pd.DataFrame(conv_rows)
        conv_df["date"] = pd.to_datetime(conv_df["date"])
        cfig = px.line(conv_df, x="date", y="conversion", title="Conversion (results/questions) over time")
        cfig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(cfig, use_container_width=True)

    st.divider()

    # Elapsed stats
    st.subheader("Performance (Assistant elapsed time)")
    avg = float(overall_stats.get("avg", 0))
    _min = float(overall_stats.get("min", 0))
    _max = float(overall_stats.get("max", 0))
    stddev = float(overall_stats.get("stddev", 0))
    median = float(overall_stats.get("median", 0))
    n = int(overall_stats.get("n", 0))
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _kpi_card("Avg (s)", round(avg, 3))
    with c2:
        _kpi_card("Min (s)", round(_min, 3))
    with c3:
        _kpi_card("Max (s)", round(_max, 3))
    with c4:
        _kpi_card("Std Dev (s)", round(stddev, 3))
    with c5:
        _kpi_card("Median (s)", round(median, 3))

    # Per-type stats (SQL, Summary, Chart)
    st.subheader("Performance by Output Type")
    pt_df = pd.DataFrame(
        [
            {
                "Type": "SQL",
                "Avg (s)": perf_types["sql"]["avg"],
                "Min (s)": perf_types["sql"]["min"],
                "Max (s)": perf_types["sql"]["max"],
                "Std Dev (s)": perf_types["sql"]["stddev"],
                "Median (s)": perf_types["sql"]["median"],
                "Samples": perf_types["sql"]["n"],
            },
            {
                "Type": "Summary",
                "Avg (s)": perf_types["summary"]["avg"],
                "Min (s)": perf_types["summary"]["min"],
                "Max (s)": perf_types["summary"]["max"],
                "Std Dev (s)": perf_types["summary"]["stddev"],
                "Median (s)": perf_types["summary"]["median"],
                "Samples": perf_types["summary"]["n"],
            },
            {
                "Type": "Chart",
                "Avg (s)": perf_types["chart"]["avg"],
                "Min (s)": perf_types["chart"]["min"],
                "Max (s)": perf_types["chart"]["max"],
                "Std Dev (s)": perf_types["chart"]["stddev"],
                "Median (s)": perf_types["chart"]["median"],
                "Samples": perf_types["chart"]["n"],
            },
            {
                "Type": "DataFrame",
                "Avg (s)": perf_types["dataframe"]["avg"],
                "Min (s)": perf_types["dataframe"]["min"],
                "Max (s)": perf_types["dataframe"]["max"],
                "Std Dev (s)": perf_types["dataframe"]["stddev"],
                "Median (s)": perf_types["dataframe"]["median"],
                "Samples": perf_types["dataframe"]["n"],
            },
        ]
    )
    st.dataframe(pt_df, use_container_width=True, hide_index=True)

    # Distribution
    with SessionLocal() as session:
        dist = (
            session.query(func.coalesce(Message.elapsed_time, 0).label("elapsed"))
            .filter(Message.role == RoleType.ASSISTANT.value)
            .all()
        )
    if dist:
        import numpy as np
        import plotly.express as px

        arr = np.array([float(r.elapsed or 0) for r in dist])
        hist = pd.DataFrame({"elapsed": arr})
        fig2 = px.histogram(hist, x="elapsed", nbins=40, title="Elapsed time distribution (s)")
        fig2.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Top Users by Questions
    st.subheader("Top Users by Questions")
    import datetime as _dt

    import plotly.express as px

    since = _dt.datetime.now() - _dt.timedelta(days=days_int)
    with SessionLocal() as session:
        top_rows = (
            session.query(User.username, func.count().label("questions"))
            .join(Message, Message.user_id == User.id)
            .filter(Message.role == RoleType.USER.value, Message.created_at >= since)
            .group_by(User.username)
            .order_by(func.count().desc())
            .limit(10)
            .all()
        )
    if top_rows:
        top_df = pd.DataFrame(top_rows, columns=["User", "Questions"])
        bar = px.bar(top_df, x="Questions", y="User", orientation="h")
        bar.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(bar, use_container_width=True)

    st.divider()

    # Latest Questions
    st.subheader("Latest Questions")
    limit = st.selectbox("Show last", options=[25, 50, 100, 200], index=0)
    with SessionLocal() as session:
        base = (
            session.query(
                Message.content.label("question"),
                Message.created_at.label("asked_at"),
                User.username.label("username"),
            )
            .join(User, User.id == Message.user_id)
            .filter(Message.role == RoleType.USER.value)
            .order_by(Message.created_at.desc())
            .limit(int(limit))
            .all()
        )
        questions = [row.question for row in base]
        chart_types = [
            MessageType.PLOTLY_CHART.value,
            MessageType.ST_LINE_CHART.value,
            MessageType.ST_BAR_CHART.value,
            MessageType.ST_AREA_CHART.value,
            MessageType.ST_SCATTER_CHART.value,
        ]
        agg = []
        if questions:
            agg = (
                session.query(
                    Message.question.label("q"),
                    func.sum(case((Message.type.in_(chart_types), 1), else_=0)).label("charts"),
                    func.sum(case((Message.type == MessageType.DATAFRAME.value, 1), else_=0)).label("dataframes"),
                    func.sum(case((Message.type == MessageType.SUMMARY.value, 1), else_=0)).label("summaries"),
                    func.sum(case((Message.type == MessageType.ERROR.value, 1), else_=0)).label("errors"),
                    func.sum(func.coalesce(Message.elapsed_time, 0)).label("elapsed"),
                )
                .filter(
                    Message.role == RoleType.ASSISTANT.value,
                    Message.question.isnot(None),
                    Message.question.in_(questions),
                )
                .group_by(Message.question)
                .all()
            )
    metrics = {row.q: row for row in agg} if agg else {}
    latest_rows = []
    for question, asked_at, username in base:
        m = metrics.get(question)
        errors = int(getattr(m, "errors", 0) or 0) if m else 0
        success = (
            int(getattr(m, "charts", 0) or 0)
            + int(getattr(m, "dataframes", 0) or 0)
            + int(getattr(m, "summaries", 0) or 0)
        ) > 0 and errors == 0
        latest_rows.append(
            {
                "User": username,
                "Question": question,
                "Asked At": asked_at,
                "Status": "Success" if success else ("Error" if errors > 0 else "Unknown"),
                "Elapsed (s)": round(float(getattr(m, "elapsed", 0) or 0.0), 3) if m else 0.0,
            }
        )
    if latest_rows:
        ldf = pd.DataFrame(latest_rows)
        st.dataframe(ldf, use_container_width=True, hide_index=True)

    st.divider()

    # All Users Stats (counts per user)
    st.subheader("All Users Stats")
    with SessionLocal() as session:
        rows = (
            session.query(
                User.username,
                func.sum(case((Message.role == RoleType.USER.value, 1), else_=0)).label("questions"),
                func.sum(case((Message.type == MessageType.DATAFRAME.value, 1), else_=0)).label("dataframes"),
                func.sum(case((Message.type == MessageType.SUMMARY.value, 1), else_=0)).label("summaries"),
                func.sum(case((Message.type.in_(chart_types), 1), else_=0)).label("charts"),
                func.sum(case((Message.type == MessageType.ERROR.value, 1), else_=0)).label("errors"),
            )
            .join(Message, Message.user_id == User.id, isouter=True)
            .group_by(User.username)
            .all()
        )
    if rows:
        audf = pd.DataFrame(rows, columns=["User", "Questions", "DataFrames", "Summaries", "Charts", "Errors"])
        st.dataframe(
            audf.sort_values(["Questions", "Errors"], ascending=[False, True]),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
