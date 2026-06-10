"""Admin → Analytics sub-tab (Epic #144 / #147).

Consolidates the former Admin Analytics and Agentic Analytics top-level
pages into a single 4-inner-tab surface (Overview · LLM Performance ·
Errors · Agentic). The User Activity tab from the old Admin Analytics
is relocated to Admin → Audit → User Activity (Epic #144 / #148).

Chart cuts applied via Epic #144 in the source modules: Top Users bar +
All Users Stats table (Overview), Activity Type pie (User Activity, now
under Audit), Errors severity pie.
"""

import streamlit as st

# Reuse the per-tab render functions from views/admin_analytics.py (which
# is now an internal helpers module — its top-level main() is no longer
# invoked as a page, but the _render_*_tab() functions are imported here).
from views.admin_analytics import _render_llm_tab, _render_overview_tab
from views import admin_agentic
from views import errors as errors_view


def render(days_int: int) -> None:
    tabs = st.tabs(["Overview", "LLM Performance", "Errors", "Agentic"])
    with tabs[0]:
        _render_overview_tab(days_int)
    with tabs[1]:
        _render_llm_tab(days_int)
    with tabs[2]:
        errors_view.render(days_int)
    with tabs[3]:
        admin_agentic.render(days_int)
