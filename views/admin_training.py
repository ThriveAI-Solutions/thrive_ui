"""Admin → Training sub-tab (Epic #144).

Relocated from views/user.py tab2. Post-Epic #126 library-first layout.
The Manage Users → Activity deep-link button writes the
audit_trail_pref_user_id session-state key and navigates via st.switch_page.
"""

import streamlit as st
from pandas import DataFrame

from utils.chat_bot_helper import get_vn
from utils.vanna_calls import auto_generate_sql_pairs, refresh_stats, train_all

from views._admin_helpers import (
    confirm_destructive,
    delete_all_training,
    export_training_data_to_csv,
    import_training_data_from_csv,
    pop_train,
)

# delete_all_messages used by the Danger Zone — it's per-user; lives in orm.functions
from orm.functions import delete_all_messages


def render(days_int: int | None = None) -> None:
    """Render the Training Data admin sub-tab."""
    # Section 1 — Training Data Library
    st.subheader("Training Data Library")
    df = get_vn().get_training_data()
    if isinstance(df, DataFrame) and not df.empty:
        display_df = df[["id", "training_data_type", "question", "content"]].copy()
        display_df.columns = ["ID", "Type", "Question", "Content"]

        filter_col1, filter_col2 = st.columns([0.6, 0.4])
        with filter_col1:
            search_query = st.text_input(
                "Search training data",
                placeholder="Search by question or content...",
                key="training_search",
            )
        with filter_col2:
            available_types = sorted(display_df["Type"].dropna().unique().tolist())
            type_filter = st.multiselect("Filter by type", options=available_types, key="training_type_filter")

        if search_query:
            mask = display_df["Question"].astype(str).str.contains(search_query, case=False, na=False) | display_df[
                "Content"
            ].astype(str).str.contains(search_query, case=False, na=False)
            display_df = display_df[mask]
        if type_filter:
            display_df = display_df[display_df["Type"].isin(type_filter)]

        if st.session_state.cookies.get("role_name") == "Admin":
            display_df.insert(0, "Select", False)

            edited_df = st.data_editor(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Select": st.column_config.CheckboxColumn(
                        "Delete?",
                        help="Select rows to delete",
                        default=False,
                        width="small",
                    ),
                    "ID": st.column_config.TextColumn(
                        "ID",
                        width="small",
                        disabled=True,
                    ),
                    "Type": st.column_config.TextColumn(
                        "Type",
                        width="small",
                        disabled=True,
                    ),
                    "Question": st.column_config.TextColumn(
                        "Question",
                        width="medium",
                        disabled=True,
                    ),
                    "Content": st.column_config.TextColumn(
                        "Content",
                        width="large",
                        disabled=True,
                    ),
                },
                num_rows="fixed",
                key="training_data_grid",
            )

            selected_rows = edited_df[edited_df["Select"]]
            if not selected_rows.empty:
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button(
                        f"🗑️ Delete {len(selected_rows)} Selected",
                        type="primary",
                        help="Delete all selected training data entries",
                    ):
                        vn = get_vn()
                        for _, row in selected_rows.iterrows():
                            vn.remove_from_training(row["ID"])
                        st.toast(f"Deleted {len(selected_rows)} training data entries!")
                        st.rerun()
                with col2:
                    st.caption(f"Selected {len(selected_rows)} of {len(display_df)} entries for deletion")
        else:
            st.dataframe(
                display_df[["ID", "Type", "Question", "Content"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ID": st.column_config.TextColumn("ID", width="small"),
                    "Type": st.column_config.TextColumn("Type", width="small"),
                    "Question": st.column_config.TextColumn("Question", width="medium"),
                    "Content": st.column_config.TextColumn("Content", width="large"),
                },
            )

        st.caption(f"Total: {len(df)} training data entries")
    else:
        st.info("No training data available.")

    # Section 2 — Add Training Data
    st.divider()
    st.subheader("Add Training Data")
    add_col1, add_col2 = st.columns(2)
    with add_col1:
        if st.button("Add SQL", key="add_sql_btn"):
            pop_train("sql")
    with add_col2:
        if st.button("Add Documentation", key="add_doc_btn"):
            pop_train("documentation")

    # Section 3 — Bulk Operations
    st.divider()
    st.subheader("Bulk Operations")

    st.markdown("**Import CSV**")
    uploaded_file = st.file_uploader(
        "Choose a CSV file to upload training data",
        type=["csv"],
        help="Upload a CSV file with columns: training_data_type, question, content",
    )
    if uploaded_file is not None:
        st.info(f"📁 File: {uploaded_file.name} ({uploaded_file.size} bytes)")
        imp_col1, imp_col2 = st.columns(2)
        with imp_col1:
            if st.button("📤 Import Training Data", type="primary"):
                with st.spinner("Importing training data..."):
                    import_training_data_from_csv(uploaded_file)
        with imp_col2:
            if st.button("❌ Cancel"):
                st.rerun()

    st.markdown("**Export CSV**")
    if st.button("Export CSV", key="export_csv_btn"):
        export_training_data_to_csv()

    st.markdown("**Auto-Generate SQL Pairs**")
    st.caption("Use the connected LLM to generate, validate, and train SQL question-answer pairs automatically.")
    gen_col1, gen_col2, gen_spacer = st.columns((0.20, 0.20, 0.60))
    with gen_col1:
        pair_count = st.number_input(
            "Number of pairs",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
            key="auto_gen_pair_count",
        )
    with gen_col2:
        st.write("")  # vertical spacing
        st.write("")
        if st.button("Generate", type="primary", key="auto_gen_btn"):
            with st.status("Generating SQL training pairs...", expanded=True) as gen_status:
                gen_results = auto_generate_sql_pairs(count=int(pair_count))
                error_msg = gen_results.get("error")
                if error_msg:
                    gen_status.update(
                        label=f"Generation failed: {error_msg}",
                        state="error",
                    )
                    st.error(error_msg)
                else:
                    passed = gen_results.get("passed", 0)
                    failed = gen_results.get("failed", 0)
                    if passed > 0:
                        gen_status.update(
                            label=f"Generated {passed} pair(s) ({failed} failed)",
                            state="complete",
                        )
                        st.success(f"Successfully trained {passed} SQL pair(s).")
                    else:
                        gen_status.update(
                            label=f"Generation complete: 0 passed, {failed} failed",
                            state="error",
                        )
                        st.warning("No valid pairs were generated. Check LLM connectivity and training data.")
                failed_details = [d for d in gen_results.get("details", []) if d["status"] == "failed"]
                if failed_details:
                    with st.expander(f"Failed pairs ({len(failed_details)})"):
                        for d in failed_details:
                            st.text(f"Pair {d['index']}: {d.get('reason', 'Unknown')}")

    # Section 4 — Training Pipeline
    st.divider()
    st.subheader("Training Pipeline")
    train_col1, train_col2, _train_spacer = st.columns((0.20, 0.20, 0.60))
    with train_col1:
        st.button(
            "Train All",
            type="primary",
            on_click=train_all,
            help="Full pipeline: DDL, Schema Plan, Auto-Enhance, AI Docs",
        )
    with train_col2:
        st.button(
            "Refresh Stats",
            on_click=refresh_stats,
            help="Re-run column statistics only (use when data changed but schema is the same)",
        )

    # Section 5 — Danger Zone
    st.divider()
    st.subheader("Danger Zone")
    if st.button("Remove All Training Data", key="danger_remove_all_btn"):
        confirm_destructive(
            body_md=(
                "Permanently removes every training-data row visible to you "
                "(role-filtered). Schema embeddings will need to be "
                "regenerated via **Train All**."
            ),
            token="DELETE",
            on_confirm=delete_all_training,
            button_label="Remove All Training Data",
        )
    if st.button("Delete My Chat History", key="danger_delete_chat_btn"):
        confirm_destructive(
            body_md=(
                "Permanently removes **your** chat history from this account. Other users' messages are not affected."
            ),
            token="DELETE",
            on_confirm=delete_all_messages,
            button_label="Delete My Chat History",
        )
