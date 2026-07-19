"""Patient 360 page (#246 UC1-1) — one-pass full-patient summary.

Runs the predefined domain-by-domain pipeline for the patient currently
selected in Chat and renders the master chart-review summary plus per-domain
detail. This is a predefined workflow (a button), not a chat turn.
"""

import streamlit as st

from agent.patient360.prompts import GENERATOR_VERSION
from agent.patient360.service import run_patient360

_STATUS_BADGE = {"done": "✅", "empty": "—", "failed": "⚠️"}

st.title("📋 Patient 360")

# Defense in depth (#320): the nav only lists this page for clinical roles, but
# also refuse to render for a PATIENT-role session reached by direct navigation.
if st.session_state.get("user_role") == 3:  # RoleTypeEnum.PATIENT
    st.error("Patient 360 is available to clinical staff only.")
    st.stop()

source_id = st.session_state.get("selected_patient_source_id")
display_name = st.session_state.get("selected_patient_display_name")
date_of_death = st.session_state.get("selected_patient_date_of_death")

if not source_id:
    st.info("No patient selected. Choose a patient in **Chat** first, then return here to generate their 360 summary.")
    st.stop()

st.caption(f"Selected patient: **{display_name or source_id}**")
if date_of_death:
    st.warning(
        f"Deceased patient — date of death: {date_of_death}. Historical chart only; do not present ongoing care."
    )

if st.button("Generate Patient 360", type="primary", width="stretch"):
    with st.spinner("Summarizing each clinical domain and composing the chart review…"):
        try:
            from agent.patient360.cache import SqlitePatient360Cache
            from orm.models import SessionLocal

            with SessionLocal() as session:
                result = run_patient360(
                    source_id,
                    enforce_consent=bool(st.secrets.get("security", {}).get("enforce_consent", False)),
                    user_role=st.session_state.get("user_role"),
                    cache=SqlitePatient360Cache(session),
                )
            st.session_state["_patient360_result"] = result.model_dump()
        except Exception as exc:  # surface, don't crash the page
            st.session_state["_patient360_result"] = None
            st.error(f"Patient 360 failed to generate: {exc}")

result = st.session_state.get("_patient360_result")
if result and result.get("source_id") == source_id:
    if result.get("master_summary"):
        st.markdown(result["master_summary"])
    else:
        st.info("No summarizable clinical data was found for this patient.")

    with st.expander("Per-domain detail"):
        for section in result.get("sections", []):
            badge = _STATUS_BADGE.get(section["status"], "")
            cached = " · cached" if section.get("cached") else ""
            st.markdown(f"**{badge} {section['name'].title()}** — {section['row_count']} rows{cached}")
            if section.get("narrative"):
                st.markdown(section["narrative"])
            elif section["status"] == "failed":
                st.caption(f"failed to summarize: {section.get('error')}")
    st.caption(f"Patient 360 generator v{result.get('generator_version', GENERATOR_VERSION)}")
