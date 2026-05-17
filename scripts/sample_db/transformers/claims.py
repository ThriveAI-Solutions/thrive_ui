# scripts/sample_db/transformers/claims.py
"""Synthea claims.csv + procedures.csv → 4 federated_claims_*_v tables.

Real warehouse claims are 100% standardized (ICD-10 / HCPCS); no noise.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from scripts.sample_db.crosswalks.loader import snomed_to_icd10
from scripts.sample_db.transformers.base import TransformContext


def transform_claims(
    claims: pd.DataFrame,
    encounters: pd.DataFrame,
    procedures: pd.DataFrame,
    source_map: dict[str, str],
    ctx: TransformContext,
) -> None:
    cw = snomed_to_icd10()
    diag, proc, facility, summary = [], [], [], []

    enc_by_id = encounters.set_index("Id") if not encounters.empty else None

    for line_seq, (_, c) in enumerate(claims.iterrows(), start=1):
        patient = c["PATIENTID"]
        source_id = source_map.get(patient)
        if source_id is None:
            continue
        claim_id = c["Id"]
        line_id = f"{claim_id}-L{line_seq:03d}"
        svc_date_raw = c.get("SERVICEDATE")
        svc_date = pd.to_datetime(svc_date_raw, utc=True).date() if pd.notna(svc_date_raw) else dt.date.today()
        moyr = svc_date.replace(day=1)

        # Diagnosis detail (one row per diagnosis code on the claim).
        for col in [f"DIAGNOSIS{i}" for i in range(1, 9)]:
            snomed = c.get(col)
            if pd.notna(snomed) and snomed != "":
                icd10 = cw.get(str(snomed), str(snomed))
                diag.append(
                    {
                        "claim_line_identifier": line_id,
                        "icd_diagnosis_code": icd10,
                        "icd_type": "ICD-10",
                        "primary_flag": 1 if col == "DIAGNOSIS1" else 0,
                        "diagnosis_date": svc_date,
                        "diagnosis_sequence_number": int(col.replace("DIAGNOSIS", "")),
                        "source_file_moyr": moyr,
                        "source_file_name": f"claims_{moyr:%Y_%m}.csv",
                        "source_format": "ICD",
                        "source_name": "Synthetic",
                    }
                )

        # Procedure detail — pull from encounters' linked procedures if any.
        if enc_by_id is not None and c.get("APPOINTMENTID") in enc_by_id.index:
            enc_procs = procedures[procedures["ENCOUNTER"] == c["APPOINTMENTID"]]
            for i, (_, p) in enumerate(enc_procs.iterrows(), start=1):
                proc.append(
                    {
                        "claim_line_identifier": line_id,
                        "icd_procedure_code": str(p["CODE"]),
                        "icd_type": "ICD-10-PCS",
                        "primary_flag": 1 if i == 1 else 0,
                        "procedure_date": pd.to_datetime(p["START"], utc=True).date(),
                        "procedure_sequence_number": i,
                        "source_file_moyr": moyr,
                        "source_file_name": f"claims_{moyr:%Y_%m}.csv",
                        "source_format": "ICD",
                        "source_name": "Synthetic",
                    }
                )

        # Facility detail (one row per claim).
        facility.append(
            {
                "claim_line_identifier": line_id,
                "hcpcs": "99213",  # office visit, established patient
                "service_date": svc_date,
                "source_file_moyr": moyr,
                "source_file_name": f"claims_{moyr:%Y_%m}.csv",
                "source_format": "HCPCS",
                "source_name": "Synthetic",
            }
        )

        # Claim summary.
        summary.append(
            {
                "claim_line_identifier": line_id,
                "service_begin_date": svc_date,
                "service_end_date": svc_date,
                "source_file_moyr": moyr,
                "source_file_name": f"claims_{moyr:%Y_%m}.csv",
                "source_name": "Synthetic",
                "source_format": "ICD",
            }
        )

    ctx.add_rows("dw.federated_claims_icd_diagnosis_detail_v", diag)
    ctx.add_rows("dw.federated_claims_icd_procedure_detail_v", proc)
    ctx.add_rows("dw.federated_claims_medical_facility_detail_v", facility)
    ctx.add_rows("dw.federated_claims_summary_v", summary)
