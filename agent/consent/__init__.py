"""Consent enforcement for patient-level disclosure (#244 CON-4).

Two independent controls:

- ``gate.ConsentGate`` — fail-closed per-patient consent check. A non-consented
  patient collapses to the same "not found" response as a nonexistent or
  ambiguous one, so consent status is never inferable by comparing responses.
- ``suppression`` — small-cell suppression for the aggregate/cohort exemption.
  HeL confirmed aggregates need no consent filter, but low counts are a
  re-identification risk, so count-like cells below a threshold render as a
  fixed label before the model, any cache, or the UI sees the number.
"""
