"""Patient 360 — predefined full-patient summary workflow (#246 UC1-1).

A deterministic, domain-by-domain pipeline (NOT one big summarization prompt):
each clinical domain is fetched and summarized on its own, in a fixed clinical
order, then a synthesis pass composes a master chart-review summary. The
``visits`` section is a UNION of the encounters and ADT feeds — the same design
Chiron needed for the encounters/hospitalization questions (#247 Q7/Q10) that a
single prompt over one table gets wrong.

``pipeline.generate_patient360`` is the entry point; the LLM call is injected as
a ``summarizer`` callable so the deterministic assembly (domain order, the
visits union, empty/failed handling, synthesis input) is fully testable without
a live model.
"""
