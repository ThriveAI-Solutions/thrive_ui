import pytest

pymilvus = pytest.importorskip("pymilvus")

from utils.milvus_vector import ThriveAI_Milvus


def _make_store(tmp_path, role: int):
    return ThriveAI_Milvus(
        user_role=role,
        config={
            "mode": "lite",
            "text_dim": 64,
            # Use a shared collection so role-based filtering applies across writers
            "collection_prefix": "unit_shared",
            "persist_path": str(tmp_path / "milvus_lite.db"),
        },
    )


def test_roundtrip_insert_retrieve_with_metadata(tmp_path):
    admin = _make_store(tmp_path, role=0)
    nurse = _make_store(tmp_path, role=2)
    patient = _make_store(tmp_path, role=3)

    # Write by different roles
    admin.add_question_sql("How many patients?", "SELECT COUNT(*) FROM patients;")
    nurse.add_ddl("CREATE TABLE nursing_notes (id INT, note TEXT);")
    patient.add_documentation("Patient can download their own records via portal.")

    # Admin should see everything (filter user_role >= 0)
    df_admin = admin.get_training_data()
    assert set(df_admin["training_data_type"]) == {"sql", "ddl", "documentation"}
    assert len(df_admin) == 3

    # Nurse should see nurse+patient items (filter user_role >= 2) → excludes admin item
    df_nurse = nurse.get_training_data()
    assert len(df_nurse) == 2
    assert set(df_nurse["training_data_type"]) == {"ddl", "documentation"}

    # Patient should only see patient items (filter user_role >= 3)
    df_patient = patient.get_training_data()
    assert len(df_patient) == 1
    assert set(df_patient["training_data_type"]) == {"documentation"}


def test_deterministic_ids_stable(tmp_path):
    store = _make_store(tmp_path, role=1)

    doc = "A small piece of documentation describing UNIQUE_TOKEN behavior."
    ddl = "CREATE TABLE t (id INT);"
    q, sql = "What is the count?", "SELECT COUNT(*) FROM t;"

    # Same content → same IDs
    d1 = store.add_documentation(doc)
    d2 = store.add_documentation(doc)
    assert d1 == d2

    k1 = store.add_ddl(ddl)
    k2 = store.add_ddl(ddl)
    assert k1 == k2

    s1 = store.add_question_sql(q, sql)
    s2 = store.add_question_sql(q, sql)
    assert s1 == s2
