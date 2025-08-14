from unittest.mock import patch

import pytest

pymilvus = pytest.importorskip("pymilvus")

from utils.milvus_vector import ThriveAI_Milvus
from utils.vanna_calls import UserContext, VannaService


@pytest.mark.milvus
@pytest.mark.skipif(pytest.importorskip("pymilvus", reason="pymilvus not installed") is None, reason="pymilvus missing")
def test_vannaservice_with_milvus_rbac(tmp_path):
    admin_role, nurse_role, patient_role = 0, 2, 3

    milvus_cfg = {
        "mode": "lite",
        "persist_path": str(tmp_path / "milvus_app_lite.db"),
        "text_dim": 64,
        "collection_prefix": "app_rbac",
    }

    # Create services for each role without running _setup_vanna
    with patch.object(VannaService, "_setup_vanna"):
        admin_svc = VannaService(UserContext(user_id="u_admin", user_role=admin_role), config={"security": {}})
        nurse_svc = VannaService(UserContext(user_id="u_nurse", user_role=nurse_role), config={"security": {}})
        patient_svc = VannaService(UserContext(user_id="u_patient", user_role=patient_role), config={"security": {}})

    # Attach Milvus-backed stores
    admin_svc.vn = ThriveAI_Milvus(user_role=admin_role, config=milvus_cfg)
    nurse_svc.vn = ThriveAI_Milvus(user_role=nurse_role, config=milvus_cfg)
    patient_svc.vn = ThriveAI_Milvus(user_role=patient_role, config=milvus_cfg)

    # Seed via app-level train() API to ensure metadata propagation
    assert admin_svc.train(question="How many patients?", sql="SELECT COUNT(*) FROM patients;")
    assert nurse_svc.train(ddl="CREATE TABLE nursing_notes (id INT, note TEXT);")
    assert patient_svc.train(documentation="Patient can download their own records via portal.")

    # Admin should see all three
    df_admin = admin_svc.get_training_data()
    assert set(df_admin["training_data_type"]) == {"sql", "ddl", "documentation"}
    assert len(df_admin) == 3

    # Nurse should see nurse+patient (2)
    df_nurse = nurse_svc.get_training_data()
    assert len(df_nurse) == 2
    assert set(df_nurse["training_data_type"]) == {"ddl", "documentation"}

    # Patient should see only patient (1)
    df_patient = patient_svc.get_training_data()
    assert len(df_patient) == 1
    assert set(df_patient["training_data_type"]) == {"documentation"}


