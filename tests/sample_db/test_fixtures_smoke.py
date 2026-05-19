def test_patients_fixture_loads(patients_csv):
    assert len(patients_csv) == 3
    assert "Id" in patients_csv.columns


def test_conditions_fixture_loads(conditions_csv):
    assert len(conditions_csv) == 3
    assert set(conditions_csv["CODE"].astype(str)) == {"44054006", "38341003", "55822004"}


def test_ctx_rng_is_deterministic(ctx):
    a = ctx.rng("test")
    b = ctx.rng("test")
    assert a is b  # same key returns same RNG
    c = ctx.rng("other")
    assert c is not a
