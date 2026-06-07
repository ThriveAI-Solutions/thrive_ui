"""Regression tests for auto_enhance_schema().

Issue #111: `auto_enhance_schema` called the long-removed `get_vn()` helper
instead of `VannaService.from_streamlit_session()`, so train_all step 3 always
raised NameError and was silently swallowed. These tests pin the call site to
the correct factory so the regression cannot reappear.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


MOCK_SECRETS = {
    "ai_keys": {"ollama_model": "test_model"},
    "rag_model": {"chroma_path": "/tmp/test_chroma"},
    "postgres": {
        "host": "localhost",
        "port": 5432,
        "database": "test_db",
        "user": "test_user",
        "password": "test_pass",
        "schema_name": "public",
        "object_type": "tables",
    },
}


class TestAutoEnhanceSchemaInvocation:
    """Pin the call site at utils/vanna_calls.py:auto_enhance_schema()."""

    @pytest.fixture(autouse=True)
    def _patch_streamlit(self):
        status_cm = MagicMock()
        status_cm.__enter__ = MagicMock(return_value=status_cm)
        status_cm.__exit__ = MagicMock(return_value=False)
        with (
            patch("utils.vanna_calls.st.toast"),
            patch("utils.vanna_calls.st.write"),
            patch("utils.vanna_calls.st.success"),
            patch("utils.vanna_calls.st.warning"),
            patch("utils.vanna_calls.st.error"),
            patch("utils.vanna_calls.st.status", return_value=status_cm),
        ):
            yield

    def test_uses_vannaservice_factory_not_legacy_get_vn(self):
        """Regression for #111: the function must call
        ``VannaService.from_streamlit_session()`` rather than the removed
        ``get_vn()`` helper.
        """
        from utils import vanna_calls

        assert not hasattr(vanna_calls, "get_vn"), (
            "get_vn was removed from the module; auto_enhance_schema must not reference it (issue #111)."
        )

        mock_service = MagicMock()
        mock_service.get_training_data.return_value = pd.DataFrame()

        with (
            patch("utils.vanna_calls.st.secrets", new=MOCK_SECRETS),
            patch(
                "utils.vanna_calls.VannaService.from_streamlit_session",
                return_value=mock_service,
            ) as mock_factory,
            patch(
                "utils.vanna_calls.read_forbidden_from_json",
                return_value=([], [], ""),
            ),
        ):
            # Should not raise NameError. The function may early-exit because
            # of empty mocks; that's fine — we only pin the factory call.
            try:
                vanna_calls.auto_enhance_schema(clear_existing=True)
            except NameError as e:
                pytest.fail(f"auto_enhance_schema raised NameError (issue #111 regression): {e}")
            except Exception:
                # Any other exception is acceptable for this regression test —
                # we are only proving that the call site no longer references
                # an undefined name.
                pass

        mock_factory.assert_called_once()

    def test_clear_existing_path_does_not_raise_nameerror(self):
        """The ``clear_existing=True`` branch was where the NameError lived.

        Confirm the path executes far enough to call get_training_data on the
        VannaService instance returned by from_streamlit_session — proving the
        first statement of the function resolved a real callable, not the
        deleted ``get_vn`` symbol.
        """
        from utils import vanna_calls

        mock_service = MagicMock()
        mock_service.get_training_data.return_value = pd.DataFrame()

        with (
            patch("utils.vanna_calls.st.secrets", new=MOCK_SECRETS),
            patch(
                "utils.vanna_calls.VannaService.from_streamlit_session",
                return_value=mock_service,
            ),
            patch(
                "utils.vanna_calls.read_forbidden_from_json",
                return_value=([], [], ""),
            ),
        ):
            try:
                vanna_calls.auto_enhance_schema(clear_existing=True)
            except NameError as e:
                pytest.fail(f"auto_enhance_schema raised NameError on clear_existing path: {e}")
            except Exception:
                pass

        mock_service.get_training_data.assert_called()
