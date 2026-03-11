"""Unit tests for Converter Batch endpoint integration.

Validates:
- LUT name resolution failure returns 404 (Requirement 9)
- Partial failure continues processing remaining images (Requirement 9.3)
- All images succeed returns correct BatchResponse (Requirement 9.1, 9.2)
- Empty successful results still returns a valid ZIP (Requirement 9.3)
"""

from __future__ import annotations

import io
import os
import tempfile
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from api.app import app
from api.dependencies import get_file_registry, get_session_store
from api.file_registry import FileRegistry
from api.session_store import SessionStore

_test_store: SessionStore = SessionStore(ttl=1800)
_test_registry: FileRegistry = FileRegistry()

app.dependency_overrides[get_session_store] = lambda: _test_store
app.dependency_overrides[get_file_registry] = lambda: _test_registry

client: TestClient = TestClient(app)


def _make_test_image_buf(name: str = "test.png") -> tuple[str, io.BytesIO, str]:
    """Create a minimal PNG image upload tuple (filename, buf, content_type)."""
    img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return (name, buf, "image/png")


def _make_fake_3mf(suffix: str = ".3mf") -> str:
    """Create a temporary fake 3MF file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.write(fd, b"fake-3mf-content")
    os.close(fd)
    return path


# =========================================================================
# 1. LUT not found returns 404 - Requirement 9
# =========================================================================


class TestBatchLutNotFound:
    """Verify unresolvable lut_name returns HTTP 404."""

    def test_batch_unknown_lut_returns_404(self) -> None:
        with patch(
            "api.routers.converter.LUTManager.get_lut_path",
            return_value=None,
        ):
            response = client.post(
                "/api/convert/batch",
                files=[("images", _make_test_image_buf("a.png"))],
                data={"lut_name": "nonexistent_lut"},
            )
        assert response.status_code == 404
        assert "LUT" in response.json()["detail"]


# =========================================================================
# 2. Partial failure continues processing - Requirement 9.3
# =========================================================================


class TestBatchPartialFailure:
    """Verify that if some images fail, remaining images still process."""

    def test_batch_partial_failure_continues(self) -> None:
        fake_3mf = _make_fake_3mf()
        call_count = 0

        def _mock_convert(**kwargs: object) -> tuple:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated failure")
            return (fake_3mf, None, None, "OK")

        with patch(
            "api.routers.converter.LUTManager.get_lut_path",
            return_value="/tmp/fake.npy",
        ), patch(
            "api.routers.converter.upload_to_tempfile",
            return_value="/tmp/uploaded.png",
        ), patch(
            "api.routers.converter.convert_image_to_3d",
            side_effect=_mock_convert,
        ):
            response = client.post(
                "/api/convert/batch",
                files=[
                    ("images", _make_test_image_buf("fail.png")),
                    ("images", _make_test_image_buf("ok.png")),
                ],
                data={"lut_name": "test_lut"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert len(body["results"]) == 2

        # First image failed
        assert body["results"][0]["filename"] == "fail.png"
        assert body["results"][0]["status"] == "failed"
        assert body["results"][0]["error"] is not None

        # Second image succeeded
        assert body["results"][1]["filename"] == "ok.png"
        assert body["results"][1]["status"] == "success"

        # download_url present
        assert body["download_url"].startswith("/api/files/")


# =========================================================================
# 3. All images succeed - Requirement 9.1, 9.2
# =========================================================================


class TestBatchAllSuccess:
    """Verify all images succeed returns correct BatchResponse."""

    def test_batch_all_success(self) -> None:
        fake_3mf = _make_fake_3mf()

        with patch(
            "api.routers.converter.LUTManager.get_lut_path",
            return_value="/tmp/fake.npy",
        ), patch(
            "api.routers.converter.upload_to_tempfile",
            return_value="/tmp/uploaded.png",
        ), patch(
            "api.routers.converter.convert_image_to_3d",
            return_value=(fake_3mf, None, None, "OK"),
        ):
            response = client.post(
                "/api/convert/batch",
                files=[
                    ("images", _make_test_image_buf("img1.png")),
                    ("images", _make_test_image_buf("img2.png")),
                ],
                data={"lut_name": "test_lut"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "2/2" in body["message"]
        assert len(body["results"]) == 2
        assert all(r["status"] == "success" for r in body["results"])
        assert body["download_url"].startswith("/api/files/")


# =========================================================================
# 4. All images fail returns status "failed" - Requirement 9.3
# =========================================================================


class TestBatchAllFailed:
    """Verify all images failing returns status 'failed' with valid ZIP."""

    def test_batch_all_fail(self) -> None:
        with patch(
            "api.routers.converter.LUTManager.get_lut_path",
            return_value="/tmp/fake.npy",
        ), patch(
            "api.routers.converter.upload_to_tempfile",
            return_value="/tmp/uploaded.png",
        ), patch(
            "api.routers.converter.convert_image_to_3d",
            side_effect=RuntimeError("boom"),
        ):
            response = client.post(
                "/api/convert/batch",
                files=[("images", _make_test_image_buf("bad.png"))],
                data={"lut_name": "test_lut"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "failed"
        assert "0/1" in body["message"]
        assert body["results"][0]["status"] == "failed"
        assert body["download_url"].startswith("/api/files/")
