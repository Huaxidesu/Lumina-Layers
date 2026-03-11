"""Unit tests for Converter Generate endpoint integration.

Validates:
- Generate endpoint passes ALL advanced parameters to generate_final_model() (Requirement 8.4)
- Session not found returns 404 (Requirement 8)
- No preview_cache returns 409 (Requirement 8)
- Missing image_path returns 409 (Requirement 8)
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_file_registry, get_session_store
from api.file_registry import FileRegistry
from api.session_store import SessionStore
from config import ModelingMode as CoreModelingMode

_test_store: SessionStore = SessionStore(ttl=1800)
_test_registry: FileRegistry = FileRegistry()

app.dependency_overrides[get_session_store] = lambda: _test_store
app.dependency_overrides[get_file_registry] = lambda: _test_registry

client: TestClient = TestClient(app)


def _create_session_with_preview_and_files(store: SessionStore) -> str:
    """Create a session pre-populated with preview_cache and file paths."""
    sid: str = store.create()
    store.put(sid, "preview_cache", {"some": "data"})
    store.put(sid, "image_path", "/tmp/test_image.png")
    store.put(sid, "lut_path", "/tmp/test_lut.npy")
    store.put(sid, "replacement_regions", [])
    store.put(sid, "free_color_set", set())
    return sid


# =========================================================================
# 1. Session not found returns 404 - Requirement 8
# =========================================================================


class TestGenerateSessionNotFound:
    """Verify unknown session_id returns HTTP 404."""

    def test_generate_unknown_session_returns_404(self) -> None:
        payload = {
            "session_id": "nonexistent-session-id",
            "params": {"lut_name": "test_lut"},
        }
        response = client.post("/api/convert/generate", json=payload)
        assert response.status_code == 404
        assert "Session" in response.json()["detail"]


# =========================================================================
# 2. No preview_cache returns 409 - Requirement 8
# =========================================================================


class TestGenerateNoPreviewCache:
    """Verify missing preview_cache returns HTTP 409."""

    def test_generate_no_preview_cache_returns_409(self) -> None:
        sid: str = _test_store.create()
        # Session exists but has no preview_cache
        payload = {
            "session_id": sid,
            "params": {"lut_name": "test_lut"},
        }
        response = client.post("/api/convert/generate", json=payload)
        assert response.status_code == 409
        assert "preview" in response.json()["detail"].lower()


# =========================================================================
# 3. Missing image_path returns 409 - Requirement 8
# =========================================================================


class TestGenerateMissingImagePath:
    """Verify missing or non-existent image_path returns HTTP 409."""

    def test_generate_missing_image_path_returns_409(self) -> None:
        sid: str = _test_store.create()
        _test_store.put(sid, "preview_cache", {"some": "data"})
        # No image_path stored
        payload = {
            "session_id": sid,
            "params": {"lut_name": "test_lut"},
        }
        response = client.post("/api/convert/generate", json=payload)
        assert response.status_code == 409
        assert "Image file missing" in response.json()["detail"]

    def test_generate_nonexistent_image_file_returns_409(self) -> None:
        sid: str = _test_store.create()
        _test_store.put(sid, "preview_cache", {"some": "data"})
        _test_store.put(sid, "image_path", "/tmp/does_not_exist_12345.png")
        _test_store.put(sid, "lut_path", "/tmp/test_lut.npy")
        payload = {
            "session_id": sid,
            "params": {"lut_name": "test_lut"},
        }
        response = client.post("/api/convert/generate", json=payload)
        assert response.status_code == 409


# =========================================================================
# 4. Parameter completeness - Requirement 8.4
# =========================================================================


class TestGenerateParameterCompleteness:
    """Verify all advanced parameters are correctly passed to generate_final_model()."""

    def test_generate_passes_all_advanced_params(self) -> None:
        sid: str = _create_session_with_preview_and_files(_test_store)

        payload = {
            "session_id": sid,
            "params": {
                "lut_name": "test_lut",
                "target_width_mm": 80.0,
                "spacer_thick": 1.5,
                "structure_mode": "Single-sided",
                "auto_bg": True,
                "bg_tol": 60,
                "color_mode": "4-Color",
                "modeling_mode": "high-fidelity",
                "quantize_colors": 64,
                "enable_cleanup": False,
                "separate_backing": True,
                "add_loop": True,
                "loop_width": 5.0,
                "loop_length": 10.0,
                "loop_hole": 3.0,
                "loop_pos": [50.0, 50.0],
                "enable_relief": True,
                "color_height_map": {"#ff0000": 2.0},
                "heightmap_max_height": 8.0,
                "enable_outline": True,
                "outline_width": 3.0,
                "enable_cloisonne": True,
                "wire_width_mm": 0.6,
                "wire_height_mm": 0.5,
                "enable_coating": True,
                "coating_height_mm": 0.1,
            },
        }

        mock_result = ("/tmp/out.3mf", "/tmp/out.glb", None, "OK", None)

        with patch(
            "api.routers.converter.generate_final_model",
            return_value=mock_result,
        ) as mock_gen, patch("os.path.exists", return_value=True):
            response = client.post("/api/convert/generate", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["download_url"].startswith("/api/files/")
        assert body["preview_3d_url"].startswith("/api/files/")

        # Verify generate_final_model was called exactly once
        mock_gen.assert_called_once()

        # Extract keyword arguments from the mock call
        call_kwargs = mock_gen.call_args.kwargs

        # Verify file paths from session
        assert call_kwargs["image_path"] == "/tmp/test_image.png"
        assert call_kwargs["lut_path"] == "/tmp/test_lut.npy"

        # Verify basic parameters
        assert call_kwargs["target_width_mm"] == 80.0
        assert call_kwargs["spacer_thick"] == 1.5
        assert call_kwargs["structure_mode"] == "Single-sided"
        assert call_kwargs["auto_bg"] is True
        assert call_kwargs["bg_tol"] == 60
        assert call_kwargs["color_mode"] == "4-Color"
        assert call_kwargs["quantize_colors"] == 64
        assert call_kwargs["enable_cleanup"] is False
        assert call_kwargs["separate_backing"] is True

        # Verify modeling_mode is converted to core enum
        assert call_kwargs["modeling_mode"] == CoreModelingMode("high-fidelity")

        # Verify keychain loop parameters
        assert call_kwargs["add_loop"] is True
        assert call_kwargs["loop_width"] == 5.0
        assert call_kwargs["loop_length"] == 10.0
        assert call_kwargs["loop_hole"] == 3.0
        assert call_kwargs["loop_pos"] == (50.0, 50.0)

        # Verify relief parameters
        assert call_kwargs["enable_relief"] is True
        assert call_kwargs["color_height_map"] == {"#ff0000": 2.0}
        assert call_kwargs["heightmap_max_height"] == 8.0

        # Verify outline parameters
        assert call_kwargs["enable_outline"] is True
        assert call_kwargs["outline_width"] == 3.0

        # Verify cloisonne parameters
        assert call_kwargs["enable_cloisonne"] is True
        assert call_kwargs["wire_width_mm"] == 0.6
        assert call_kwargs["wire_height_mm"] == 0.5

        # Verify coating parameters
        assert call_kwargs["enable_coating"] is True
        assert call_kwargs["coating_height_mm"] == 0.1

        # Verify session-derived parameters
        # Empty list/set from session becomes None via `or None` in router
        assert call_kwargs["replacement_regions"] is None
        assert call_kwargs["free_color_set"] is None

    def test_generate_passes_replacement_regions_from_request(self) -> None:
        """Verify replacement_regions from request body override session data."""
        sid: str = _create_session_with_preview_and_files(_test_store)

        payload = {
            "session_id": sid,
            "params": {
                "lut_name": "test_lut",
                "replacement_regions": [
                    {
                        "quantized_hex": "#ff0000",
                        "matched_hex": "#ee0000",
                        "replacement_hex": "#00ff00",
                    }
                ],
            },
        }

        mock_result = ("/tmp/out.3mf", "/tmp/out.glb", None, "OK", None)

        with patch(
            "api.routers.converter.generate_final_model",
            return_value=mock_result,
        ) as mock_gen, patch("os.path.exists", return_value=True):
            response = client.post("/api/convert/generate", json=payload)

        assert response.status_code == 200
        call_kwargs = mock_gen.call_args.kwargs
        regions = call_kwargs["replacement_regions"]
        assert len(regions) == 1
        assert regions[0]["quantized_hex"] == "#ff0000"
        assert regions[0]["matched_hex"] == "#ee0000"
        assert regions[0]["replacement_hex"] == "#00ff00"
