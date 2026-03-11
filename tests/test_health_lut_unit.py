"""Unit tests for Health and LUT list endpoints.

Validates:
- GET /api/health returns correct structure and values (Requirement 11)
- GET /api/lut/list returns sorted LUT dictionary (Requirement 10)
"""

from fastapi.testclient import TestClient

from api.app import app

client: TestClient = TestClient(app)


# =========================================================================
# 1. Health endpoint (GET /api/health) - Requirement 11
# =========================================================================


class TestHealthEndpoint:
    """Verify the health check endpoint returns correct structure and values."""

    def test_health_returns_200(self) -> None:
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_contains_required_fields(self) -> None:
        data: dict = client.get("/api/health").json()
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data

    def test_health_status_is_ok(self) -> None:
        data: dict = client.get("/api/health").json()
        assert data["status"] == "ok"

    def test_health_version_is_2_0(self) -> None:
        data: dict = client.get("/api/health").json()
        assert data["version"] == "2.0"

    def test_health_uptime_is_non_negative_float(self) -> None:
        data: dict = client.get("/api/health").json()
        uptime = data["uptime_seconds"]
        assert isinstance(uptime, float)
        assert uptime >= 0.0


# =========================================================================
# 2. LUT list endpoint (GET /api/lut/list) - Requirement 10
# =========================================================================


class TestLUTListEndpoint:
    """Verify the LUT list endpoint returns correct structure and sorted keys."""

    def test_lut_list_returns_200(self) -> None:
        response = client.get("/api/lut/list")
        assert response.status_code == 200

    def test_lut_list_contains_required_fields(self) -> None:
        data: dict = client.get("/api/lut/list").json()
        assert "luts" in data

    def test_lut_list_luts_is_list(self) -> None:
        data: dict = client.get("/api/lut/list").json()
        assert isinstance(data["luts"], list)

    def test_lut_list_items_have_required_fields(self) -> None:
        data: dict = client.get("/api/lut/list").json()
        for item in data["luts"]:
            assert "name" in item
            assert "color_mode" in item
            assert "path" in item

    def test_lut_list_names_sorted_alphabetically(self) -> None:
        data: dict = client.get("/api/lut/list").json()
        names = [item["name"] for item in data["luts"]]
        assert names == sorted(names)
