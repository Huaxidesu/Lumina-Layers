"""Lumina Studio API — Health Check Router.
Lumina Studio API — 健康检查路由。

Provides a ``GET /api/health`` endpoint that returns service status,
version, and uptime information.
提供 ``GET /api/health`` 端点，返回服务状态、版本号和运行时间。
"""

import time

from fastapi import APIRouter

from api.schemas.responses import HealthResponse

router = APIRouter(prefix="/api", tags=["Health"])

_start_time: float = time.time()


@router.get("/health")
def health_check() -> HealthResponse:
    """Return service health status.
    返回服务健康状态信息。
    """
    return HealthResponse(
        status="ok",
        version="2.0",
        uptime_seconds=round(time.time() - _start_time, 2),
    )
