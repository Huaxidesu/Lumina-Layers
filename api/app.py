"""Lumina Studio API — Application Factory.
Lumina Studio API — 应用工厂模块。

Provides a ``create_app()`` factory function that builds a fully-configured
FastAPI instance with CORS middleware and all domain routers registered.
提供 ``create_app()`` 工厂函数，构建配置完整的 FastAPI 实例，
包含 CORS 中间件和所有领域路由的注册。
"""

import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import (
    file_registry,
    get_file_registry,
    get_session_store,
    session_store,
)
from api.file_bridge import file_to_response
from api.routers import (
    calibration_router,
    converter_router,
    extractor_router,
    five_color_router,
    health_router,
    lut_router,
    slicer_router,
    system_router,
)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    创建并配置 FastAPI 应用实例。

    Returns:
        FastAPI: A fully-configured application instance with CORS middleware
            and all domain routers registered.
            配置完整的应用实例，已注册 CORS 中间件和所有领域路由。
    """
    app = FastAPI(title="Lumina Studio API", version="2.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(converter_router)
    app.include_router(extractor_router)
    app.include_router(calibration_router)
    app.include_router(five_color_router)
    app.include_router(health_router)
    app.include_router(lut_router)
    app.include_router(slicer_router)
    app.include_router(system_router)

    @app.get("/api/files/{file_id}")
    def serve_file(file_id: str):
        """Serve a registered file by file_id."""
        result = file_registry.resolve(file_id)
        if result is None:
            raise HTTPException(status_code=404, detail="File not found or expired")
        path, filename = result
        return file_to_response(path, filename)

    @app.on_event("startup")
    async def start_session_cleanup() -> None:
        async def _cleanup_loop() -> None:
            while True:
                await asyncio.sleep(60)
                count = session_store.cleanup_expired()
                if count > 0:
                    print(f"[SESSION] Cleaned up {count} expired sessions")
        asyncio.create_task(_cleanup_loop())

    return app


app: FastAPI = create_app()
