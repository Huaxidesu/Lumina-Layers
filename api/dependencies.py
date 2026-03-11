"""Lumina Studio API — Dependency Injection.
Lumina Studio API — 依赖注入模块。

Global singletons and FastAPI dependency functions.
Separated from app.py to avoid circular imports between
app and router modules.
全局单例和 FastAPI 依赖注入函数。
从 app.py 分离以避免 app 与 router 模块之间的循环导入。
"""

from api.file_registry import FileRegistry
from api.session_store import SessionStore

# Global singletons
session_store: SessionStore = SessionStore(ttl=1800)
file_registry: FileRegistry = FileRegistry()


def get_session_store() -> SessionStore:
    """FastAPI dependency: return global SessionStore."""
    return session_store


def get_file_registry() -> FileRegistry:
    """FastAPI dependency: return global FileRegistry."""
    return file_registry
