import os
import pathlib
from sqlalchemy.ext.asyncio import create_async_engine


def make_engine(sqlite_path: str):
    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)

    return create_async_engine(
        f"sqlite+aiosqlite:///{sqlite_path}",
        future=True,
        echo=False,
    )


async def init_db(engine):
    base_dir = pathlib.Path(__file__).resolve().parent
    schema_path = base_dir / "schema.sql"

    if not schema_path.exists():
        raise RuntimeError(f"schema.sql not found at {schema_path}")

    schema = schema_path.read_text()

    # ✅ Clean split for SQLite (safe enough for CREATE TABLE only schemas)
    statements = []
    buffer = ""

    for line in schema.splitlines():
        line = line.strip()

        # skip empty lines and comments
        if not line or line.startswith("--"):
            continue

        buffer += line + " "

        if line.endswith(";"):
            statements.append(buffer.strip())
            buffer = ""

    async with engine.begin() as conn:
        for stmt in statements:
            await conn.exec_driver_sql(stmt)

    # ── Migration: add new columns to existing databases ──────────────────────
    # SQLite does not support IF NOT EXISTS on ALTER TABLE, so we catch the
    # "duplicate column" error per-column and ignore it safely.
    migrations = [
        "ALTER TABLE performance_logs ADD COLUMN match_ms REAL",
        "ALTER TABLE performance_logs ADD COLUMN debug_frame_ms REAL",
        "ALTER TABLE performance_logs ADD COLUMN debug_face_ms REAL",
        "ALTER TABLE performance_logs ADD COLUMN image_full_save_ms REAL",
        "ALTER TABLE performance_logs ADD COLUMN image_raw_save_ms REAL",
        "ALTER TABLE performance_logs ADD COLUMN face_aligned_save_ms REAL",
        "ALTER TABLE performance_logs ADD COLUMN face_raw_save_ms REAL",
        "ALTER TABLE performance_logs ADD COLUMN faces_identified INTEGER DEFAULT 0",
        "ALTER TABLE performance_logs ADD COLUMN last_identified_id TEXT DEFAULT ''",
        "ALTER TABLE performance_logs ADD COLUMN last_identified_name TEXT DEFAULT ''",
        "ALTER TABLE performance_logs ADD COLUMN last_face_save_path TEXT DEFAULT ''",
    ]
    async with engine.begin() as conn:
        for stmt in migrations:
            try:
                await conn.exec_driver_sql(stmt)
            except Exception:
                pass  # column already exists — ignore
