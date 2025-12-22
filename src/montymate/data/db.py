from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Pragmas: set per-connection.
    conn.execute("PRAGMA foreign_keys = ON;")  # per-connection  [oai_citation:6‡SQLite](https://sqlite.org/foreignkeys.html?utm_source=chatgpt.com)
    conn.execute("PRAGMA journal_mode = WAL;")  # enable WAL  [oai_citation:7‡SQLite](https://sqlite.org/wal.html?utm_source=chatgpt.com)
    conn.execute("PRAGMA busy_timeout = 5000;")  # reduce SQLITE_BUSY flakiness
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection, *, write: bool) -> Iterator[sqlite3.Connection]:
    """
    Use BEGIN IMMEDIATE for write transactions to avoid upgrading read->write mid-flight,
    which helps prevent SQLITE_BUSY in some patterns.
    """
    try:
        if write:
            conn.execute("BEGIN IMMEDIATE;")
        else:
            conn.execute("BEGIN;")
        yield conn
        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise