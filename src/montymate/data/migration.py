from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from importlib import resources
from typing import Iterable


MIGRATION_RE = re.compile(r"^(\d{4})_.*\.sql$")


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str


def load_migrations(package: str, subdir: str) -> list[Migration]:
    """
    Reads resources/db/migrations/*.sql from the installed package.
    """
    mig_pkg = resources.files(package).joinpath(subdir)
    migrations: list[Migration] = []
    for entry in mig_pkg.iterdir():
        if not entry.name.endswith(".sql"):
            continue
        m = MIGRATION_RE.match(entry.name)
        if not m:
            continue
        version = int(m.group(1))
        sql = entry.read_text(encoding="utf-8")
        migrations.append(Migration(version=version, name=entry.name, sql=sql))
    migrations.sort(key=lambda x: x.version)
    return migrations


def get_user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version;").fetchone()[0])


def apply_migrations(conn: sqlite3.Connection, migrations: Iterable[Migration]) -> None:
    current = get_user_version(conn)
    for mig in migrations:
        if mig.version <= current:
            continue
        conn.executescript(mig.sql)
        current = get_user_version(conn)


def ensure_application_id(conn: sqlite3.Connection, expected: int) -> None:
    # application_id is stored in the DB header and should be fixed for MontyMate DBs.  [oai_citation:10‡SQLite](https://sqlite.org/pragma.html?utm_source=chatgpt.com)
    got = int(conn.execute("PRAGMA application_id;").fetchone()[0])
    if got in (0, expected):
        if got == 0:
            conn.execute(f"PRAGMA application_id = {expected};")
        return
    raise RuntimeError(f"Not a MontyMate DB: application_id={got}, expected={expected}")