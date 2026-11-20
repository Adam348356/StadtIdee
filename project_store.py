"""Local, persistent storage for submitted city improvement proposals."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image


PROJECTS_DB = Path(__file__).resolve().parent / "submitted_projects.sqlite3"

_OBJECT_WORDS = {
    "Bäume": ("Baum", "Bäume"),
    "Beete": ("Beet", "Beete"),
    "Fahrradständer": ("Fahrradständer", "Fahrradständer"),
    "Bänke": ("Bank", "Bänke"),
}


def _connect(database: Path) -> sqlite3.Connection:
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            location TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            counts_json TEXT NOT NULL,
            items_json TEXT NOT NULL,
            original_png BLOB NOT NULL,
            edited_png BLOB NOT NULL,
            annotated_png BLOB NOT NULL
        )
        """
    )
    project_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(projects)").fetchall()
    }
    if "location" not in project_columns:
        connection.execute("ALTER TABLE projects ADD COLUMN location TEXT NOT NULL DEFAULT ''")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS project_comments (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               project_id TEXT NOT NULL,
               author TEXT NOT NULL,
               body TEXT NOT NULL,
               created_at TEXT NOT NULL,
               FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
           )"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS project_likes (
               project_id TEXT NOT NULL,
               voter_id TEXT NOT NULL,
               created_at TEXT NOT NULL,
               PRIMARY KEY(project_id, voter_id),
               FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
           )"""
    )
    return connection


def _image_png(image: Image.Image | np.ndarray) -> bytes:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def describe_project(counts: dict[str, int]) -> tuple[str, str]:
    """Create an editable German title and summary from the classified objects."""
    active = [(name, int(count)) for name, count in counts.items() if int(count) > 0]
    if not active:
        raise ValueError("Für einen Projektvorschlag muss mindestens ein Objekt erkannt sein.")

    phrases = []
    for name, count in active:
        singular, plural = _OBJECT_WORDS.get(name, (name, name))
        phrases.append(f"{count} {singular if count == 1 else plural}")

    if len(phrases) == 1:
        objects_text = phrases[0]
    elif len(phrases) == 2:
        objects_text = " und ".join(phrases)
    else:
        objects_text = ", ".join(phrases[:-1]) + " und " + phrases[-1]

    categories = {name for name, _ in active}
    has_greenery = bool(categories & {"Bäume", "Beete"})
    has_bikes = "Fahrradständer" in categories
    has_benches = "Bänke" in categories
    if has_greenery and has_bikes:
        theme = "Grüner und fahrradfreundlicher Platz"
    elif has_greenery:
        theme = "Mehr Grün am Platz"
    elif has_bikes and has_benches:
        theme = "Gut ankommen und verweilen"
    elif has_bikes:
        theme = "Besser mit dem Rad unterwegs"
    elif has_benches:
        theme = "Ein Platz zum Verweilen"
    else:
        theme = "Eine Idee für den Stadtplatz"

    total = sum(count for _, count in active)
    title = f"{theme}: {objects_text}"
    description_parts = [f"Der Entwurf ergänzt den öffentlichen Ort um {objects_text}."]
    benefits = []
    if "Bäume" in categories:
        benefits.append("Bäume spenden Schatten und bringen mehr Grün.")
    if "Beete" in categories:
        benefits.append("Beete bringen Blüten und Farbe.")
    if has_bikes:
        benefits.append("Fahrradständer schaffen geordnete Abstellplätze.")
    if has_benches:
        benefits.append("Bänke bieten Sitzplätze zum Ausruhen.")
    description_parts.extend(benefits)
    description_parts.append(
        f"Insgesamt umfasst der Vorschlag {total} "
        f'{"Ergänzung" if total == 1 else "Ergänzungen"}.'
    )
    return title, " ".join(description_parts)


def save_project(
    title: str,
    description: str,
    counts: dict[str, int],
    items: list[dict],
    original: Image.Image,
    edited: np.ndarray,
    annotated: np.ndarray,
    database: Path = PROJECTS_DB,
    location: str = "",
) -> str:
    clean_title = title.strip()
    clean_description = description.strip()
    clean_location = location.strip()[:120]
    if not clean_title or not clean_description:
        raise ValueError("Titel und Beschreibung dürfen nicht leer sein.")

    project_id = uuid4().hex
    serializable_items = [
        {
            key: item[key]
            for key in ("number", "label", "confidence", "position", "size")
        }
        for item in items
    ]
    connection = _connect(database)
    try:
        connection.execute(
            """INSERT INTO projects
               (id, title, description, location, created_at, counts_json, items_json,
                original_png, edited_png, annotated_png)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                project_id,
                clean_title,
                clean_description,
                clean_location,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                json.dumps(counts, ensure_ascii=False),
                json.dumps(serializable_items, ensure_ascii=False),
                _image_png(original),
                _image_png(edited),
                _image_png(annotated),
            ),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return project_id


def load_projects(database: Path = PROJECTS_DB, voter_id: str | None = None) -> list[dict]:
    if not database.exists():
        return []
    connection = _connect(database)
    try:
        rows = connection.execute(
            """SELECT id, title, description, location, created_at, counts_json, items_json,
                      original_png, edited_png, annotated_png,
                      (SELECT COUNT(*) FROM project_likes l WHERE l.project_id = p.id) AS like_count,
                      (SELECT COUNT(*) FROM project_comments c WHERE c.project_id = p.id) AS comment_count,
                      EXISTS(SELECT 1 FROM project_likes l
                             WHERE l.project_id = p.id AND l.voter_id = ?) AS liked_by_viewer
               FROM projects p ORDER BY like_count DESC, created_at DESC, rowid DESC""",
            (voter_id or "",),
        ).fetchall()
    finally:
        connection.close()

    return [
        {
            "id": row["id"],
            "title": row["title"],
            "description": row["description"],
            "location": row["location"],
            "created_at": row["created_at"],
            "counts": json.loads(row["counts_json"]),
            "items": json.loads(row["items_json"]),
            "original_png": row["original_png"],
            "edited_png": row["edited_png"],
            "annotated_png": row["annotated_png"],
            "like_count": row["like_count"],
            "comment_count": row["comment_count"],
            "liked_by_viewer": bool(row["liked_by_viewer"]),
        }
        for row in rows
    ]


def load_project(project_id: str, database: Path = PROJECTS_DB) -> dict | None:
    if not database.exists():
        return None
    for project in load_projects(database):
        if project["id"] == project_id:
            return project
    return None


def toggle_like(project_id: str, voter_id: str, database: Path = PROJECTS_DB) -> bool:
    """Toggle one like per Streamlit browser session; return the new liked state."""
    if not voter_id.strip():
        raise ValueError("Für ein Like fehlt die Sitzungskennung.")
    connection = _connect(database)
    try:
        exists = connection.execute(
            "SELECT 1 FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        if exists is None:
            raise ValueError("Dieses Projekt wurde nicht gefunden.")
        liked = connection.execute(
            "SELECT 1 FROM project_likes WHERE project_id = ? AND voter_id = ?",
            (project_id, voter_id),
        ).fetchone()
        if liked:
            connection.execute(
                "DELETE FROM project_likes WHERE project_id = ? AND voter_id = ?",
                (project_id, voter_id),
            )
            new_state = False
        else:
            connection.execute(
                "INSERT INTO project_likes (project_id, voter_id, created_at) VALUES (?, ?, ?)",
                (project_id, voter_id, datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )
            new_state = True
        connection.commit()
        return new_state
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def add_comment(
    project_id: str,
    body: str,
    author: str = "",
    database: Path = PROJECTS_DB,
) -> int:
    clean_body = body.strip()
    clean_author = author.strip()[:80] or "Anonym"
    if not clean_body:
        raise ValueError("Bitte gib einen Kommentar ein.")
    if len(clean_body) > 2000:
        raise ValueError("Kommentare dürfen höchstens 2000 Zeichen lang sein.")

    connection = _connect(database)
    try:
        exists = connection.execute(
            "SELECT 1 FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        if exists is None:
            raise ValueError("Dieses Projekt wurde nicht gefunden.")
        cursor = connection.execute(
            """INSERT INTO project_comments (project_id, author, body, created_at)
               VALUES (?, ?, ?, ?)""",
            (
                project_id,
                clean_author,
                clean_body,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def load_comments(project_id: str, database: Path = PROJECTS_DB) -> list[dict]:
    if not database.exists():
        return []
    connection = _connect(database)
    try:
        rows = connection.execute(
            """SELECT id, author, body, created_at FROM project_comments
               WHERE project_id = ? ORDER BY created_at DESC, id DESC""",
            (project_id,),
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]
