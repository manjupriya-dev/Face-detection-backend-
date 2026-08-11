CREATE TABLE IF NOT EXISTS employees (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  photo_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS prototypes (
  employee_id TEXT PRIMARY KEY,
  embedding BLOB NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(employee_id) REFERENCES employees(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS samples (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  employee_id TEXT NOT NULL,
  embedding BLOB NOT NULL,
  quality REAL NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(employee_id) REFERENCES employees(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  camera_id TEXT,
  track_id INTEGER,
  result TEXT NOT NULL,
  score REAL,
  meta TEXT
);

CREATE TABLE IF NOT EXISTS performance_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    fps REAL,
    capture_ms REAL,
    detect_ms REAL,
    align_ms REAL,
    embed_ms REAL,
    match_ms REAL,
    write_ms REAL,
    total_ms REAL,
    face_detected INTEGER,
    embedding_generated INTEGER,
    debug_frame_ms REAL,
    debug_face_ms REAL,
    image_full_save_ms REAL,
    image_raw_save_ms REAL,
    face_aligned_save_ms REAL,
    face_raw_save_ms REAL,
    video_write_ms REAL,
    faces_identified INTEGER DEFAULT 0,
    last_identified_id TEXT,
    last_identified_name TEXT,
    last_face_save_path TEXT
);
