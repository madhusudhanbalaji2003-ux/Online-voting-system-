"""
SecureVote — Online Voting System
Main Flask Application with Face Recognition, Liveness Detection, and Blockchain.
"""

import base64
import csv
import hashlib
import io
import json
import os
import random
import smtplib
import sqlite3
import string
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from functools import wraps

import numpy as np
from flask import (Flask, Response, flash, jsonify, redirect, render_template, request,
                   session, url_for)

from blockchain import Blockchain
from face_utils import (analyze_liveness_frames, compare_faces,
                        decode_base64_image, deserialize_encoding,
                        find_duplicate_face, get_face_encoding,
                        save_face_image, serialize_encoding)

# ──────────────────────────────────────────────
# Flask App Configuration
# ──────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_template_dir = os.path.join(_BASE_DIR, "templates")
_static_dir = os.path.join(_BASE_DIR, "static")

app = Flask(
    __name__,
    template_folder=_template_dir,
    static_folder=_static_dir
)
# Use a fixed secret key (from env var) so sessions survive cold starts on Vercel
app.secret_key = os.environ.get("SECRET_KEY", "securevote-default-secret-key-change-in-prod")

# Serverless detection (Vercel / AWS Lambda)
_IS_SERVERLESS = bool(
    os.environ.get("VERCEL")
    or os.environ.get("VERCEL_ENV")
    or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
    or os.environ.get("LAMBDA_TASK_ROOT")
)
_IS_VERCEL = _IS_SERVERLESS

# Use /tmp on Vercel (read-only filesystem outside /tmp); fall back to static/ locally
_static_dir = str(app.static_folder or "static")
_TMP_BASE = "/tmp" if _IS_SERVERLESS else os.path.join(_static_dir, "uploads")
app.config["UPLOAD_FOLDER"] = os.path.join(_TMP_BASE, "faces")
app.config["CANDIDATE_UPLOAD_FOLDER"] = os.path.join(_TMP_BASE, "candidates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

# Create upload directories (safe on both Vercel /tmp and local static/)
try:
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["CANDIDATE_UPLOAD_FOLDER"], exist_ok=True)
except Exception as _e:
    print(f"[Uploads] Notice on creating folders: {_e}")


@app.route("/static/uploads/<path:filename>")
def custom_static_uploads(filename):
    """Serve uploaded images from /tmp on Vercel or local static/uploads."""
    from flask import send_from_directory
    if os.path.exists(os.path.join(_TMP_BASE, filename)):
        return send_from_directory(_TMP_BASE, filename)
    local_dir = os.path.join(_static_dir, "uploads")
    if os.path.exists(os.path.join(local_dir, filename)):
        return send_from_directory(local_dir, filename)
    return ("File not found", 404)

# ──────────────────────────────────────────────
# Email Configuration (Multi-Provider: Resend, Brevo, Gmail SMTP)
# Set these in your .env file or environment variables
# ──────────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    try:
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
    except Exception:
        pass

SMTP_EMAIL = os.environ.get("SMTP_EMAIL", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
BREVO_API_KEY = os.environ.get("BREVO_API_KEY", "")
EMAIL_SENDER_NAME = os.environ.get("EMAIL_SENDER_NAME", "SecureVote")
EMAIL_FROM = os.environ.get("EMAIL_FROM", "onboarding@resend.dev")

# Admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# ──────────────────────────────────────────────
# Background Scheduler (voting-start auto-notify)
# Disabled on Vercel (serverless — no persistent threads)
# ──────────────────────────────────────────────
_scheduler = None
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.date import DateTrigger
    if not _IS_SERVERLESS:
        _scheduler = BackgroundScheduler(daemon=True)
        _scheduler.start()
except Exception:
    BackgroundScheduler = None  # type: ignore
    DateTrigger = None  # type: ignore

# ──────────────────────────────────────────────
# Blockchain Instance
# Persisted to SQLite so votes survive Vercel cold starts
# ──────────────────────────────────────────────
blockchain = Blockchain()

# ──────────────────────────────────────────────
# Database Setup
# ──────────────────────────────────────────────
# Use /tmp on Vercel (writable); fall back to project root locally
if _IS_SERVERLESS:
    DB_PATH = "/tmp/database.db"
    _bundled_db = os.path.join(os.path.dirname(__file__), "database.db")
    if not os.path.exists(DB_PATH) and os.path.exists(_bundled_db):
        try:
            import shutil
            shutil.copy2(_bundled_db, DB_PATH)
            print("[Serverless] Seeded /tmp/database.db from bundled database.db")
        except Exception as _e:
            print(f"[Serverless] Warning copying database: {_e}")
else:
    DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")


def get_db():
    """Get a database connection."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    if not _IS_SERVERLESS:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
    else:
        try:
            conn.execute("PRAGMA journal_mode=DELETE")
        except Exception:
            pass
    return conn


def init_db():
    """Initialize the database schema and seed default election."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS voters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            voter_id TEXT UNIQUE NOT NULL,
            date_of_birth TEXT,
            face_encoding BLOB,
            face_image_path TEXT,
            has_voted INTEGER DEFAULT 0,
            registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS otps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            otp TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            used INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS elections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            is_active INTEGER DEFAULT 1,
            results_visible INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            party TEXT,
            symbol TEXT,
            election_id INTEGER,
            FOREIGN KEY (election_id) REFERENCES elections(id)
        );

        CREATE TABLE IF NOT EXISTS votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            voter_id TEXT NOT NULL,
            election_id INTEGER,
            block_hash TEXT NOT NULL,
            voted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Seed default election if none exists
    existing = cursor.execute("SELECT COUNT(*) FROM elections").fetchone()[0]
    if existing == 0:
        cursor.execute(
            "INSERT INTO elections (title, description) VALUES (?, ?)",
            ("General Election 2026", "National General Election — Cast your vote securely!")
        )
        election_id = cursor.lastrowid

        candidates = [
            ("Candidate A", "Democratic Party", "🗳️"),
            ("Candidate B", "Progressive Alliance", "🌟"),
            ("Candidate C", "National Front", "🏛️"),
            ("Candidate D", "People's Party", "🤝"),
        ]
        for name, party, symbol in candidates:
            cursor.execute(
                "INSERT INTO candidates (name, party, symbol, election_id) VALUES (?, ?, ?, ?)",
                (name, party, symbol, election_id)
            )

    # Migrate: add photo column to candidates if it doesn't exist
    try:
        cursor.execute("ALTER TABLE candidates ADD COLUMN photo TEXT")
        conn.commit()
    except Exception:
        pass  # Column already exists

    # Migrate: add results_visible column to elections if it doesn't exist
    try:
        cursor.execute("ALTER TABLE elections ADD COLUMN results_visible INTEGER DEFAULT 0")
        conn.commit()
    except Exception:
        pass  # Column already exists

    # Migrate: add voting_start / voting_end columns for time-period control
    for col in ("voting_start", "voting_end"):
        try:
            cursor.execute(f"ALTER TABLE elections ADD COLUMN {col} TEXT")
            conn.commit()
        except Exception:
            pass  # Column already exists

    # Create allowed_voters whitelist table (CSV import)
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS allowed_voters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            voter_name TEXT NOT NULL,
            voter_id   TEXT UNIQUE NOT NULL,
            imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    conn.close()


# Initialize database on startup

# -----------------------------------------------------------------
# Jinja2 custom filter: format ISO datetime as "DD-MM-YYYY HH:MM"
# -----------------------------------------------------------------
@app.template_filter('ddmmyyyy')
def ddmmyyyy(value):
    """Convert ISO8601 string (e.g. "2026-11-28T11:07") to
    "DD-MM-YYYY HH:MM" format (First date, next month, then year).
    """
    if not value:
        return ""
    try:
        val = str(value).strip().replace(' ', 'T')
        if len(val) == 16:
            val += ':00'
        dt = datetime.fromisoformat(val)
        return dt.strftime('%d-%m-%Y %H:%M')
    except Exception:
        try:
            parts = str(value).strip().split('T') if 'T' in str(value) else str(value).strip().split(' ')
            d_parts = parts[0].split('-')
            if len(d_parts) == 3 and len(d_parts[0]) == 4:
                time_part = parts[1][:5] if len(parts) > 1 else ''
                return f"{d_parts[2]}-{d_parts[1]}-{d_parts[0]} {time_part}".strip()
        except Exception:
            pass
        return str(value)
try:
    init_db()
except Exception as _e:
    print(f"[Warning] init_db failed on startup: {_e}")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def generate_otp(length=6):
    """Generate a random numeric OTP."""
    return "".join(random.choices(string.digits, k=length))


def _send_via_resend(to_email, subject, html_content):
    """Send email via Resend Free HTTP API (3000 free emails/mo, instant HTTPS delivery)."""
    api_key = os.environ.get("RESEND_API_KEY", "").strip()
    if not api_key:
        return False, "RESEND_API_KEY is not set"

    sender_name = os.environ.get("EMAIL_SENDER_NAME", "SecureVote")
    sender_email = os.environ.get("EMAIL_FROM", "onboarding@resend.dev")
    payload = {
        "from": f"{sender_name} <{sender_email}>",
        "to": [to_email],
        "subject": subject,
        "html": html_content
    }
    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "SecureVote/1.0"
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read().decode("utf-8")
            print(f"[Resend API] Successfully delivered email to {to_email}: {data}")
            return True, None
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="ignore")
        print(f"[Resend API Error {e.code}] {err}")
        return False, f"Resend API error {e.code}: {err}"
    except Exception as e:
        print(f"[Resend Error] {e}")
        return False, str(e)


def _send_via_brevo(to_email, subject, html_content):
    """Send email via Brevo Free HTTP API (300 free emails/day)."""
    api_key = os.environ.get("BREVO_API_KEY", "").strip()
    if not api_key:
        return False, "BREVO_API_KEY is not set"

    sender_name = os.environ.get("EMAIL_SENDER_NAME", "SecureVote")
    sender_email = os.environ.get("EMAIL_FROM", "noreply@securevote.internal")
    payload = {
        "sender": {"name": sender_name, "email": sender_email},
        "to": [{"email": to_email}],
        "subject": subject,
        "htmlContent": html_content
    }
    req = urllib.request.Request(
        "https://api.brevo.com/v3/smtp/email",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read().decode("utf-8")
            print(f"[Brevo API] Successfully delivered email to {to_email}: {data}")
            return True, None
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="ignore")
        print(f"[Brevo API Error {e.code}] {err}")
        return False, f"Brevo API error {e.code}: {err}"
    except Exception as e:
        print(f"[Brevo Error] {e}")
        return False, str(e)


def _send_via_smtp(to_email, subject, html_content):
    """Send email via Gmail SMTP (TLS 587 or SSL 465)."""
    user_smtp = (os.environ.get("SMTP_EMAIL") or SMTP_EMAIL or "").strip()
    pass_smtp = (os.environ.get("SMTP_PASSWORD") or SMTP_PASSWORD or "").strip().replace(" ", "")
    if not user_smtp or not pass_smtp:
        return False, "SMTP credentials missing"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"SecureVote <{user_smtp}>"
    msg["To"] = to_email
    msg.attach(MIMEText(html_content, "html"))

    # Attempt 1: Port 587 STARTTLS
    try:
        with smtplib.SMTP(SMTP_SERVER, 587, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(user_smtp, pass_smtp)
            server.send_message(msg)
        print(f"[SMTP Port 587] Successfully sent email to {to_email}")
        return True, None
    except Exception as e_tls:
        print(f"[SMTP TLS 587 Warning] {e_tls} — Trying SSL 465...")

    # Attempt 2: Port 465 SSL
    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, 465, timeout=10) as server:
            server.login(user_smtp, pass_smtp)
            server.send_message(msg)
        print(f"[SMTP Port 465] Successfully sent email to {to_email}")
        return True, None
    except Exception as e_ssl:
        print(f"[SMTP SSL 465 Error] {e_ssl}")
        return False, f"SMTP error: {e_ssl}"


def send_email_message(to_email, subject, html_content):
    """
    Unified email dispatcher.
    Tries in priority order:
      1. Resend HTTP REST API (fastest, free 3000/mo, no port blocks)
      2. Brevo HTTP REST API (free 300/day)
      3. Gmail SMTP (official Google App Password)
    """
    # 1. Resend
    if os.environ.get("RESEND_API_KEY", "").strip():
        ok, err = _send_via_resend(to_email, subject, html_content)
        if ok:
            return True, "Resend HTTP API"
        print(f"[Mailer] Resend failed ({err}), falling back...")

    # 2. Brevo
    if os.environ.get("BREVO_API_KEY", "").strip():
        ok, err = _send_via_brevo(to_email, subject, html_content)
        if ok:
            return True, "Brevo HTTP API"
        print(f"[Mailer] Brevo failed ({err}), falling back...")

    # 3. Gmail SMTP
    user_smtp = (os.environ.get("SMTP_EMAIL") or SMTP_EMAIL or "").strip()
    pass_smtp = (os.environ.get("SMTP_PASSWORD") or SMTP_PASSWORD or "").strip()
    if user_smtp and pass_smtp:
        ok, err = _send_via_smtp(to_email, subject, html_content)
        if ok:
            return True, "Gmail SMTP"
        print(f"[Mailer] Gmail SMTP failed ({err}), falling back to dev mode...")

    return False, "dev_mode"


def send_otp_email(email, otp):
    """Send OTP via email using the fastest free model (Resend, Brevo, or Gmail SMTP)."""
    subject = "🔐 SecureVote — Your OTP Verification Code"
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="margin:0;padding:40px;background-color:#0b0f19;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#f3f4f6;">
      <table align="center" border="0" cellpadding="0" cellspacing="0" width="100%" style="max-width:500px;background:#151c2e;border:1px solid #1f293d;border-radius:18px;overflow:hidden;box-shadow:0 10px 30px rgba(0,0,0,0.5);">
        <tr>
          <td style="padding:32px 36px;text-align:center;background:linear-gradient(135deg,rgba(124,58,237,0.15),rgba(99,102,241,0.05));border-bottom:1px solid #1f293d;">
            <div style="font-size:26px;font-weight:800;color:#ffffff;letter-spacing:-0.5px;">
              🗳️ <span style="background:linear-gradient(135deg,#c4b5fd,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">SecureVote</span>
            </div>
            <div style="color:#9ca3af;font-size:12px;margin-top:6px;letter-spacing:1px;text-transform:uppercase;">Decentralized Digital Voting</div>
          </td>
        </tr>
        <tr>
          <td style="padding:36px 36px 20px 36px;">
            <h2 style="margin:0 0 12px 0;font-size:20px;font-weight:700;color:#f9fafb;text-align:center;">Voter Verification Code</h2>
            <p style="margin:0 0 24px 0;font-size:14px;color:#9ca3af;line-height:1.6;text-align:center;">
              Enter the 6-digit code below to authenticate your identity and proceed with voting.
            </p>
            <div style="background:#0b0f19;border:1.5px dashed #7c3aed;border-radius:12px;padding:22px;text-align:center;margin:24px 0;">
              <span style="font-size:38px;font-weight:800;letter-spacing:10px;color:#a78bfa;font-family:monospace;display:inline-block;padding-left:10px;">{otp}</span>
            </div>
            <p style="margin:0;font-size:12px;color:#6b7280;text-align:center;line-height:1.5;">
              ⏱️ This code expires in <strong>5 minutes</strong>. Never share this code with anyone.
            </p>
          </td>
        </tr>
        <tr>
          <td style="padding:20px 36px;background:#0d1322;border-top:1px solid #1f293d;text-align:center;">
            <p style="margin:0;font-size:11px;color:#4b5563;">
              Secured with SHA-256 blockchain verification • Do not reply
            </p>
          </td>
        </tr>
      </table>
    </body>
    </html>
    """

    ok, method = send_email_message(email, subject, html_content)
    if ok:
        print(f"[EMAIL SUCCESS] Delivered OTP to {email} via {method}")
        return True

    # Dev Mode / Fallback: Never block voter, log clearly to console & store in session
    print("\n" + "="*62)
    print(f"[SECUREVOTE DEV OTP] Code: {otp} for {email}")
    print(">> Enter this code in your browser to verify immediately!")
    print(">> (To deliver real emails to Gmail, add RESEND_API_KEY or Gmail App Password to your .env)")
    print("="*62 + "\n")
    try:
        from flask import has_request_context
        if has_request_context():
            session["dev_otp"] = otp
            session["dev_otp_time"] = time.time()
    except Exception:
        pass
    return True


def send_voting_start_notification(email, voter_name, election_title):
    """Send 'Voting has started NOW!' email to a single voter."""
    subject = f"🗳️ SecureVote — Voting is NOW OPEN: {election_title}"
    html = f"""
    <html>
    <body style="font-family:'Segoe UI',sans-serif;background:#0f0f23;color:#e0e0e0;padding:40px;">
        <div style="max-width:520px;margin:auto;background:#1a1a2e;border-radius:16px;padding:40px;border:1px solid #16213e;">
            <h2 style="color:#34d399;margin-bottom:4px;">🗳️ Voting is NOW OPEN!</h2>
            <p style="color:#a78bfa;font-size:0.9rem;margin-top:0;">{election_title}</p>
            <hr style="border:none;border-top:1px solid #16213e;margin:20px 0;">
            <p style="color:#e0e0e0;">Dear <strong>{voter_name}</strong>,</p>
            <p style="color:#c4b5fd;">The voting period has just begun. You can now cast your vote securely.</p>
            <div style="background:#16213e;border-radius:12px;padding:24px;margin:24px 0;border-left:4px solid #34d399;">
                <div style="font-size:1.2rem;font-weight:700;color:#34d399;text-align:center;">
                    🟢 Voting is LIVE right now!
                </div>
            </div>
            <div style="text-align:center;margin-top:28px;">
                <a href="http://127.0.0.1:5000/login"
                   style="background:linear-gradient(135deg,#34d399,#059669);color:#fff;
                          text-decoration:none;padding:14px 36px;border-radius:8px;
                          font-weight:700;font-size:1rem;">
                    🗳️ Vote Now
                </a>
            </div>
            <p style="color:#555;font-size:0.75rem;margin-top:28px;text-align:center;">
                This is an automated message from SecureVote. Do not reply.
            </p>
        </div>
    </body>
    </html>
    """
    ok, method = send_email_message(email, subject, html)
    if not ok:
        print(f"[DEV MODE] Voting-start notification for {email}: {election_title}")
    return True


def _auto_notify_voting_start(election_id):
    """Background job: fires at voting_start time, emails all registered voters."""
    print(f"[Scheduler] Voting started for election {election_id} — sending notifications…")
    conn = get_db()
    election = conn.execute("SELECT * FROM elections WHERE id = ?", (election_id,)).fetchone()
    if not election:
        conn.close()
        return
    voters = conn.execute("SELECT name, email FROM voters").fetchall()
    conn.close()
    sent, failed = 0, 0
    for v in voters:
        ok = send_voting_start_notification(v["email"], v["name"], election["title"])
        if ok:
            sent += 1
        else:
            failed += 1
    print(f"[Scheduler] Voting-start notifications sent: {sent} ok, {failed} failed.")


def schedule_voting_start_job(election_id, start_dt):
    """Schedule (or re-schedule) the auto-notify job for this election."""
    if _scheduler is None or DateTrigger is None or not callable(DateTrigger):
        print("[Scheduler] Disabled on Vercel — skipping auto-notify scheduling.")
        return
    job_id = f"voting_start_{election_id}"
    # Remove existing job if any
    if _scheduler.get_job(job_id):
        _scheduler.remove_job(job_id)
    # Only schedule if start_dt is in the future
    if start_dt > datetime.now():
        date_trigger = DateTrigger  # type: ignore
        _scheduler.add_job(
            _auto_notify_voting_start,
            trigger=date_trigger(run_date=start_dt),
            id=job_id,
            args=[election_id],
            replace_existing=True
        )
        print(f"[Scheduler] Voting-start notification scheduled for {start_dt.isoformat()}")
    else:
        print("[Scheduler] start_dt is in the past — skipping auto-notify scheduling.")


def send_announcement_email(email, voter_name, election_title, start_dt, end_dt):
    """Send voting announcement email to a single voter."""
    start_str = start_dt.strftime("%d %B %Y at %I:%M %p")
    end_str   = end_dt.strftime("%d %B %Y at %I:%M %p")
    subject = f"📢 SecureVote — Voting Announcement: {election_title}"

    html = f"""
    <html>
    <body style="font-family:'Segoe UI',sans-serif;background:#0f0f23;color:#e0e0e0;padding:40px;">
        <div style="max-width:520px;margin:auto;background:#1a1a2e;border-radius:16px;padding:40px;border:1px solid #16213e;">
            <h2 style="color:#7c3aed;margin-bottom:4px;">📢 SecureVote</h2>
            <p style="color:#a78bfa;font-size:0.9rem;margin-top:0;">Official Voting Announcement</p>
            <hr style="border:none;border-top:1px solid #16213e;margin:20px 0;">
            <p style="color:#e0e0e0;">Dear <strong>{voter_name}</strong>,</p>
            <p style="color:#c4b5fd;">The voting period for <strong>{election_title}</strong> has been officially scheduled.</p>
            <div style="background:#16213e;border-radius:12px;padding:24px;margin:24px 0;border-left:4px solid #7c3aed;">
                <div style="margin-bottom:14px;">
                    <div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;color:#a78bfa;margin-bottom:4px;">🗓️ Voting Starts</div>
                    <div style="font-size:1.1rem;font-weight:700;color:#34d399;">{start_str}</div>
                </div>
                <div>
                    <div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;color:#a78bfa;margin-bottom:4px;">🏁 Voting Ends</div>
                    <div style="font-size:1.1rem;font-weight:700;color:#f87171;">{end_str}</div>
                </div>
            </div>
            <p style="color:#a0a0b8;font-size:0.9rem;">Please make sure to cast your vote within the scheduled window. Your vote is secured by blockchain technology.</p>
            <div style="text-align:center;margin-top:28px;">
                <a href="http://127.0.0.1:5000/login" style="background:linear-gradient(135deg,#7c3aed,#4c1d95);color:#fff;text-decoration:none;padding:12px 32px;border-radius:8px;font-weight:700;font-size:0.95rem;">🗳️ Go Vote Now</a>
            </div>
            <p style="color:#555;font-size:0.75rem;margin-top:28px;text-align:center;">This is an automated message from SecureVote. Do not reply.</p>
        </div>
    </body>
    </html>
    """
    ok, method = send_email_message(email, subject, html)
    if not ok:
        print(f"[DEV MODE] Announcement for {email}: {election_title} starts {start_dt}")
    return True


def send_vote_acknowledgement_email(email, voter_name, voted_at, block_hash, election_title):
    """Send a congratulations acknowledgement email to the voter after successful voting."""
    subject = "🎉 SecureVote — Thank You for Voting!"
    try:
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
        <body style="margin:0;padding:0;background:#0a0a14;font-family:'Segoe UI',Arial,sans-serif;">

          <!-- Outer wrapper -->
          <table width="100%" cellpadding="0" cellspacing="0" border="0"
                 style="background:#0a0a14;padding:40px 16px;">
            <tr><td align="center">

              <!-- Card -->
              <table width="520" cellpadding="0" cellspacing="0" border="0"
                     style="max-width:520px;background:#0f0f1e;
                            border-radius:20px;
                            border:1px solid #1e1e38;
                            box-shadow:0 24px 64px rgba(0,0,0,.6);">

                <!-- Top accent bar -->
                <tr>
                  <td style="height:4px;background:linear-gradient(90deg,#34d399,#38bdf8,#a78bfa);border-radius:20px 20px 0 0;"></td>
                </tr>

                <!-- Header -->
                <tr>
                  <td align="center" style="padding:40px 40px 24px;">

                    <!-- Glow circle icon -->
                    <div style="width:88px;height:88px;border-radius:50%;
                                background:linear-gradient(135deg,#34d399,#38bdf8);
                                display:inline-flex;align-items:center;justify-content:center;
                                box-shadow:0 0 0 12px rgba(52,211,153,.1),
                                           0 0 0 24px rgba(52,211,153,.05),
                                           0 12px 32px rgba(52,211,153,.35);
                                margin-bottom:24px;">
                      <span style="font-size:40px;line-height:1;">✅</span>
                    </div>

                    <h1 style="margin:0 0 8px;font-size:2rem;font-weight:800;
                               color:#ffffff;letter-spacing:-0.02em;">Congratulations!</h1>
                    <p style="margin:0;font-size:1rem;color:#94a3b8;line-height:1.5;">
                      You have successfully cast your vote in<br>
                      <strong style="color:#c4b5fd;">{election_title}</strong>
                    </p>

                  </td>
                </tr>

                <!-- Greeting -->
                <tr>
                  <td style="padding:0 40px 20px;">
                    <p style="margin:0;font-size:0.95rem;color:#e2e8f0;line-height:1.6;">
                      Dear <strong style="color:#ffffff;">{voter_name}</strong>,
                    </p>
                    <p style="margin:12px 0 0;font-size:0.92rem;color:#94a3b8;line-height:1.65;">
                      Thank you for exercising your democratic right. Your vote has been
                      securely recorded on the blockchain and is immutable and tamper-proof.
                    </p>
                  </td>
                </tr>

                <!-- Timestamp card -->
                <tr>
                  <td style="padding:0 40px 24px;">
                    <table width="100%" cellpadding="0" cellspacing="0" border="0"
                           style="background:linear-gradient(135deg,rgba(52,211,153,.08),rgba(56,189,248,.06));
                                  border:1px solid rgba(52,211,153,.3);
                                  border-radius:14px;padding:20px 24px;">
                      <tr>
                        <td>
                          <div style="font-size:0.65rem;font-weight:800;text-transform:uppercase;
                                      letter-spacing:2px;color:#34d399;margin-bottom:8px;">
                            &#128336;&nbsp; Voted On
                          </div>
                          <div style="font-size:1.05rem;font-weight:700;color:#ffffff;
                                      letter-spacing:-0.01em;line-height:1.3;
                                      text-shadow:0 0 20px rgba(52,211,153,.4);">
                            {voted_at}
                          </div>
                        </td>
                      </tr>
                    </table>
                  </td>
                </tr>

                <!-- Privacy notice -->
                <tr>
                  <td style="padding:0 40px 24px;">
                    <table width="100%" cellpadding="0" cellspacing="0" border="0"
                           style="background:rgba(167,139,250,.07);
                                  border:1px solid rgba(167,139,250,.22);
                                  border-radius:40px;padding:12px 20px;">
                      <tr>
                        <td align="center" style="font-size:0.82rem;color:#c4b5fd;line-height:1.4;">
                          &#128274;&nbsp; Your ballot is completely private.
                          The choice you made will never be disclosed.
                        </td>
                      </tr>
                    </table>
                  </td>
                </tr>

                <!-- Block hash -->
                <tr>
                  <td style="padding:0 40px 32px;">
                    <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
                                letter-spacing:1.5px;color:#64748b;margin-bottom:8px;">
                      &#128279;&nbsp; Blockchain Receipt (Block Hash)
                    </div>
                    <div style="font-family:'Courier New',monospace;font-size:0.7rem;
                                color:#a78bfa;word-break:break-all;line-height:1.55;
                                background:rgba(124,58,237,.07);
                                border:1px solid rgba(124,58,237,.2);
                                border-radius:8px;padding:12px 14px;">
                      {block_hash}
                    </div>
                    <p style="margin:8px 0 0;font-size:0.72rem;color:#475569;line-height:1.5;">
                      This cryptographic hash confirms your vote was recorded without revealing your choice.
                    </p>
                  </td>
                </tr>

                <!-- CTA button -->
                <tr>
                  <td align="center" style="padding:0 40px 36px;">
                    <a href="http://127.0.0.1:5000/"
                       style="display:inline-block;background:#ffffff;color:#000000;
                              text-decoration:none;padding:13px 36px;
                              border-radius:40px;font-weight:700;font-size:0.92rem;
                              box-shadow:0 4px 18px rgba(255,255,255,.2);">
                      &#127968;&nbsp; Back to Home
                    </a>
                  </td>
                </tr>

                <!-- Footer -->
                <tr>
                  <td style="padding:20px 40px 32px;border-top:1px solid #1e1e38;">
                    <p style="margin:0;font-size:0.72rem;color:#475569;text-align:center;line-height:1.5;">
                      This is an automated acknowledgement from <strong>SecureVote</strong>.
                      Please do not reply to this email.
                    </p>
                  </td>
                </tr>

              </table>
            </td></tr>
          </table>

        </body>
        </html>
        """
        ok, method = send_email_message(email, subject, html)
        if not ok:
            print(f"[DEV MODE] Vote acknowledgement for {email} — voted at {voted_at}")
        return True
    except Exception as e:
        print(f"Vote acknowledgement error for {email}: {e}")
        return False


def login_required(f):
    """Decorator to require voter authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "voter_email" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    """Decorator to require admin authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("is_admin"):
            flash("Admin access required.", "danger")
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated


# ──────────────────────────────────────────────
# Public Routes
# ──────────────────────────────────────────────
@app.route("/")
def index():
    """Landing page — passes voting period, candidates, and live stats for one-by-one scrolling sections."""
    conn = get_db()
    election = conn.execute("SELECT * FROM elections WHERE is_active = 1 LIMIT 1").fetchone()
    candidates = []
    vote_counts = {}
    total_votes = 0
    results_visible = False
    total_voters = 0

    if election:
        results_visible = bool(election["results_visible"])
        candidates = conn.execute(
            "SELECT * FROM candidates WHERE election_id = ?", (election["id"],)
        ).fetchall()
        vote_counts = blockchain.get_vote_counts(election["id"])
        total_votes = sum(vote_counts.values())

    try:
        total_voters = conn.execute("SELECT COUNT(*) FROM voters").fetchone()[0]
    except Exception:
        total_voters = 0
    conn.close()

    voting_start = ""
    voting_end   = ""
    voting_status = "not_set"  # not_set | scheduled | active | ended

    if election:
        vs = election["voting_start"]
        ve = election["voting_end"]
        if vs and ve:
            voting_start = vs
            voting_end   = ve
            now = datetime.now()
            start_dt = datetime.fromisoformat(vs)
            end_dt   = datetime.fromisoformat(ve)
            if now < start_dt:
                voting_status = "scheduled"
            elif now > end_dt:
                voting_status = "ended"
            else:
                voting_status = "active"

    total_blocks = len(blockchain.chain)
    latest_block = blockchain.get_latest_block()

    return render_template("index.html",
                           election=election,
                           candidates=candidates,
                           vote_counts=vote_counts,
                           total_votes=total_votes,
                           results_visible=results_visible,
                           total_voters=total_voters,
                           total_blocks=total_blocks,
                           latest_block=latest_block,
                           voting_start=voting_start,
                           voting_end=voting_end,
                           voting_status=voting_status)



@app.route("/register", methods=["GET", "POST"])
def register():
    """Voter registration with face capture."""
    if request.method == "GET":
        return render_template("register.html")

    # Handle JSON registration
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "Invalid request data"}), 400

    name = data.get("name", "").strip()
    email = data.get("email", "").strip()
    voter_id = data.get("voter_id", "").strip()
    dob_str = data.get("date_of_birth", "").strip()
    face_image_b64 = data.get("face_image", "")

    # Validation — all fields required
    if not all([name, email, voter_id, dob_str, face_image_b64]):
        return jsonify({"success": False, "message": "All fields are required, including your Date of Birth."}), 400

    # ── Whitelist check — voter_id must be in allowed_voters (if whitelist has entries) ──
    wl_conn = get_db()
    whitelist_count = wl_conn.execute("SELECT COUNT(*) FROM allowed_voters").fetchone()[0]
    if whitelist_count > 0:
        allowed = wl_conn.execute(
            "SELECT id FROM allowed_voters WHERE voter_id = ?", (voter_id,)
        ).fetchone()
        if not allowed:
            wl_conn.close()
            return jsonify({
                "success": False,
                "message": "⛔ Your Voter ID is not on the authorized voter list. "
                           "Only pre-approved voters can register. "
                           "Please contact the election admin if you believe this is an error."
            }), 403
    wl_conn.close()
    # ─────────────────────────────────────────────────────────────────────────

    # Age verification — must be 18 or older
    try:
        dob = datetime.strptime(dob_str, "%Y-%m-%d")
        today = datetime.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        if age < 18:
            return jsonify({
                "success": False,
                "message": f"You must be at least 18 years old to register. Your current age is {age} years."
            }), 403
    except ValueError:
        return jsonify({"success": False, "message": "Invalid date of birth format. Please use YYYY-MM-DD."}), 400

    conn = get_db()

    # Check duplicates
    existing = conn.execute(
        "SELECT id FROM voters WHERE email = ? OR voter_id = ?",
        (email, voter_id)
    ).fetchone()
    if existing:
        conn.close()
        return jsonify({"success": False, "message": "Email or Voter ID already registered."}), 400

    # Process face image
    face_array = decode_base64_image(face_image_b64)
    if face_array is None:
        conn.close()
        return jsonify({"success": False, "message": "Could not process face image."}), 400

    # Get face encoding
    encoding = get_face_encoding(face_array)
    if encoding is None:
        conn.close()
        return jsonify({"success": False, "message": "No face detected in the image. Please try again with better lighting."}), 400

    # ── Duplicate face check ──────────────────────────────────────────────────
    # Fetch all registered face encodings and compare against the new face.
    # This prevents the same person from creating multiple voter accounts.
    registered_encodings = conn.execute(
        "SELECT name, face_encoding FROM voters WHERE face_encoding IS NOT NULL"
    ).fetchall()
    registered_voters = [(row["name"], row["face_encoding"]) for row in registered_encodings]

    is_duplicate, matched_name = find_duplicate_face(encoding, registered_voters)
    if is_duplicate:
        conn.close()
        return jsonify({
            "success": False,
            "message": "⚠️ Your face is already registered in the system. "
                       "Each person can only register once. "
                       "If you believe this is an error, please contact the election office."
        }), 409
    # ─────────────────────────────────────────────────────────────────────────

    # Save face image
    filename = f"{voter_id}_{int(time.time())}.jpg"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    save_face_image(face_array, filepath)

    # Store in database
    try:
        conn.execute(
            "INSERT INTO voters (name, email, voter_id, date_of_birth, face_encoding, face_image_path) VALUES (?, ?, ?, ?, ?, ?)",
            (name, email, voter_id, dob_str, serialize_encoding(encoding), filename)
        )
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": "Registration successful! You can now log in to vote."})
    except Exception as e:
        conn.close()
        return jsonify({"success": False, "message": f"Registration failed: {str(e)}"}), 500


@app.route("/login", methods=["GET", "POST"])
def login():
    """Login page — enter email and voter ID to receive OTP."""
    if request.method == "GET":
        return render_template("login.html")

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "Invalid request"}), 400

    email = data.get("email", "").strip()
    voter_id = data.get("voter_id", "").strip()

    if not email or not voter_id:
        return jsonify({"success": False, "message": "Email and Voter ID are required."}), 400

    conn = get_db()
    voter = conn.execute(
        "SELECT * FROM voters WHERE email = ? AND voter_id = ?",
        (email, voter_id)
    ).fetchone()

    if not voter:
        conn.close()
        return jsonify({"success": False, "message": "No registered voter found with these credentials."}), 404

    # Generate and store OTP
    otp = generate_otp()
    expires_at = (datetime.now() + timedelta(minutes=5)).isoformat()

    conn.execute(
        "INSERT INTO otps (email, otp, expires_at) VALUES (?, ?, ?)",
        (email, otp, expires_at)
    )
    conn.commit()
    conn.close()

    # Send OTP
    if send_otp_email(email, otp):
        session["pending_email"] = email
        session["pending_voter_id"] = voter_id
        return jsonify({"success": True, "message": "OTP sent to your email!", "redirect": url_for("verify_otp")})
    else:
        return jsonify({"success": False, "message": "Failed to send OTP. Please try again."}), 500


@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp():
    """OTP verification page."""
    if "pending_email" not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for("login"))

    if request.method == "GET":
        dev_otp = session.get("dev_otp")
        return render_template("verify_otp.html", email=session["pending_email"], dev_otp=dev_otp)

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "Invalid request"}), 400

    otp = data.get("otp", "").strip()
    email = session.get("pending_email")

    if not otp:
        return jsonify({"success": False, "message": "Please enter the OTP."}), 400

    conn = get_db()
    record = conn.execute(
        "SELECT * FROM otps WHERE email = ? AND otp = ? AND used = 0 ORDER BY created_at DESC LIMIT 1",
        (email, otp)
    ).fetchone()

    if not record:
        conn.close()
        return jsonify({"success": False, "message": "Invalid OTP. Please try again."}), 400

    # Check expiry
    expires_at = datetime.fromisoformat(record["expires_at"])
    if datetime.now() > expires_at:
        conn.close()
        return jsonify({"success": False, "message": "OTP has expired. Please request a new one."}), 400

    # Mark OTP as used
    conn.execute("UPDATE otps SET used = 1 WHERE id = ?", (record["id"],))
    conn.commit()
    conn.close()

    session["otp_verified"] = True
    session.pop("dev_otp", None)
    return jsonify({"success": True, "message": "OTP verified!", "redirect": url_for("verify_face")})


@app.route("/verify-face", methods=["GET", "POST"])
def verify_face():
    """Face recognition + liveness detection page."""
    if not session.get("otp_verified"):
        flash("Please verify OTP first.", "warning")
        return redirect(url_for("login"))

    if request.method == "GET":
        return render_template("verify_face.html")

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "Invalid request"}), 400

    action = data.get("action")

    if action == "verify_face":
        # Face recognition step
        face_image_b64 = data.get("face_image", "")
        if not face_image_b64:
            return jsonify({"success": False, "message": "No face image provided."}), 400

        face_array = decode_base64_image(face_image_b64)
        if face_array is None:
            return jsonify({"success": False, "message": "Could not process image."}), 400

        # Get stored encoding
        email = session.get("pending_email")
        conn = get_db()
        voter = conn.execute("SELECT * FROM voters WHERE email = ?", (email,)).fetchone()
        conn.close()

        if not voter or not voter["face_encoding"]:
            return jsonify({"success": False, "message": "No registered face found."}), 400

        known_encoding = deserialize_encoding(voter["face_encoding"])
        _, distance = compare_faces(known_encoding, face_array)

        confidence = max(0, (1 - distance) * 100)

        if confidence > 70:
            session["face_verified"] = True
            return jsonify({
                "success": True,
                "message": f"Face matched! Confidence: {confidence:.1f}%",
                "confidence": confidence
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Face did not match (Required: >70%, Got: {confidence:.1f}%). Please try again in better lighting.",
                "confidence": confidence
            }), 400

    elif action == "liveness_check":
        # Liveness detection step
        frames = data.get("frames", [])
        is_live, blink_count, message = analyze_liveness_frames(frames)

        if is_live:
            session["liveness_verified"] = True
            return jsonify({"success": True, "message": message, "blinks": blink_count})
        else:
            return jsonify({"success": False, "message": message}), 400

    return jsonify({"success": False, "message": "Unknown action"}), 400


@app.route("/vote", methods=["GET", "POST"])
def vote():
    """Voting page — show ballot and cast vote."""
    if not session.get("face_verified") or not session.get("liveness_verified"):
        flash("Please complete face verification first.", "warning")
        return redirect(url_for("login"))

    email = session.get("pending_email")
    voter_id = session.get("pending_voter_id")

    conn = get_db()
    voter = conn.execute("SELECT * FROM voters WHERE email = ?", (email,)).fetchone()

    if not voter:
        conn.close()
        flash("Voter not found.", "danger")
        return redirect(url_for("login"))

    if voter["has_voted"]:
        conn.close()
        flash("You have already voted!", "info")
        return redirect(url_for("already_voted"))

    if request.method == "GET":
        election = conn.execute("SELECT * FROM elections WHERE is_active = 1 LIMIT 1").fetchone()
        if not election:
            conn.close()
            flash("No active election at this time.", "info")
            return redirect(url_for("index"))

        # ── Enforce voting time window ───────────────────────────────────────
        now = datetime.now()
        voting_start = election["voting_start"]
        voting_end = election["voting_end"]
        if voting_start and voting_end:
            start_dt = datetime.fromisoformat(voting_start)
            end_dt   = datetime.fromisoformat(voting_end)
            if now < start_dt:
                conn.close()
                return redirect(url_for("voting_closed",
                                        reason="not_started",
                                        start=voting_start,
                                        end=voting_end))
            if now > end_dt:
                conn.close()
                return redirect(url_for("voting_closed",
                                        reason="ended",
                                        start=voting_start,
                                        end=voting_end))
        # ─────────────────────────────────────────────────────────────────────

        candidates = conn.execute(
            "SELECT * FROM candidates WHERE election_id = ?", (election["id"],)
        ).fetchall()
        conn.close()
        return render_template("vote.html", election=election, candidates=candidates, voter=voter, voting_end=voting_end if 'voting_end' in locals() or 'voting_end' in globals() else election.get("voting_end"))

    # Cast vote
    data = request.get_json()
    if not data:
        conn.close()
        return jsonify({"success": False, "message": "Invalid request"}), 400

    candidate_id = data.get("candidate_id")
    if not candidate_id:
        conn.close()
        return jsonify({"success": False, "message": "Please select a candidate."}), 400

    candidate = conn.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
    if not candidate:
        conn.close()
        return jsonify({"success": False, "message": "Invalid candidate."}), 400

    election = conn.execute("SELECT * FROM elections WHERE id = ? AND is_active = 1", (candidate["election_id"],)).fetchone()
    if not election:
        conn.close()
        return jsonify({"success": False, "message": "Election is not active."}), 400

    # ── Enforce voting time window on POST too ───────────────────────────────
    now = datetime.now()
    voting_start = election["voting_start"]
    voting_end   = election["voting_end"]
    if voting_start and voting_end:
        start_dt = datetime.fromisoformat(voting_start)
        end_dt   = datetime.fromisoformat(voting_end)
        if now < start_dt:
            conn.close()
            return jsonify({"success": False, "message": "Voting has not started yet."}), 403
        if now > end_dt:
            conn.close()
            return jsonify({"success": False, "message": "The voting period has ended."}), 403
    # ─────────────────────────────────────────────────────────────────────────

    # Add vote to blockchain
    block = blockchain.add_vote(
        voter_id=voter_id,
        candidate_id=candidate_id,
        candidate_name=candidate["name"],
        election_id=election["id"]
    )

    # Update database
    conn.execute("UPDATE voters SET has_voted = 1 WHERE email = ?", (email,))
    conn.execute(
        "INSERT INTO votes (voter_id, election_id, block_hash) VALUES (?, ?, ?)",
        (voter_id, election["id"], block.hash)
    )
    conn.commit()
    conn.close()

    # Clear session
    session.pop("face_verified", None)
    session.pop("liveness_verified", None)
    session.pop("otp_verified", None)
    voted_at_str = datetime.now().strftime("%A, %d %B %Y at %I:%M:%S %p")
    session["vote_block_hash"] = block.hash
    session["voted_at"] = voted_at_str

    # Send congratulations acknowledgement email to the voter (in background thread)
    import threading
    threading.Thread(
        target=send_vote_acknowledgement_email,
        args=(
            email,
            voter["name"],
            voted_at_str,
            block.hash,
            election["title"]
        ),
        daemon=True
    ).start()

    return jsonify({
        "success": True,
        "message": "Vote cast successfully!",
        "block_hash": block.hash,
        "redirect": url_for("vote_success")
    })


@app.route("/vote-success")
def vote_success():
    """Vote confirmation page — shows congratulations with timestamp only."""
    block_hash = session.get("vote_block_hash", "N/A")
    voted_at   = session.get("voted_at", datetime.now().strftime("%A, %d %B %Y at %I:%M:%S %p"))
    # Deliberately do NOT pass candidate name — the page must not reveal who was voted for
    return render_template("success.html", block_hash=block_hash, voted_at=voted_at)


@app.route("/already-voted")
def already_voted():
    """Page shown when voter has already voted."""
    return render_template("already_voted.html")


@app.route("/voting-closed")
def voting_closed():
    """Page shown when voter visits outside the allowed voting window."""
    reason = request.args.get("reason", "ended")  # 'not_started' or 'ended'
    start  = request.args.get("start", "")
    end    = request.args.get("end", "")
    return render_template("voting_closed.html", reason=reason, start=start, end=end)


@app.route("/results")
def results():
    """Public election results page — only visible when admin has enabled results."""
    conn = get_db()
    election = conn.execute("SELECT * FROM elections WHERE is_active = 1 LIMIT 1").fetchone()
    candidates = []
    vote_counts = {}
    total_votes = 0
    results_visible = False

    if election:
        results_visible = bool(election["results_visible"])
        if results_visible:
            candidates = conn.execute(
                "SELECT * FROM candidates WHERE election_id = ?", (election["id"],)
            ).fetchall()
            vote_counts = blockchain.get_vote_counts(election["id"])
            total_votes = sum(vote_counts.values())

    conn.close()
    return render_template("results.html", election=election, candidates=candidates,
                           vote_counts=vote_counts, total_votes=total_votes,
                           results_visible=results_visible)


@app.route("/logout")
def logout():
    """Clear session and logout."""
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


# ──────────────────────────────────────────────
# Admin Routes
# ──────────────────────────────────────────────
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    """Admin login page."""
    if request.method == "GET":
        return render_template("admin/login.html")

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "Invalid request"}), 400

    username = data.get("username", "")
    password = data.get("password", "")

    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session["is_admin"] = True
        return jsonify({"success": True, "redirect": url_for("admin_dashboard")})
    else:
        return jsonify({"success": False, "message": "Invalid credentials."}), 401


@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    """Admin dashboard with statistics."""
    conn = get_db()
    total_voters = conn.execute("SELECT COUNT(*) FROM voters").fetchone()[0]
    total_voted = conn.execute("SELECT COUNT(*) FROM voters WHERE has_voted = 1").fetchone()[0]
    election = conn.execute("SELECT * FROM elections WHERE is_active = 1 LIMIT 1").fetchone()
    chain_length = len(blockchain.chain)
    chain_valid = blockchain.is_chain_valid()

    candidates = []
    vote_counts = {}
    now = datetime.now()
    voting_status = "not_set"  # not_set | scheduled | active | ended

    if election:
        candidates = conn.execute(
            "SELECT * FROM candidates WHERE election_id = ?", (election["id"],)
        ).fetchall()
        vote_counts = blockchain.get_vote_counts(election["id"])

        vs = election["voting_start"]
        ve = election["voting_end"]
        if vs and ve:
            start_dt = datetime.fromisoformat(vs)
            end_dt   = datetime.fromisoformat(ve)
            if now < start_dt:
                voting_status = "scheduled"
            elif now > end_dt:
                voting_status = "ended"
            else:
                voting_status = "active"

    conn.close()
    return render_template("admin/dashboard.html",
                           total_voters=total_voters,
                           total_voted=total_voted,
                           election=election,
                           chain_length=chain_length,
                           chain_valid=chain_valid,
                           candidates=candidates,
                           vote_counts=vote_counts,
                           voting_status=voting_status)


@app.route("/admin/voters")
@admin_required
def admin_voters():
    """Admin voter management page (registered voters + whitelist)."""
    conn = get_db()
    voters = conn.execute("SELECT * FROM voters ORDER BY registered_at DESC").fetchall()
    allowed = conn.execute("SELECT * FROM allowed_voters ORDER BY imported_at DESC").fetchall()
    conn.close()
    return render_template("admin/voters.html", voters=voters, allowed_voters=allowed)


@app.route("/admin/blockchain")
@admin_required
def admin_blockchain():
    """Blockchain explorer page."""
    chain = blockchain.get_chain()
    is_valid = blockchain.is_chain_valid()
    return render_template("admin/blockchain.html", chain=chain, is_valid=is_valid)


@app.route("/admin/toggle_results", methods=["POST"])
@admin_required
def admin_toggle_results():
    """Toggle result visibility for voters (admin only)."""
    conn = get_db()
    election = conn.execute("SELECT * FROM elections WHERE is_active = 1 LIMIT 1").fetchone()
    if election:
        new_state = 0 if election["results_visible"] else 1
        conn.execute("UPDATE elections SET results_visible = ? WHERE id = ?", (new_state, election["id"]))
        conn.commit()
        msg = "Results are now VISIBLE to voters." if new_state else "Results are now HIDDEN from voters."
        flash(msg, "success" if new_state else "warning")
    else:
        flash("No active election found.", "danger")
    conn.close()
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/set_voting_period", methods=["POST"])
@admin_required
def admin_set_voting_period():
    """Set or clear the voting start/end window for the active election."""
    conn = get_db()
    election = conn.execute("SELECT * FROM elections WHERE is_active = 1 LIMIT 1").fetchone()
    if not election:
        flash("No active election found.", "danger")
        conn.close()
        return redirect(url_for("admin_dashboard"))

    action = request.form.get("action", "set")
    if action == "clear":
        conn.execute(
            "UPDATE elections SET voting_start = NULL, voting_end = NULL WHERE id = ?",
            (election["id"],)
        )
        conn.commit()
        conn.close()
        flash("Voting time period cleared — voting is now unrestricted.", "warning")
        return redirect(url_for("admin_dashboard"))

    voting_start = request.form.get("voting_start", "").strip()
    voting_end   = request.form.get("voting_end", "").strip()

    if not voting_start or not voting_end:
        flash("Both start and end date/time are required.", "danger")
        conn.close()
        return redirect(url_for("admin_dashboard"))

    try:
        start_dt = datetime.fromisoformat(voting_start)
        end_dt   = datetime.fromisoformat(voting_end)
    except ValueError:
        flash("Invalid date/time format.", "danger")
        conn.close()
        return redirect(url_for("admin_dashboard"))

    if end_dt <= start_dt:
        flash("End time must be after start time.", "danger")
        conn.close()
        return redirect(url_for("admin_dashboard"))

    conn.execute(
        "UPDATE elections SET voting_start = ?, voting_end = ? WHERE id = ?",
        (start_dt.isoformat(timespec="minutes"), end_dt.isoformat(timespec="minutes"), election["id"])
    )
    conn.commit()

    # ── Always schedule the auto-notify job for voting start ─────────────────
    schedule_voting_start_job(election["id"], start_dt)
    # ─────────────────────────────────────────────────────────────────────────

    # ── Optionally send announcement (schedule) emails immediately ───────────
    notify = request.form.get("notify_voters", "")
    sent_count = 0
    failed_count = 0
    if notify == "1":
        voters = conn.execute("SELECT name, email FROM voters").fetchall()
        election_title = election["title"]
        for v in voters:
            ok = send_announcement_email(v["email"], v["name"], election_title, start_dt, end_dt)
            if ok:
                sent_count += 1
            else:
                failed_count += 1
        if voters:
            flash(
                f"Voting period set & announcements sent to {sent_count} voter(s)" +
                (f" ({failed_count} failed)." if failed_count else "."),
                "success"
            )
        else:
            flash("Voting period set. No registered voters found to notify.", "warning")
    else:
        flash(
            f"Voting period set: {start_dt.strftime('%d %b %Y %H:%M')} → {end_dt.strftime('%d %b %Y %H:%M')}. "
            f"Auto-notification scheduled for voting start.",
            "success"
        )
    # ─────────────────────────────────────────────────────────────────────────

    conn.close()
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/notify_voters", methods=["POST"])
@admin_required
def admin_notify_voters():
    """Manually send/re-send voting announcement to all registered voters."""
    conn = get_db()
    election = conn.execute("SELECT * FROM elections WHERE is_active = 1 LIMIT 1").fetchone()
    if not election or not election["voting_start"] or not election["voting_end"]:
        flash("No active election with a voting period set. Please set a voting period first.", "danger")
        conn.close()
        return redirect(url_for("admin_dashboard"))

    start_dt = datetime.fromisoformat(election["voting_start"])
    end_dt   = datetime.fromisoformat(election["voting_end"])
    voters   = conn.execute("SELECT name, email FROM voters").fetchall()
    conn.close()

    if not voters:
        flash("No registered voters found to notify.", "warning")
        return redirect(url_for("admin_dashboard"))

    sent, failed = 0, 0
    for v in voters:
        ok = send_announcement_email(v["email"], v["name"], election["title"], start_dt, end_dt)
        if ok:
            sent += 1
        else:
            failed += 1

    flash(
        f"Announcement sent to {sent} voter(s)" +
        (f" ({failed} failed)." if failed else "."),
        "success" if sent else "danger"
    )
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/import_voters_csv", methods=["POST"])
@admin_required
def admin_import_voters_csv():
    """Import allowed voter list from CSV (columns: voter_name, voter_id)."""
    csv_file = request.files.get("csv_file")
    if not csv_file or not csv_file.filename:
        flash("Please select a CSV file to upload.", "danger")
        return redirect(url_for("admin_voters") + "#whitelist")

    filename = csv_file.filename.lower()
    if not filename.endswith(".csv"):
        flash("Only .csv files are accepted.", "danger")
        return redirect(url_for("admin_voters") + "#whitelist")

    try:
        stream   = io.StringIO(csv_file.stream.read().decode("utf-8-sig"), newline=None)
        reader   = csv.DictReader(stream)
        headers  = [h.strip().lower() for h in (reader.fieldnames or [])]

        # Accept flexible header names
        name_col = next((h for h in headers if "name" in h), None)
        id_col   = next((h for h in headers if "id" in h or "voter" in h and h != name_col), None)

        if not name_col or not id_col:
            flash(
                "CSV must have columns named like 'voter_name' (or 'name') "
                "and 'voter_id' (or 'id'). Please check your file headers.",
                "danger"
            )
            return redirect(url_for("admin_voters"))

        conn = get_db()
        inserted = 0
        skipped  = 0
        errors   = []

        for row_num, row in enumerate(reader, start=2):
            # Normalise keys to lowercase
            row_norm = {k.strip().lower(): (v or "").strip() for k, v in row.items()}
            v_name = row_norm.get(name_col, "").strip()
            v_id   = row_norm.get(id_col, "").strip()

            if not v_name or not v_id:
                errors.append(f"Row {row_num}: empty name or voter_id — skipped.")
                skipped += 1
                continue

            try:
                conn.execute(
                    "INSERT OR IGNORE INTO allowed_voters (voter_name, voter_id) VALUES (?, ?)",
                    (v_name, v_id)
                )
                if conn.execute("SELECT changes()").fetchone()[0]:
                    inserted += 1
                else:
                    skipped += 1  # duplicate
            except Exception as row_err:
                errors.append(f"Row {row_num}: {row_err}")
                skipped += 1

        conn.commit()
        conn.close()

        msg = f"CSV imported: {inserted} voter(s) added, {skipped} skipped (duplicates/blank)."
        if errors:
            msg += " Errors: " + " | ".join(errors[:5])
        flash(msg, "success" if inserted else "warning")

    except Exception as e:
        flash(f"Failed to parse CSV: {e}", "danger")

    return redirect(url_for("admin_voters") + "#whitelist")



@app.route("/admin/allowed_voters")
@admin_required
def admin_allowed_voters():
    """View the imported allowed-voter whitelist."""
    conn = get_db()
    allowed = conn.execute("SELECT * FROM allowed_voters ORDER BY imported_at DESC").fetchall()
    voters  = conn.execute("SELECT * FROM voters ORDER BY registered_at DESC").fetchall()
    conn.close()
    return render_template("admin/voters.html", voters=voters, allowed_voters=allowed)


@app.route("/api/scheduler/status")
@admin_required
def api_scheduler_status():
    """Return info about pending scheduled jobs (admin only)."""
    if _scheduler is None:
        return jsonify({"jobs": [], "note": "Scheduler disabled on Vercel serverless."})
    jobs = []
    for job in _scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None
        })
    return jsonify({"jobs": jobs})


@app.route("/admin/delete_allowed_voter/<int:av_id>", methods=["POST"])
@admin_required
def admin_delete_allowed_voter(av_id):
    """Remove an entry from the allowed-voter whitelist."""
    conn = get_db()
    conn.execute("DELETE FROM allowed_voters WHERE id = ?", (av_id,))
    conn.commit()
    conn.close()
    flash("Voter removed from whitelist.", "success")
    return redirect(url_for("admin_voters") + "#whitelist")


@app.route("/admin/clear_whitelist", methods=["POST"])
@admin_required
def admin_clear_whitelist():
    """Remove all entries from the allowed-voter whitelist."""
    conn = get_db()
    deleted = conn.execute("DELETE FROM allowed_voters").rowcount
    conn.commit()
    conn.close()
    flash(f"Whitelist cleared — {deleted} entries removed. Registration is now unrestricted.", "warning")
    return redirect(url_for("admin_voters") + "#whitelist")


@app.route("/admin/export_whitelist_csv")
@admin_required
def admin_export_whitelist_csv():
    """Export the allowed-voter whitelist as a downloadable CSV."""
    import csv as _csv
    conn = get_db()
    rows = conn.execute("SELECT voter_name, voter_id, imported_at FROM allowed_voters ORDER BY voter_name").fetchall()
    conn.close()

    output = io.StringIO()
    writer = _csv.writer(output)
    writer.writerow(["voter_name", "voter_id", "imported_at"])
    for r in rows:
        writer.writerow([r["voter_name"], r["voter_id"], r["imported_at"]])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=authorized_voters.csv"}
    )


@app.route("/admin/reset", methods=["POST"])
@admin_required
def admin_reset():
    """Reset election data (admin only)."""
    conn = get_db()
    conn.execute("UPDATE voters SET has_voted = 0")
    conn.execute("DELETE FROM votes")
    conn.execute("DELETE FROM otps")
    conn.execute("UPDATE elections SET results_visible = 0")
    conn.commit()
    conn.close()
    blockchain.reset()
    flash("Election data has been reset.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/delete_voter/<int:voter_id>", methods=["POST"])
@admin_required
def admin_delete_voter(voter_id):
    """Delete a voter (admin only)."""
    conn = get_db()
    conn.execute("DELETE FROM voters WHERE id = ?", (voter_id,))
    conn.commit()
    conn.close()
    flash("Voter deleted successfully.", "success")
    return redirect(url_for("admin_voters"))


@app.route("/admin/candidates")
@admin_required
def admin_candidates():
    """Admin candidate management page."""
    conn = get_db()
    election = conn.execute("SELECT * FROM elections WHERE is_active = 1 LIMIT 1").fetchone()
    candidates = []
    if election:
        candidates = conn.execute(
            "SELECT * FROM candidates WHERE election_id = ? ORDER BY id DESC", (election["id"],)
        ).fetchall()
    conn.close()
    return render_template("admin/candidates.html", candidates=candidates, election=election)


@app.route("/admin/add_candidate", methods=["POST"])
@admin_required
def admin_add_candidate():
    """Add a new candidate to the active election (supports photo upload)."""
    name = request.form.get("name", "").strip()
    party = request.form.get("party", "").strip()
    symbol = request.form.get("symbol", "").strip()

    if not name:
        flash("Candidate name is required.", "danger")
        return redirect(url_for("admin_candidates"))

    # Handle photo upload
    photo_filename = None
    photo_file = request.files.get("photo")
    if photo_file and photo_file.filename:
        ext = os.path.splitext(photo_file.filename)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
            flash("Invalid photo format. Use JPG, PNG, WEBP or GIF.", "danger")
            return redirect(url_for("admin_candidates"))
        safe_name = name.replace(" ", "_").lower()
        photo_filename = f"{safe_name}_{int(time.time())}{ext}"
        photo_file.save(os.path.join(app.config["CANDIDATE_UPLOAD_FOLDER"], photo_filename))

    conn = get_db()
    election = conn.execute("SELECT * FROM elections WHERE is_active = 1 LIMIT 1").fetchone()
    if election:
        conn.execute(
            "INSERT INTO candidates (name, party, symbol, photo, election_id) VALUES (?, ?, ?, ?, ?)",
            (name, party, symbol, photo_filename, election["id"])
        )
        conn.commit()
        flash(f"Candidate '{name}' added successfully!", "success")
    else:
        flash("No active election found.", "danger")

    conn.close()
    return redirect(url_for("admin_candidates"))


@app.route("/admin/delete_candidate/<int:candidate_id>", methods=["POST"])
@admin_required
def admin_delete_candidate(candidate_id):
    """Delete a candidate (admin only)."""
    conn = get_db()
    # Remove the photo file if it exists
    candidate = conn.execute("SELECT photo FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
    if candidate and candidate["photo"]:
        photo_path = os.path.join(app.config["CANDIDATE_UPLOAD_FOLDER"], candidate["photo"])
        if os.path.exists(photo_path):
            os.remove(photo_path)
    conn.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
    conn.commit()
    conn.close()
    flash("Candidate deleted successfully.", "success")
    return redirect(url_for("admin_candidates"))


@app.route("/admin/logout")
def admin_logout():
    """Admin logout."""
    session.pop("is_admin", None)
    flash("Admin logged out.", "info")
    return redirect(url_for("index"))


# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────
@app.route("/api/blockchain/status")
def api_blockchain_status():
    """Get blockchain status."""
    return jsonify({
        "chain_length": len(blockchain.chain),
        "is_valid": blockchain.is_chain_valid(),
        "total_votes": blockchain.get_total_votes(),
        "latest_block_hash": blockchain.get_latest_block().hash
    })


@app.route("/api/results")
def api_results():
    """Get election results as JSON — only when admin has made them visible."""
    conn = get_db()
    election = conn.execute("SELECT * FROM elections WHERE is_active = 1 LIMIT 1").fetchone()
    if not election:
        conn.close()
        return jsonify({"error": "No active election"}), 404

    if not election["results_visible"]:
        conn.close()
        return jsonify({"error": "Results are not yet available. Please check back later."}), 403

    candidates = conn.execute(
        "SELECT * FROM candidates WHERE election_id = ?", (election["id"],)
    ).fetchall()
    vote_counts = blockchain.get_vote_counts(election["id"])
    total_votes = sum(vote_counts.values())

    results = []
    for c in candidates:
        count = vote_counts.get(c["name"], 0)
        results.append({
            "name": c["name"],
            "party": c["party"],
            "votes": count,
            "percentage": round((count / total_votes * 100), 1) if total_votes > 0 else 0
        })

    conn.close()
    return jsonify({
        "election": election["title"],
        "total_votes": total_votes,
        "results": sorted(results, key=lambda x: x["votes"], reverse=True)
    })


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    is_prod = bool(os.environ.get("PORT") or os.environ.get("RENDER") or os.environ.get("RAILWAY_STATIC_URL"))
    print("\n" + "=" * 60)
    print("  SecureVote - Online Voting System")
    print("  Face Recognition + Liveness + Blockchain")
    print(f"  Listening on http://0.0.0.0:{port}")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=port, debug=not is_prod)
