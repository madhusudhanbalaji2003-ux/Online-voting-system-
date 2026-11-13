"""
SecureVote — Online Voting System
Main Flask Application with Face Recognition, Liveness Detection, and Blockchain.
"""

import base64
import csv
import hashlib
import io
import os
import secrets
import smtplib
import sqlite3
import string
import sys
import time
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from functools import wraps
from math import ceil
from typing import Tuple

# Load .env file (uses python-dotenv if available, with built-in fallback)
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _load_env_file(env_path: str = _ENV_PATH) -> None:
    """Load key=value pairs from .env into os.environ without requiring external packages."""
    loaded_via_dotenv = False
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=True)
        loaded_via_dotenv = True
    except ImportError:
        pass

    if not loaded_via_dotenv and os.path.isfile(env_path):
        with open(env_path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]
                os.environ[key] = value


_load_env_file()

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger

import numpy as np
from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   session, url_for)

from blockchain import Blockchain
from face_utils import (analyze_liveness_frames, compare_faces,
                        decode_base64_image, deserialize_encoding,
                        detect_face_core, detect_face_in_frame,
                        find_duplicate_face, get_face_encoding,
                        save_face_image, serialize_encoding)

# ──────────────────────────────────────────────
# Flask App Configuration
# ──────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.urandom(32).hex()
app.config["UPLOAD_FOLDER"] = os.path.join(app.static_folder, "uploads", "faces")
app.config["CANDIDATE_UPLOAD_FOLDER"] = os.path.join(app.static_folder, "uploads", "candidates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

# Create upload directories
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["CANDIDATE_UPLOAD_FOLDER"], exist_ok=True)

# ──────────────────────────────────────────────
# Email Configuration (Gmail SMTP)
# Set these environment variables or modify directly
# ──────────────────────────────────────────────
def _normalize_smtp_password(password: str, server: str) -> str:
    """Strip quotes/whitespace and remove grouping spaces from Gmail App Passwords."""
    cleaned = password.strip()
    if server == "smtp.gmail.com":
        compact = cleaned.replace(" ", "")
        if len(compact) == 16:
            return compact
    return cleaned


def _get_smtp_credentials() -> Tuple[str, str]:
    """Return configured (smtp_email, smtp_password), reloading .env if empty."""
    email_val = os.environ.get("SMTP_EMAIL", "").strip()
    pass_val = _normalize_smtp_password(os.environ.get("SMTP_PASSWORD", ""), SMTP_SERVER)
    if not email_val or not pass_val:
        _load_env_file()
        email_val = os.environ.get("SMTP_EMAIL", "").strip()
        pass_val = _normalize_smtp_password(os.environ.get("SMTP_PASSWORD", ""), SMTP_SERVER)
    return email_val, pass_val


SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_EMAIL, SMTP_PASSWORD = _get_smtp_credentials()
OTP_EXPIRY_SECONDS = 120
OTP_RESEND_COOLDOWN_SECONDS = 60


def _send_smtp_message(msg) -> None:
    """Send a MIME email via STARTTLS (587) with automatic SSL (465) fallback."""
    smtp_email, smtp_password = _get_smtp_credentials()
    if not smtp_email or not smtp_password:
        raise RuntimeError("SMTP credentials are not configured.")

    if not msg["From"]:
        msg["From"] = smtp_email

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=12) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(smtp_email, smtp_password)
            server.send_message(msg)
            print("[SMTP] Successfully delivered email via port 587 (STARTTLS).")
    except Exception as _first_err:
        print(f"[SMTP] Port 587 attempt failed ({_first_err}) — falling back to port 465 (SSL)...")
        with smtplib.SMTP_SSL(SMTP_SERVER, 465, timeout=15) as server:
            server.ehlo()
            server.login(smtp_email, smtp_password)
            server.send_message(msg)
            print("[SMTP] Successfully delivered email via port 465 (SSL).")

# Admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# ──────────────────────────────────────────────
# Background Scheduler (voting-start auto-notify)
# ──────────────────────────────────────────────
_scheduler = BackgroundScheduler(daemon=True)
_scheduler.start()

# ──────────────────────────────────────────────
# Blockchain Instance
# ──────────────────────────────────────────────
blockchain = Blockchain()

# ──────────────────────────────────────────────
# Database Setup
# ──────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")


def get_db():
    """Get a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
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
            otp_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            used INTEGER DEFAULT 0,
            attempts INTEGER DEFAULT 0,
            resent_at TIMESTAMP
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

    # Migrate: add voting_start / voting_end / client_tz_offset columns for time-period control
    for col in ("voting_start", "voting_end"):
        try:
            cursor.execute(f"ALTER TABLE elections ADD COLUMN {col} TEXT")
            conn.commit()
        except Exception:
            pass  # Column already exists

    try:
        cursor.execute("ALTER TABLE elections ADD COLUMN client_tz_offset INTEGER DEFAULT -330")
        conn.commit()
    except Exception:
        pass

    # Create allowed_voters whitelist table (CSV import)
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS allowed_voters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            voter_name TEXT NOT NULL,
            voter_id   TEXT UNIQUE NOT NULL,
            imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # ── Migrate OTP table: rename otp → otp_hash, add attempts + resent_at ──
    # Check which columns exist in the otps table
    otp_cols = [row[1] for row in cursor.execute("PRAGMA table_info(otps)").fetchall()]

    # If old plain-text 'otp' column exists and new 'otp_hash' doesn't, migrate
    if "otp" in otp_cols and "otp_hash" not in otp_cols:
        # SQLite doesn't support RENAME COLUMN before v3.25; safest: recreate table
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS otps_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                otp_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                used INTEGER DEFAULT 0,
                attempts INTEGER DEFAULT 0,
                resent_at TIMESTAMP
            );
            INSERT INTO otps_new (id, email, otp_hash, created_at, expires_at, used)
                SELECT id, email, otp, created_at, expires_at, used FROM otps;
            DROP TABLE otps;
            ALTER TABLE otps_new RENAME TO otps;
        """)

    # Add any missing columns to the already-correct otps table
    for col_def in (
        ("attempts", "INTEGER DEFAULT 0"),
        ("resent_at", "TIMESTAMP"),
    ):
        col_name, col_type = col_def
        if col_name not in otp_cols and "otp_hash" in otp_cols:
            try:
                cursor.execute(f"ALTER TABLE otps ADD COLUMN {col_name} {col_type}")
                conn.commit()
            except Exception:
                pass  # Already exists
    # ─────────────────────────────────────────────────────────────────────────

    conn.commit()
    conn.close()


# Initialize database on startup
init_db()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def generate_otp(length=6):
    """Generate a cryptographically secure random 6-digit OTP using secrets module."""
    return "".join(secrets.choice(string.digits) for _ in range(length))


def hash_otp(otp: str) -> str:
    """Return a SHA-256 hex digest of the OTP so we never store plaintext OTPs."""
    return hashlib.sha256(otp.encode()).hexdigest()


def otp_timer_seconds(created_at: str, expires_at: str) -> Tuple[int, int]:
    """Return the OTP expiry and resend cooldown durations remaining.

    Both created_at and expires_at are stored as local-time naive datetimes,
    so we always compare against datetime.now() (local) — never UTC — to
    avoid a timezone-offset mismatch that inflates the displayed wait time.
    """
    now = datetime.now()
    expires_remaining = max(
        0,
        int((datetime.fromisoformat(expires_at) - now).total_seconds())
    )
    resend_remaining = max(
        0,
        int((
            datetime.fromisoformat(created_at)
            + timedelta(seconds=OTP_RESEND_COOLDOWN_SECONDS)
            - now
        ).total_seconds())
    )
    return expires_remaining, resend_remaining


def mask_email(email: str) -> str:
    """Return a privacy-masked version of an email address.
    E.g. 'john.doe@example.com' -> 'jo***@example.com'
    """
    if "@" not in email:
        return "***"
    local, domain = email.rsplit("@", 1)
    visible = local[:2] if len(local) >= 2 else local[:1]
    return f"{visible}***@{domain}"


def send_otp_email(email, otp):
    """Send OTP via email using Gmail SMTP."""
    smtp_email, smtp_password = _get_smtp_credentials()
    if not smtp_email or not smtp_password:
        print(f"[OTP WARNING] SMTP_EMAIL or SMTP_PASSWORD not set on server! Add them in Render Dashboard -> Environment.")
        print(f"[DEV FALLBACK] Generated OTP for {email}: {otp}")
        return True  # Dev fallback — skip email

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "🔐 SecureVote — Your OTP Verification Code"
        msg["From"] = f"SecureVote <{smtp_email}>"
        msg["To"] = email

        plain = (
            f"SecureVote Verification Code\n\n"
            f"Your one-time verification code is: {otp}\n\n"
            f"This code expires in 2 minutes. Do not share it with anyone."
        )
        html = f"""
        <html>
        <body style="font-family: 'Segoe UI', sans-serif; background: #0f0f23; color: #e0e0e0; padding: 40px;">
            <div style="max-width: 480px; margin: auto; background: #1a1a2e; border-radius: 16px; padding: 40px; border: 1px solid #16213e;">
                <h2 style="color: #7c3aed; margin-bottom: 8px;">🔐 SecureVote</h2>
                <p style="color: #a0a0b8;">Your one-time verification code:</p>
                <div style="font-size: 36px; font-weight: bold; color: #a78bfa; letter-spacing: 8px; text-align: center; padding: 24px; background: #16213e; border-radius: 12px; margin: 24px 0;">
                    {otp}
                </div>
                <p style="color: #666; font-size: 13px;">This code expires in 2 minutes. Do not share it with anyone.</p>
            </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html, "html"))

        _send_smtp_message(msg)
        print(f"[MAIL] OTP email sent to {email}")
        return True
    except smtplib.SMTPAuthenticationError as e:
        print(f"[MAIL ERROR] Gmail authentication failed: {e}. Check SMTP_EMAIL and 16-char Google App Password in .env.")
        return False
    except Exception as e:
        print(f"[MAIL ERROR] Email send error for {email}: {e}")
        return False


def send_voting_start_notification(email, voter_name, election_title):
    """Send 'Voting has started NOW!' email to a single voter."""
    smtp_email, smtp_password = _get_smtp_credentials()
    if not smtp_email or not smtp_password:
        print(f"[DEV MODE] Voting-start notification for {email}: {election_title}")
        return True

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🗳️ SecureVote — Voting is NOW OPEN: {election_title}"
        msg["From"] = f"SecureVote <{smtp_email}>"
        msg["To"] = email

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
        msg.attach(MIMEText(html, "html"))

        _send_smtp_message(msg)
        return True
    except Exception as e:
        print(f"Voting-start notification error for {email}: {e}")
        return False


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


def get_election_time_context(election) -> Tuple[datetime, str]:
    """
    Return (current_now_in_election_tz, status) where status is:
    'not_set' | 'scheduled' | 'active' | 'ended'
    
    Resolves cloud server timezone offsets (e.g. Render UTC vs voter's local IST).
    """
    if not election:
        return datetime.now(), "not_set"

    vs = election["voting_start"]
    ve = election["voting_end"]
    if not vs or not ve:
        return datetime.now(), "not_set"

    tz_offset_min = -330  # Default to IST (UTC+5:30) if unset
    try:
        if "client_tz_offset" in election.keys() and election["client_tz_offset"] is not None:
            tz_offset_min = int(election["client_tz_offset"])
    except Exception:
        pass

    # JavaScript getTimezoneOffset(): IST is -330 min, so client_tz = UTC - (-330) = UTC + 330 (+5:30)
    client_tz = timezone(timedelta(minutes=-tz_offset_min))
    current_now = datetime.now(timezone.utc).astimezone(client_tz).replace(tzinfo=None)

    try:
        start_dt = datetime.fromisoformat(vs)
        end_dt   = datetime.fromisoformat(ve)
    except Exception:
        return current_now, "not_set"

    if current_now < start_dt:
        return current_now, "scheduled"
    elif current_now > end_dt:
        return current_now, "ended"
    else:
        return current_now, "active"


def schedule_voting_start_job(election_id, start_dt, tz_offset_min=-330):
    """Schedule (or re-schedule) the auto-notify job for this election."""
    job_id = f"voting_start_{election_id}"
    if _scheduler.get_job(job_id):
        _scheduler.remove_job(job_id)

    client_tz = timezone(timedelta(minutes=-tz_offset_min))
    now_client = datetime.now(timezone.utc).astimezone(client_tz).replace(tzinfo=None)
    if start_dt > now_client:
        run_date = start_dt.replace(tzinfo=client_tz)
        _scheduler.add_job(
            _auto_notify_voting_start,
            trigger=DateTrigger(run_date=run_date),
            id=job_id,
            args=[election_id],
            replace_existing=True
        )
        print(f"[Scheduler] Voting-start notification scheduled for {start_dt.isoformat()} ({client_tz})")
    else:
        print(f"[Scheduler] start_dt is in the past — skipping auto-notify scheduling.")


def send_announcement_email(email, voter_name, election_title, start_dt, end_dt):
    """Send voting announcement email to a single voter."""
    smtp_email, smtp_password = _get_smtp_credentials()
    if not smtp_email or not smtp_password:
        print(f"[DEV MODE] Announcement for {email}: {election_title} starts {start_dt}")
        return True

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"📢 SecureVote — Voting Announcement: {election_title}"
        msg["From"] = f"SecureVote <{smtp_email}>"
        msg["To"] = email

        start_str = start_dt.strftime("%d %B %Y at %I:%M %p")
        end_str   = end_dt.strftime("%d %B %Y at %I:%M %p")

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
        msg.attach(MIMEText(html, "html"))

        _send_smtp_message(msg)
        return True
    except Exception as e:
        print(f"Announcement email error for {email}: {e}")
        return False


def send_vote_acknowledgement_email(email, voter_name, voted_at, block_hash, election_title):
    """Send a congratulations acknowledgement email to the voter after successful voting."""
    smtp_email, smtp_password = _get_smtp_credentials()
    if not smtp_email or not smtp_password:
        print(f"[DEV MODE] Vote acknowledgement for {email} — voted at {voted_at}")
        return True

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "✅ SecureVote — Successfully Voted! Thank You for Participating."
        msg["From"] = f"SecureVote <{smtp_email}>"
        msg["To"] = email

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
                    <p style="margin:0 0 12px;font-size:1.25rem;font-weight:700;
                               color:#34d399;letter-spacing:0.01em;">
                      ✅ Successful voting is done by you.
                    </p>
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
        msg.attach(MIMEText(html, "html"))

        _send_smtp_message(msg)
        return True
    except Exception as e:
        print(f"Vote acknowledgement email error for {email}: {e}")
        return False


def send_certificate_email(email: str, voter_name: str, election_title: str,
                           voted_at: str, block_hash: str, png_bytes: bytes) -> bool:
    """Email the voter's 'I Voted' certificate PNG as an attachment."""
    smtp_email, smtp_password = _get_smtp_credentials()
    if not smtp_email or not smtp_password:
        print(f"[DEV MODE] Certificate email skipped for {email} (no SMTP config).")
        return True

    try:
        # Outer container: mixed so we can embed HTML body + image attachment
        outer = MIMEMultipart("mixed")
        outer["Subject"] = "🏅 SecureVote — Your Official Voting Certificate"
        outer["From"]    = f"SecureVote <{smtp_email}>"
        outer["To"]      = email

        # ── HTML body (alternative: plain-text fallback + rich HTML) ──
        alt = MIMEMultipart("alternative")

        plain_text = (
            f"Dear {voter_name},\n\n"
            "Thank you for voting! Please find your official 'I Voted' certificate attached.\n\n"
            f"Election : {election_title}\n"
            f"Voted on : {voted_at}\n"
            f"Block Hash: {block_hash}\n\n"
            "— SecureVote"
        )
        alt.attach(MIMEText(plain_text, "plain"))

        html_body = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head><meta charset="UTF-8"></head>
        <body style="margin:0;padding:0;background:#0a0a14;font-family:'Segoe UI',Arial,sans-serif;">
          <table width="100%" cellpadding="0" cellspacing="0" border="0"
                 style="background:#0a0a14;padding:40px 16px;">
            <tr><td align="center">
              <table width="520" cellpadding="0" cellspacing="0" border="0"
                     style="max-width:520px;background:#0f0f1e;border-radius:20px;
                            border:1px solid #1e1e38;box-shadow:0 24px 64px rgba(0,0,0,.6);">

                <!-- Top accent bar -->
                <tr>
                  <td style="height:4px;background:linear-gradient(90deg,#34d399,#38bdf8,#a78bfa);
                              border-radius:20px 20px 0 0;"></td>
                </tr>

                <!-- Header -->
                <tr>
                  <td align="center" style="padding:36px 40px 20px;">
                    <div style="width:80px;height:80px;border-radius:50%;
                                background:linear-gradient(135deg,#34d399,#38bdf8);
                                display:inline-flex;align-items:center;justify-content:center;
                                box-shadow:0 0 0 12px rgba(52,211,153,.1),
                                           0 12px 32px rgba(52,211,153,.35);
                                margin-bottom:20px;">
                      <span style="font-size:36px;line-height:1;">🏅</span>
                    </div>
                    <h1 style="margin:0 0 6px;font-size:1.8rem;font-weight:800;
                               color:#fff;letter-spacing:-0.02em;">
                      Your Voting Certificate
                    </h1>
                    <p style="margin:0;font-size:0.95rem;color:#94a3b8;">
                      Official acknowledgement of your participation in<br>
                      <strong style="color:#c4b5fd;">{election_title}</strong>
                    </p>
                  </td>
                </tr>

                <!-- Body -->
                <tr>
                  <td style="padding:0 40px 20px;">
                    <p style="margin:0 0 6px;color:#e2e8f0;">
                      Dear <strong style="color:#fff;">{voter_name}</strong>,
                    </p>
                    <p style="margin:0;font-size:0.9rem;color:#94a3b8;line-height:1.65;">
                      Your official <strong style="color:#34d399;">&#8220;I Voted&#8221;</strong>
                      certificate is attached to this email as a PNG image.
                      You can save it, print it, or share it as proof of your participation.
                    </p>
                  </td>
                </tr>

                <!-- Details card -->
                <tr>
                  <td style="padding:0 40px 24px;">
                    <table width="100%" cellpadding="0" cellspacing="0" border="0"
                           style="background:linear-gradient(135deg,rgba(52,211,153,.08),rgba(56,189,248,.06));
                                  border:1px solid rgba(52,211,153,.3);border-radius:14px;padding:18px 22px;">
                      <tr>
                        <td>
                          <div style="font-size:0.65rem;font-weight:800;text-transform:uppercase;
                                      letter-spacing:2px;color:#34d399;margin-bottom:6px;">
                            &#128336;&nbsp; Voted On
                          </div>
                          <div style="font-size:1rem;font-weight:700;color:#fff;line-height:1.3;">
                            {voted_at}
                          </div>
                        </td>
                      </tr>
                    </table>
                  </td>
                </tr>

                <!-- Hash -->
                <tr>
                  <td style="padding:0 40px 32px;">
                    <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
                                letter-spacing:1.5px;color:#64748b;margin-bottom:6px;">
                      &#128279;&nbsp; Blockchain Receipt
                    </div>
                    <div style="font-family:'Courier New',monospace;font-size:0.68rem;
                                color:#a78bfa;word-break:break-all;line-height:1.55;
                                background:rgba(124,58,237,.07);
                                border:1px solid rgba(124,58,237,.2);
                                border-radius:8px;padding:10px 12px;">
                      {block_hash}
                    </div>
                  </td>
                </tr>

                <!-- Privacy pill -->
                <tr>
                  <td align="center" style="padding:0 40px 28px;">
                    <div style="display:inline-block;background:rgba(167,139,250,.07);
                                border:1px solid rgba(167,139,250,.22);border-radius:9999px;
                                padding:9px 18px;font-size:0.8rem;color:#c4b5fd;">
                      &#128274;&nbsp; Your ballot choice remains completely private.
                    </div>
                  </td>
                </tr>

                <!-- Footer -->
                <tr>
                  <td style="padding:16px 40px 28px;border-top:1px solid #1e1e38;">
                    <p style="margin:0;font-size:0.72rem;color:#475569;text-align:center;line-height:1.5;">
                      This is an automated message from <strong>SecureVote</strong>.
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
        alt.attach(MIMEText(html_body, "html"))
        outer.attach(alt)

        # ── PNG certificate attachment ──
        img_part = MIMEImage(png_bytes, _subtype="png")
        img_part.add_header(
            "Content-Disposition",
            "attachment",
            filename="SecureVote_Certificate.png"
        )
        outer.attach(img_part)

        _send_smtp_message(outer)

        return True

    except Exception as exc:
        print(f"Certificate email error for {email}: {exc}")
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
            _, voting_status = get_election_time_context(election)
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
        return jsonify({"success": False, "message": "⚠️ Could not extract face features. Please retake your photo — ensure your face is clearly visible, well-lit, and centred in the frame."}), 400

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
        "SELECT * FROM voters WHERE LOWER(email) = LOWER(?) AND UPPER(voter_id) = UPPER(?)",
        (email, voter_id)
    ).fetchone()

    if not voter:
        conn.close()
        return jsonify({"success": False, "message": "No registered voter found with these credentials."}), 404

    email = voter["email"]
    voter_id = voter["voter_id"]

    # ── Rate limiting: enforce the resend cooldown ────────────────────────────
    last_otp = conn.execute(
        "SELECT created_at FROM otps WHERE email = ? ORDER BY created_at DESC LIMIT 1",
        (email,)
    ).fetchone()
    if last_otp:
        last_sent = datetime.fromisoformat(last_otp["created_at"])
        seconds_since = (
            datetime.now() - last_sent
        ).total_seconds()
        if seconds_since < OTP_RESEND_COOLDOWN_SECONDS:
            conn.close()
            wait = max(1, ceil(OTP_RESEND_COOLDOWN_SECONDS - seconds_since))
            return jsonify({
                "success": False,
                "message": f"Please wait {wait} second(s) before requesting a new OTP."
            }), 429
    # ─────────────────────────────────────────────────────────────────────────

    # Generate a cryptographically secure OTP and store its hash (never plaintext)
    otp = generate_otp()
    otp_hash = hash_otp(otp)
    created_at = datetime.now()
    expires_at = (datetime.now() + timedelta(seconds=OTP_EXPIRY_SECONDS)).isoformat()

    # Invalidate all previous unused OTPs for this email
    conn.execute(
        "UPDATE otps SET used = 1 WHERE email = ? AND used = 0",
        (email,)
    )
    # Insert new OTP record (storing hash, not plaintext)
    conn.execute(
        "INSERT INTO otps (email, otp_hash, created_at, expires_at, attempts) VALUES (?, ?, ?, ?, 0)",
        (email, otp_hash, created_at.isoformat(), expires_at)
    )
    conn.commit()
    conn.close()

    # Send OTP via Gmail SMTP — if sending fails, do NOT pretend it succeeded
    if send_otp_email(email, otp):
        session["pending_email"] = email
        session["pending_voter_id"] = voter_id
        masked = mask_email(email)
        return jsonify({
            "success": True,
            "message": f"OTP sent to {masked}. It expires in 2 minutes.",
            "redirect": url_for("verify_otp")
        })
    else:
        # Roll back the OTP record so user can try again
        try:
            conn2 = get_db()
            conn2.execute(
                "UPDATE otps SET used = 1 WHERE email = ? AND used = 0",
                (email,)
            )
            conn2.commit()
            conn2.close()
        except Exception:
            pass
        return jsonify({
            "success": False,
            "message": "Failed to send OTP email. Please check your Gmail configuration or try again later."
        }), 500


@app.route("/resend-otp", methods=["POST"])
def resend_otp():
    """Resend OTP to the pending email (subject to 60-second cooldown)."""
    email = session.get("pending_email")
    if not email:
        return jsonify({"success": False, "message": "Session expired. Please log in again."}), 401

    conn = get_db()

    # Enforce the resend cooldown
    last_otp = conn.execute(
        "SELECT created_at FROM otps WHERE email = ? ORDER BY created_at DESC LIMIT 1",
        (email,)
    ).fetchone()
    if last_otp:
        last_sent = datetime.fromisoformat(last_otp["created_at"])
        seconds_since = (
            datetime.now() - last_sent
        ).total_seconds()
        if seconds_since < OTP_RESEND_COOLDOWN_SECONDS:
            conn.close()
            wait = max(1, ceil(OTP_RESEND_COOLDOWN_SECONDS - seconds_since))
            return jsonify({
                "success": False,
                "message": f"Please wait {wait} second(s) before requesting a new OTP.",
                "wait": wait
            }), 429

    # Generate new OTP, invalidate old ones
    otp = generate_otp()
    otp_hash = hash_otp(otp)
    created_at = datetime.now()
    expires_at = (datetime.now() + timedelta(seconds=OTP_EXPIRY_SECONDS)).isoformat()

    conn.execute("UPDATE otps SET used = 1 WHERE email = ? AND used = 0", (email,))
    conn.execute(
        "INSERT INTO otps (email, otp_hash, created_at, expires_at, attempts) VALUES (?, ?, ?, ?, 0)",
        (email, otp_hash, created_at.isoformat(), expires_at)
    )
    conn.commit()
    conn.close()

    if send_otp_email(email, otp):
        masked = mask_email(email)
        expires_in, resend_in = otp_timer_seconds(created_at.isoformat(), expires_at)
        return jsonify({
            "success": True,
            "message": f"New OTP sent to {masked}. It expires in 2 minutes.",
            "expires_in": expires_in,
            "resend_in": resend_in
        })
    else:
        return jsonify({
            "success": False,
            "message": "Failed to resend OTP. Please check your Gmail configuration."
        }), 500


@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp():
    """OTP verification page — validates hashed OTP with expiry and attempt limits."""
    if "pending_email" not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for("login"))

    email = session.get("pending_email")

    if request.method == "GET":
        masked = mask_email(email)
        conn = get_db()
        record = conn.execute(
            "SELECT created_at, expires_at FROM otps WHERE email = ? AND used = 0 ORDER BY id DESC LIMIT 1",
            (email,)
        ).fetchone()
        conn.close()
        otp_remaining, resend_remaining = (
            otp_timer_seconds(record["created_at"], record["expires_at"])
            if record else (0, 0)
        )
        return render_template(
            "verify_otp.html",
            email=email,
            masked_email=masked,
            otp_remaining_seconds=otp_remaining,
            resend_remaining_seconds=resend_remaining
        )

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "Invalid request"}), 400

    entered_otp = data.get("otp", "").strip()

    if not entered_otp or len(entered_otp) != 6 or not entered_otp.isdigit():
        return jsonify({"success": False, "message": "Please enter a valid 6-digit OTP."}), 400

    conn = get_db()

    # Fetch the latest unused OTP record for this email
    record = conn.execute(
        "SELECT * FROM otps WHERE email = ? AND used = 0 ORDER BY created_at DESC LIMIT 1",
        (email,)
    ).fetchone()

    if not record:
        conn.close()
        return jsonify({"success": False, "message": "No active OTP found. Please request a new one."}), 400

    # ── Check attempt limit (max 5 wrong tries) ──────────────────────────────
    MAX_ATTEMPTS = 5
    current_attempts = record["attempts"] if record["attempts"] is not None else 0
    if current_attempts >= MAX_ATTEMPTS:
        conn.execute("UPDATE otps SET used = 1 WHERE id = ?", (record["id"],))
        conn.commit()
        conn.close()
        return jsonify({
            "success": False,
            "message": "Too many incorrect attempts. This OTP has been invalidated. Please request a new one."
        }), 429
    # ─────────────────────────────────────────────────────────────────────────

    # ── Verify OTP hash — never compare plaintext ────────────────────────────
    entered_hash = hash_otp(entered_otp)
    if entered_hash != record["otp_hash"]:
        # Increment attempt counter
        new_attempts = current_attempts + 1
        conn.execute("UPDATE otps SET attempts = ? WHERE id = ?", (new_attempts, record["id"]))
        conn.commit()
        conn.close()
        remaining = MAX_ATTEMPTS - new_attempts
        if remaining <= 0:
            return jsonify({
                "success": False,
                "message": "Incorrect OTP. No attempts remaining. Please request a new OTP."
            }), 400
        return jsonify({
            "success": False,
            "message": f"Incorrect OTP. {remaining} attempt(s) remaining."
        }), 400
    # ─────────────────────────────────────────────────────────────────────────

    # ── Check OTP expiry ─────────────────────────────────────────────────────
    expires_at = datetime.fromisoformat(record["expires_at"])
    if datetime.now() > expires_at:
        conn.execute("UPDATE otps SET used = 1 WHERE id = ?", (record["id"],))
        conn.commit()
        conn.close()
        return jsonify({"success": False, "message": "OTP has expired. Please request a new one."}), 400
    # ─────────────────────────────────────────────────────────────────────────

    # ── OTP is valid — mark as used so it cannot be reused ──────────────────
    conn.execute("UPDATE otps SET used = 1 WHERE id = ?", (record["id"],))
    conn.commit()
    conn.close()

    session["otp_verified"] = True
    return jsonify({"success": True, "message": "OTP verified successfully!", "redirect": url_for("verify_face")})


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
        _, v_status = get_election_time_context(election)
        if v_status == "scheduled":
            conn.close()
            return redirect(url_for("voting_closed",
                                    reason="not_started",
                                    start=election["voting_start"],
                                    end=election["voting_end"]))
        if v_status == "ended":
            conn.close()
            return redirect(url_for("voting_closed",
                                    reason="ended",
                                    start=election["voting_start"],
                                    end=election["voting_end"]))
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
    _, v_status_post = get_election_time_context(election)
    if v_status_post == "scheduled":
        conn.close()
        return jsonify({"success": False, "message": "Voting has not started yet."}), 403
    if v_status_post == "ended":
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
    session["voted_voter_name"] = voter["name"]
    session["voted_election_title"] = election["title"]
    # Store voter_email in session so vote_success page and certificate email endpoint can use it
    session["voter_email"] = email

    # Send "Successful voting is done by you." acknowledgement email to the voter (in background thread)
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
    """Vote confirmation page — shows congratulations with timestamp, certificate, and email notice."""
    block_hash     = session.get("vote_block_hash", "N/A")
    voted_at       = session.get("voted_at", datetime.now().strftime("%A, %d %B %Y at %I:%M:%S %p"))
    voter_name     = session.get("voted_voter_name", "Voter")
    election_title = session.get("voted_election_title", "General Election")
    voter_email    = session.get("voter_email", "")
    masked_email   = mask_email(voter_email) if voter_email else "your registered email"
    # Acknowledgement flash message shown via the general alert template
    flash("Successfully voted.", "success")
    # Deliberately do NOT pass candidate name — the page must not reveal who was voted for
    return render_template(
        "success.html",
        block_hash=block_hash,
        voted_at=voted_at,
        voter_name=voter_name,
        election_title=election_title,
        masked_email=masked_email,
    )


@app.route("/already-voted")
def already_voted():
    """Page shown when voter has already voted."""
    return render_template("already_voted.html")


@app.route("/api/email-certificate", methods=["POST"])
def api_email_certificate():
    """Receive the canvas PNG (base64) from the client and email it as an attachment.

    NOTE: intentionally NOT decorated with @login_required because that decorator
    issues an HTML redirect on session-miss, which breaks the JSON fetch call.
    Session validation is handled inline below, returning proper JSON errors.
    """
    # ── Session check (returns JSON, never a redirect) ──
    voter_email = session.get("voter_email", "")
    if not voter_email:
        return jsonify({
            "success": False,
            "message": "Session expired. Please log in again to receive the certificate."
        }), 401

    data = request.get_json(silent=True)
    if not data or "image_data" not in data:
        return jsonify({"success": False, "message": "No image data provided."}), 400

    # Strip the data-URL prefix produced by canvas.toDataURL()
    img_b64 = data["image_data"]
    if img_b64.startswith("data:image/png;base64,"):
        img_b64 = img_b64[len("data:image/png;base64,"):]

    try:
        png_bytes = base64.b64decode(img_b64)
    except Exception:
        return jsonify({"success": False, "message": "Invalid image data."}), 400

    voter_name     = session.get("voted_voter_name",     "Voter")
    election_title = session.get("voted_election_title", "General Election")
    voted_at       = session.get("voted_at",             "")
    block_hash     = session.get("vote_block_hash",      "")

    ok = send_certificate_email(
        voter_email, voter_name, election_title, voted_at, block_hash, png_bytes
    )
    if ok:
        return jsonify({"success": True,  "message": "Certificate emailed successfully!"})
    return jsonify({"success": False, "message": "Failed to send certificate email."}), 500


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
    voting_status = "not_set"  # not_set | scheduled | active | ended

    if election:
        candidates = conn.execute(
            "SELECT * FROM candidates WHERE election_id = ?", (election["id"],)
        ).fetchall()
        vote_counts = blockchain.get_vote_counts(election["id"])

        vs = election["voting_start"]
        ve = election["voting_end"]
        if vs and ve:
            _, voting_status = get_election_time_context(election)
        else:
            voting_status = "not_set"

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
    try:
        tz_offset_min = int(request.form.get("client_tz_offset", -330))
    except Exception:
        tz_offset_min = -330
    client_tz = timezone(timedelta(minutes=-tz_offset_min))
    now_client = datetime.now(timezone.utc).astimezone(client_tz).replace(tzinfo=None)

    if action == "clear":
        conn.execute(
            "UPDATE elections SET voting_start = NULL, voting_end = NULL, client_tz_offset = ? WHERE id = ?",
            (tz_offset_min, election["id"])
        )
        conn.commit()
        conn.close()
        flash("Voting time period cleared — voting is now UNRESTRICTED (anyone can vote anytime).", "warning")
        return redirect(url_for("admin_dashboard"))

    if action == "go_live":
        # Immediately open voting for 24 hours from right now in client timezone
        start_dt = now_client - timedelta(minutes=2)
        end_dt   = now_client + timedelta(hours=24)
        conn.execute(
            "UPDATE elections SET voting_start = ?, voting_end = ?, client_tz_offset = ? WHERE id = ?",
            (start_dt.isoformat(timespec="minutes"), end_dt.isoformat(timespec="minutes"), tz_offset_min, election["id"])
        )
        conn.commit()
        conn.close()
        flash(f"⚡ Voting has been switched to LIVE immediately! (Open until {end_dt.strftime('%d %b %Y %H:%M')})", "success")
        return redirect(url_for("admin_dashboard"))

    if action == "end_now":
        # Immediately end voting (set end to 1 minute in the past)
        start_val = election["voting_start"] or (now_client - timedelta(hours=1)).isoformat(timespec="minutes")
        end_val   = (now_client - timedelta(minutes=1)).isoformat(timespec="minutes")
        conn.execute(
            "UPDATE elections SET voting_start = ?, voting_end = ?, client_tz_offset = ? WHERE id = ?",
            (start_val, end_val, tz_offset_min, election["id"])
        )
        conn.commit()
        conn.close()
        flash("🛑 Voting officially CLOSED. The election has concluded and polls are locked.", "warning")
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
        "UPDATE elections SET voting_start = ?, voting_end = ?, client_tz_offset = ? WHERE id = ?",
        (start_dt.isoformat(timespec="minutes"), end_dt.isoformat(timespec="minutes"), tz_offset_min, election["id"])
    )
    conn.commit()

    # ── Schedule auto-notify job for voting start ─────────────────────────────
    schedule_voting_start_job(election["id"], start_dt, tz_offset_min)
    # ─────────────────────────────────────────────────────────────────────────

    # ── Optionally send announcement emails in background ────────────────────
    notify = request.form.get("notify_voters", "")
    if notify == "1":
        voters = conn.execute("SELECT name, email FROM voters").fetchall()
        election_title = election["title"]
        import threading
        def _bg_notify(vlist, title, s_dt, e_dt):
            for v in vlist:
                send_announcement_email(v["email"], v["name"], title, s_dt, e_dt)
        threading.Thread(target=_bg_notify, args=(voters, election_title, start_dt, end_dt), daemon=True).start()
        flash(f"Voting period set! Announcement emails are being sent to {len(voters)} voter(s).", "success")
    else:
        if now_client < start_dt:
            st_text = "Scheduled — Opens at " + start_dt.strftime('%d %b %Y %H:%M')
        elif now_client > end_dt:
            st_text = "Ended"
        else:
            st_text = "LIVE NOW"
        flash(
            f"Voting period updated: {start_dt.strftime('%d %b %Y %H:%M')} → {end_dt.strftime('%d %b %Y %H:%M')} ({st_text}).",
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

    from flask import Response
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
# Face Detection API — Human Presence Check + Bounding Box
# ──────────────────────────────────────────────
@app.route("/detect_face", methods=["POST"])
def detect_face():
    """
    Lightweight endpoint: accepts a single base64-encoded frame and returns
    whether a human face is present, plus the bounding box (x,y,w,h) so the
    frontend can draw a real-time face-lock overlay on the camera feed.
    """
    data = request.get_json(silent=True) or {}
    frame_b64 = data.get("face_image", "")

    if not frame_b64:
        return jsonify({"face_detected": False, "message": "No image data provided."}), 400

    image = decode_base64_image(frame_b64)
    if image is None:
        return jsonify({"face_detected": False, "message": "Could not decode image."}), 400

    # Use detect_face_core to also get the bounding box for overlay drawing
    bgr_img, face_row = detect_face_core(image)

    if face_row is None:
        return jsonify({
            "face_detected": False,
            "face_box": None,
            "message": "No human face detected. Please face the camera directly."
        })

    # Return bounding box as {x, y, w, h} integers for JS overlay drawing
    x, y, w, h = [max(0, int(v)) for v in face_row[:4]]
    return jsonify({
        "face_detected": True,
        "face_box": {"x": x, "y": y, "w": w, "h": h},
        "message": "Human face detected."
    })


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
def main():
    _smtp_email, _smtp_pass = _get_smtp_credentials()
    print("\n" + "=" * 60)
    print("  SecureVote - Online Voting System")
    print("  Face Recognition + Liveness + Blockchain")
    if _smtp_email and _smtp_pass:
        print(f"  Gmail SMTP: Active ({_smtp_email})")
    else:
        print("  Gmail SMTP: DEV MODE (set SMTP_EMAIL & SMTP_PASSWORD in .env)")
    port = int(os.environ.get("PORT", 5000))
    print(f"  http://0.0.0.0:{port}")
    print("=" * 60 + "\n")

    # In production on Linux / Render, launch high-performance Gunicorn WSGI server
    is_prod_linux = sys.platform.startswith("linux") or os.environ.get("RENDER") is not None
    if is_prod_linux:
        try:
            import subprocess
            cmd = [
                sys.executable, "-m", "gunicorn",
                "--workers=1",
                "--threads=4",
                "--timeout=120",
                "--bind", f"0.0.0.0:{port}",
                "app:app"
            ]
            print(f"[Production] Launching Gunicorn WSGI server on port {port}...")
            sys.exit(subprocess.call(cmd))
        except Exception as _gerr:
            print(f"[WSGI Fallback] Gunicorn failed ({_gerr}), falling back to app.run()")

    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1")
    app.run(debug=debug_mode, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
