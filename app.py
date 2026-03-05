from dotenv import load_dotenv
load_dotenv()
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import re
import time
import threading
import io
import random
import datetime
import uuid
import jwt
import json
import secrets
from functools import wraps
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, redirect, url_for, session
from werkzeug.utils import secure_filename
import whisper
from deep_translator import GoogleTranslator
from passlib.context import CryptContext

# --- REPORTLAB IMPORTS ---
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont   # handles CJK with zero font files

# --- TORCH AND TRANSFORMERS IMPORTS (For BART Summarization) ---
import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# --- GROQ API IMPORT ---
from groq import Groq

# --- EMAIL IMPORTS ---
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- GOOGLE OAUTH IMPORTS ---
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from google_auth_oauthlib.flow import Flow

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "transcribe_flow_secret_key_change_in_production")

# ============================================================
# GROQ CONFIGURATION
# ============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ============================================================
# EMAIL CONFIGURATION
# ============================================================
SMTP_HOST     = os.getenv("SMTP_HOST")
SMTP_PORT     = int(os.getenv("SMTP_PORT", 587))
SMTP_USER     = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM     = os.getenv("SMTP_FROM", SMTP_USER)
APP_BASE_URL  = os.getenv("APP_BASE_URL", "http://127.0.0.1:5000")

# ============================================================
# GOOGLE OAUTH CONFIGURATION
# ============================================================
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI", "http://127.0.0.1:5000/auth/google/callback")

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# ============================================================
# AUTH CONFIGURATION
# ============================================================
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
ALGORITHM = "HS256"
DB_FILE = "users.json"

reset_tokens = {}
otp_store = {}

# ============================================================
# DATABASE HELPERS
# ============================================================
def load_users():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading database: {e}")
            return {}
    return {}

def save_users():
    try:
        with open(DB_FILE, 'w') as f:
            json.dump(users_db, f, indent=4)
    except Exception as e:
        print(f"Error saving database: {e}")

users_db = load_users()
print(f"Database loaded. {len(users_db)} users found.")

# ============================================================
# UPLOAD CONFIG
# ============================================================
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXTENSIONS = {'mp3', 'wav'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

processing_jobs = {}

# ============================================================
# MODELS LOADING
# ============================================================
print("Loading Whisper Model...")
try:
    model = whisper.load_model("base")
    print("Whisper Model Loaded Successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load Whisper model. Details: {e}")

print("Loading BART Summarization Model...")
summ_model = None
summ_tokenizer = None
try:
    summ_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    summ_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    print("BART Model Loaded Successfully.")
except Exception as e:
    print(f"WARNING: Could not load BART model. Details: {e}")

# ============================================================
# USER CLASS
# ============================================================
class User:
    def __init__(self, name, email, phone=""):
        self.user_id = str(uuid.uuid4())
        self.name = name
        self.email = email
        self.phone = phone
        self.created_at = datetime.datetime.now()

    def register(self):
        return {
            "user_id": self.user_id,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "registered_on": self.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }

# ============================================================
# AUTH HELPER FUNCTIONS
# ============================================================
def hash_password(password):
    return pwd_context.hash(password)

def verify_password(password, hashed):
    return pwd_context.verify(password, hashed)

def create_token(email):
    payload = {
        "sub": email,
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=6)
    }
    return jwt.encode(payload, app.secret_key, algorithm=ALGORITHM)

def validate_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r'[^a-zA-Z0-9]', password):
        return False, "Password must contain at least one special character (e.g. @, #, $, !)."
    return True, ""

def normalise_phone(phone: str) -> str:
    digits = re.sub(r'[\s\-\(\)]', '', phone)
    return digits

def find_user(identifier: str):
    if identifier in users_db:
        return identifier, users_db[identifier]
    norm = normalise_phone(identifier)
    for email_key, record in users_db.items():
        stored_phone = normalise_phone(record.get("phone", ""))
        if stored_phone and stored_phone == norm:
            return email_key, record
    return None, None

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if "Authorization" in request.headers:
            auth_header = request.headers["Authorization"]
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
        if not token:
            return jsonify({"detail": "Token is missing!"}), 401
        try:
            data = jwt.decode(token, app.secret_key, algorithms=[ALGORITHM])
            current_user = users_db.get(data["sub"])
            if not current_user:
                return jsonify({"detail": "User not found!"}), 401
        except Exception as e:
            return jsonify({"detail": "Token is invalid or expired!"}), 401
        return f(current_user, *args, **kwargs)
    return decorated

# ============================================================
# EMAIL SENDING FUNCTIONS
# ============================================================
def send_reset_email(to_email: str, token: str) -> bool:
    reset_link = f"{APP_BASE_URL}/auth/reset_password_page?token={token}"
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "TranscribeFlow – Password Reset Request"
    msg["From"] = SMTP_FROM
    msg["To"] = to_email

    text_body = f"""Hi,

You requested a password reset for your TranscribeFlow account.
Click the link below within 30 minutes to reset your password:

{reset_link}

If you did not request this, please ignore this email.

— TranscribeFlow Team
"""
    html_body = f"""
<html><body style="font-family:sans-serif; background:#0a0a0a; color:#fff; padding:40px;">
  <h2 style="color:#00ffff;">TranscribeFlow – Password Reset</h2>
  <p>You requested a password reset. Click the button below (valid for 30 minutes):</p>
  <a href="{reset_link}"
     style="display:inline-block; margin:20px 0; padding:14px 30px; background:#00ffff;
            color:#000; font-weight:800; border-radius:50px; text-decoration:none;">
    Reset Password
  </a>
  <p style="opacity:.5; font-size:12px;">If you didn't request this, ignore this email.</p>
</body></html>
"""
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM, to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"Email send error: {e}")
        return False

def send_otp_email(to_email: str, otp: str) -> bool:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "TranscribeFlow – Your Login OTP"
    msg["From"] = SMTP_FROM
    msg["To"] = to_email

    text_body = f"""Hi,

Your OTP for logging into TranscribeFlow is:

{otp}

This code is valid for 5 minutes. Do not share it with anyone.

— TranscribeFlow Team
"""
    html_body = f"""
<html><body style="font-family:sans-serif; background:#0a0a0a; color:#fff; padding:40px; text-align:center;">
  <h2 style="color:#00ffff;">TranscribeFlow Login OTP</h2>
  <p>Enter this code to log in:</p>
  <div style="font-size:32px; font-weight:900; color:#00ffff; background:rgba(0,255,255,0.1);
              padding:20px; border-radius:15px; display:inline-block; margin:20px 0; letter-spacing:8px;">
    {otp}
  </div>
  <p style="opacity:.6; font-size:13px;">Valid for 5 minutes. Don't share this code.</p>
</body></html>
"""
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM, to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"OTP Email send error: {e}")
        return False

# ============================================================
# AUTH ROUTES
# ============================================================

@app.route('/auth/register/request_otp', methods=['POST'])
def register_request_otp():
    data = request.get_json()
    if not data or 'email' not in data:
        return jsonify({"detail": "Missing email address"}), 400

    email = data['email'].strip().lower()
    if email in users_db:
        return jsonify({"detail": "This email is already registered. Please login instead."}), 400

    otp = str(random.randint(100000, 999999))
    otp_store[email] = {
        "otp": otp,
        "expires": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=10),
        "verified": False
    }

    sent = send_otp_email(email, otp)
    if not sent:
        otp_store.pop(email, None)
        return jsonify({"detail": "Could not send OTP. Please check SMTP configuration."}), 500

    return jsonify({"message": "OTP sent to your email. Please verify to continue registration."})


@app.route('/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or 'email' not in data or 'password' not in data or 'name' not in data or 'otp' not in data:
        return jsonify({"detail": "Missing required fields (name, email, password, otp)"}), 400

    name     = data['name'].strip()
    email    = data['email'].strip().lower()
    password = data['password']
    phone    = normalise_phone(data.get('phone', ''))
    otp      = data['otp'].strip()

    otp_data = otp_store.get(email)
    if not otp_data:
        return jsonify({"detail": "No OTP found. Please request a new one."}), 400

    if datetime.datetime.now(datetime.timezone.utc) > otp_data['expires']:
        otp_store.pop(email, None)
        return jsonify({"detail": "OTP has expired. Please request a new one."}), 400

    if otp_data['otp'] != otp:
        return jsonify({"detail": "Incorrect OTP. Please try again."}), 401

    is_valid, err_msg = validate_password(password)
    if not is_valid:
        return jsonify({"detail": err_msg}), 400

    if email in users_db:
        return jsonify({"detail": "User already exists with this email."}), 400

    if phone:
        for record in users_db.values():
            if normalise_phone(record.get("phone", "")) == phone:
                return jsonify({"detail": "Phone number already registered."}), 400

    new_user_obj = User(name, email, phone)
    user_details = new_user_obj.register()
    user_details['password_hash'] = hash_password(password)

    users_db[email] = user_details
    save_users()
    otp_store.pop(email, None)

    return jsonify({
        "message": "Account created successfully! You can now log in.",
        "user_details": {
            "user_id":       user_details['user_id'],
            "name":          user_details['name'],
            "email":         user_details['email'],
            "phone":         user_details['phone'],
            "registered_on": user_details['registered_on']
        }
    })


@app.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or 'identifier' not in data or 'password' not in data:
        return jsonify({"detail": "Missing identifier (email/phone) or password"}), 400

    identifier = data['identifier'].strip()
    password   = data['password']

    email_key, user_record = find_user(identifier)
    if not user_record or not verify_password(password, user_record['password_hash']):
        return jsonify({"detail": "Invalid credentials"}), 401

    if not user_record.get('email'):
        return jsonify({
            "detail": "Your account needs to be updated with an email address. Please contact support."
        }), 403

    return jsonify({
        "access_token": create_token(email_key),
        "user_id":      user_record['user_id'],
        "name":         user_record['name']
    })


@app.route('/auth/request_otp', methods=['POST'])
def request_otp():
    data = request.get_json()
    if not data or 'email' not in data:
        return jsonify({"detail": "Missing email address"}), 400

    email = data['email'].strip().lower()
    if email not in users_db:
        return jsonify({"message": "If that email is registered, an OTP has been sent."})

    otp = str(random.randint(100000, 999999))
    otp_store[email] = {
        "otp": otp,
        "expires": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5)
    }

    sent = send_otp_email(email, otp)
    if not sent:
        otp_store.pop(email, None)
        return jsonify({"detail": "Could not send OTP. Please check SMTP configuration."}), 500

    return jsonify({"message": "If that email is registered, an OTP has been sent."})


@app.route('/auth/verify_otp', methods=['POST'])
def verify_otp():
    data = request.get_json()
    if not data or 'email' not in data or 'otp' not in data:
        return jsonify({"detail": "Missing email or OTP"}), 400

    email = data['email'].strip().lower()
    otp   = data['otp'].strip()

    otp_data = otp_store.get(email)
    if not otp_data:
        return jsonify({"detail": "Invalid or expired OTP."}), 400

    if datetime.datetime.now(datetime.timezone.utc) > otp_data['expires']:
        otp_store.pop(email, None)
        return jsonify({"detail": "OTP has expired. Please request a new one."}), 400

    if otp_data['otp'] != otp:
        return jsonify({"detail": "Incorrect OTP."}), 401

    otp_store.pop(email, None)
    if email not in users_db:
        return jsonify({"detail": "User not found."}), 404

    user_record = users_db[email]
    return jsonify({
        "access_token": create_token(email),
        "user_id":      user_record['user_id'],
        "name":         user_record['name']
    })


@app.route('/auth/google/login')
def google_login():
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return jsonify({"detail": "Google OAuth not configured"}), 500

    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [GOOGLE_REDIRECT_URI]
            }
        },
        scopes=["openid", "https://www.googleapis.com/auth/userinfo.email",
                "https://www.googleapis.com/auth/userinfo.profile"]
    )
    flow.redirect_uri = GOOGLE_REDIRECT_URI
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true'
    )
    session['state'] = state
    return redirect(authorization_url)


@app.route('/auth/google/callback')
def google_callback():
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return "Google OAuth not configured", 500

    state = session.get('state')
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [GOOGLE_REDIRECT_URI]
            }
        },
        scopes=["openid", "https://www.googleapis.com/auth/userinfo.email",
                "https://www.googleapis.com/auth/userinfo.profile"],
        state=state
    )
    flow.redirect_uri = GOOGLE_REDIRECT_URI
    flow.fetch_token(authorization_response=request.url)

    credentials = flow.credentials
    request_session = google_requests.Request()
    id_info = id_token.verify_oauth2_token(
        credentials.id_token, request_session, GOOGLE_CLIENT_ID, clock_skew_in_seconds=10
    )

    email = id_info.get('email')
    name  = id_info.get('name', email.split('@')[0])
    if not email:
        return "Could not retrieve email from Google", 400

    email = email.lower()
    if email not in users_db:
        new_user_obj = User(name, email, phone="")
        user_details = new_user_obj.register()
        user_details['password_hash'] = ""
        users_db[email] = user_details
        save_users()

    user_record = users_db[email]
    token = create_token(email)
    return redirect(f"/?google_login=success&token={token}&name={user_record['name']}&user_id={user_record['user_id']}")


@app.route('/auth/forgot_password', methods=['POST'])
def forgot_password():
    data = request.get_json()
    if not data or 'email' not in data:
        return jsonify({"detail": "Missing email address"}), 400

    email = data['email'].strip().lower()
    if email not in users_db:
        return jsonify({"message": "If that email is registered, a reset link has been sent."})

    token = secrets.token_urlsafe(48)
    reset_tokens[token] = {
        "email":   email,
        "expires": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=30)
    }

    sent = send_reset_email(email, token)
    if not sent:
        reset_tokens.pop(token, None)
        return jsonify({"detail": "Could not send reset email. Please check SMTP configuration."}), 500

    return jsonify({"message": "If that email is registered, a reset link has been sent."})


@app.route('/auth/reset_password_page', methods=['GET'])
def reset_password_page():
    token = request.args.get('token', '')
    return render_template('reset_password.html', token=token)


@app.route('/auth/reset_password', methods=['POST'])
def reset_password():
    data = request.get_json()
    if not data or 'token' not in data or 'new_password' not in data:
        return jsonify({"detail": "Missing token or new_password"}), 400

    token        = data['token']
    new_password = data['new_password']

    token_data = reset_tokens.get(token)
    if not token_data:
        return jsonify({"detail": "Invalid or expired reset token."}), 400

    if datetime.datetime.now(datetime.timezone.utc) > token_data['expires']:
        reset_tokens.pop(token, None)
        return jsonify({"detail": "Reset token has expired. Please request a new one."}), 400

    is_valid, err_msg = validate_password(new_password)
    if not is_valid:
        return jsonify({"detail": err_msg}), 400

    email = token_data['email']
    if email not in users_db:
        return jsonify({"detail": "User not found."}), 404

    users_db[email]['password_hash'] = hash_password(new_password)
    save_users()
    reset_tokens.pop(token, None)

    return jsonify({"message": "Password reset successfully! You can now log in."})


# ============================================================
# TRANSCRIPTION & APP FUNCTIONALITY
# ============================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_timestamp(seconds):
    td = time.gmtime(seconds)
    return time.strftime("%H:%M:%S", td)

def summarize_chunk(text_chunk):
    if not summ_model or not summ_tokenizer:
        return text_chunk
    try:
        input_ids = summ_tokenizer.encode(text_chunk, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = summ_model.generate(
            input_ids, max_length=150, min_length=40, num_beams=4,
            length_penalty=2.0, early_stopping=True, no_repeat_ngram_size=3
        )
        return summ_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Chunk summarization error: {e}")
        return text_chunk

def run_transcription(file_path, filename):
    print(f"Starting transcription for: {filename}")
    try:
        processing_jobs[filename] = {'status': 'processing', 'progress': 10}

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found at {file_path}")

        result = model.transcribe(file_path, verbose=True, fp16=False)
        processing_jobs[filename]['progress'] = 60

        full_text = result['text'].strip()
        segments  = result.get('segments', [])
        formatted_transcript = ""

        for s in segments:
            start = format_timestamp(s['start'])
            end   = format_timestamp(s['end'])
            formatted_transcript += f"[{start} - {end}] {s['text'].strip()}\n"

        processing_jobs[filename]['progress'] = 75
        summary_text = ""

        if len(full_text) > 50:
            chunk_size = 3000
            chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
            summarized_chunks = []
            for chunk in chunks:
                if len(chunk.strip()) > 30:
                    clean_summary = summarize_chunk(chunk)
                    if clean_summary and clean_summary.lower() not in [s.lower() for s in summarized_chunks]:
                        summarized_chunks.append(clean_summary)
            summary_text = " ".join(summarized_chunks)
        else:
            summary_text = summarize_chunk(full_text)

        processing_jobs[filename]['progress'] = 85

        transcript_path = os.path.join(app.config['UPLOAD_FOLDER'], filename + ".txt")
        summary_path    = os.path.join(app.config['UPLOAD_FOLDER'], filename + "_summary.txt")

        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(formatted_transcript)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        processing_jobs[filename].update({
            'status': 'completed', 'progress': 100,
            'transcript': formatted_transcript, 'summary': summary_text
        })
        print(f"Transcription completed for: {filename}")

    except Exception as e:
        print(f"ERROR in run_transcription for {filename}: {e}")
        processing_jobs[filename] = {'status': 'error', 'message': str(e), 'progress': 0}


@app.route('/')
def index():
    files = []
    if os.path.exists(UPLOAD_FOLDER):
        for f in os.listdir(UPLOAD_FOLDER):
            if allowed_file(f):
                path = os.path.join(UPLOAD_FOLDER, f)
                file_size_bytes = os.path.getsize(path)
                if file_size_bytes >= 1024 * 1024:
                    size_str = f"{file_size_bytes / (1024 * 1024):.1f} MB"
                else:
                    size_str = f"{file_size_bytes / 1024:.1f} KB"
                file_type = f.rsplit('.', 1)[1].upper() if '.' in f else 'UNKNOWN'
                raw_time = os.path.getmtime(path)
                files.append({
                    'name': f,
                    'time': time.ctime(raw_time),
                    'timestamp': raw_time,
                    'size': file_size_bytes,
                    'size_str': size_str,
                    'type': file_type
                })
    files.sort(key=lambda x: x['timestamp'], reverse=True)
    return render_template('index.html', files=files)


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio part in request'}), 400
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename  = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            threading.Thread(target=run_transcription, args=(file_path, filename)).start()
            return jsonify({'message': 'Upload successful', 'filename': filename})
        else:
            return jsonify({'error': 'Invalid file type. Allowed: mp3, wav'}), 400
    except Exception as e:
        print(f"UPLOAD ROUTE ERROR: {e}")
        return jsonify({'error': f"Server Error: {str(e)}"}), 500


@app.route('/check_status/<filename>')
def check_status(filename):
    status_data = processing_jobs.get(filename, {'status': 'initializing', 'progress': 5})
    return jsonify(status_data)


@app.route('/translate_on_fly', methods=['POST'])
def translate_on_fly():
    data        = request.get_json()
    transcript  = data.get('transcript', '')
    summary     = data.get('summary', '')
    target_lang = data.get('target', 'en')
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        translated_text = ""
        for chunk in [transcript[i:i+4500] for i in range(0, len(transcript), 4500)]:
            translated_text += translator.translate(chunk)
        translated_summary = translator.translate(summary) if summary else ""
        return jsonify({'success': True, 'translated_text': translated_text, 'translated_summary': translated_summary})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# MULTILINGUAL PDF GENERATION
# ============================================================

# Directory where TTF font files live (Arabic, Hindi, Latin only)
# CJK languages use ReportLab's built-in CID fonts — no font files needed.
FONTS_DIR = os.path.join(os.getcwd(), 'fonts')

# ── CID fonts (built into every ReportLab installation) ──────────────────────
# These are PDF standard CJK fonts. Zero external files required.
# The PDF viewer on the end-user's machine supplies the actual glyphs,
# which is why characters render correctly.
CID_FONT_MAP = {
    'ja':    'HeiseiKakuGo-W5',   # Japanese
    'zh':    'STSong-Light',       # Chinese Simplified
    'zh-cn': 'STSong-Light',       # Chinese Simplified
    'zh-tw': 'STSong-Light',       # Chinese Traditional (best available built-in)
    'ko':    'HYGothic-Medium',    # Korean
}

# ── TTF fonts (files must exist in FONTS_DIR) ─────────────────────────────────
TTF_FONT_MAP = {
    'NotoSansArabic':     'NotoSansArabic-Regular.ttf',
    'NotoSansDevanagari': 'NotoSansDevanagari-Regular.ttf',
    'NotoSans':           'NotoSans-Regular.ttf',
}

_cid_registered = set()
_ttf_registered = set()


def _register_cid(cid_name: str) -> bool:
    """Register a CID font with pdfmetrics once. Returns True on success."""
    if cid_name in _cid_registered:
        return True
    try:
        pdfmetrics.registerFont(UnicodeCIDFont(cid_name))
        _cid_registered.add(cid_name)
        print(f"Registered CID font: {cid_name}")
        return True
    except Exception as e:
        print(f"WARNING: Could not register CID font '{cid_name}': {e}")
        return False


def _register_ttf(font_name: str) -> bool:
    """Register a TTF font from FONTS_DIR with pdfmetrics once. Returns True on success."""
    if font_name in _ttf_registered:
        return True
    filename = TTF_FONT_MAP.get(font_name)
    if not filename:
        return False
    path = os.path.join(FONTS_DIR, filename)
    if not os.path.exists(path):
        print(f"WARNING: TTF font file not found: {path}  (run get_fonts.py to download it)")
        return False
    try:
        pdfmetrics.registerFont(TTFont(font_name, path))
        _ttf_registered.add(font_name)
        print(f"Registered TTF font: {font_name}")
        return True
    except Exception as e:
        print(f"WARNING: Could not register TTF font '{font_name}': {e}")
        return False


def _xml_escape(text: str) -> str:
    """Escape &, <, > so ReportLab's XML Paragraph parser doesn't crash."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _get_font_config(language: str) -> dict:
    """
    Return rendering configuration for the given language code.

    Returned dict keys:
        font_name  – name to pass to ParagraphStyle
        word_wrap  – 'CJK' for Japanese/Chinese/Korean, 'LTR' otherwise
        alignment  – 0 = left, 2 = right (Arabic/RTL)
        ok         – True if font registered successfully
    """
    lang = (language or 'en').lower().strip()

    # ── CJK: use built-in CID fonts (no file needed) ──────────────────────
    if lang in CID_FONT_MAP:
        cid_name = CID_FONT_MAP[lang]
        ok = _register_cid(cid_name)
        return {
            'font_name': cid_name if ok else 'Helvetica',
            'word_wrap': 'CJK',
            'alignment': 0,
            'ok':        ok,
        }

    # ── Arabic / Persian / Urdu: TTF + right-align ────────────────────────
    if lang in ('ar', 'fa', 'ur'):
        ok = _register_ttf('NotoSansArabic')
        return {
            'font_name': 'NotoSansArabic' if ok else 'Helvetica',
            'word_wrap': 'LTR',
            'alignment': 2,   # right-align
            'ok':        ok,
        }

    # ── Hindi / Devanagari: TTF ────────────────────────────────────────────
    if lang == 'hi':
        ok = _register_ttf('NotoSansDevanagari')
        return {
            'font_name': 'NotoSansDevanagari' if ok else 'Helvetica',
            'word_wrap': 'LTR',
            'alignment': 0,
            'ok':        ok,
        }

    # ── Latin / default ───────────────────────────────────────────────────
    ok = _register_ttf('NotoSans')
    return {
        'font_name': 'NotoSans' if ok else 'Helvetica',
        'word_wrap': 'LTR',
        'alignment': 0,
        'ok':        ok,
    }


def create_multilingual_pdf(output_path: str, title: str,
                             transcription: str, summary: str,
                             language: str = 'en') -> None:
    """
    Build a properly-rendered multilingual PDF.

    CJK languages (ja, zh, zh-cn, zh-tw, ko) use ReportLab's built-in
    CID fonts — no downloaded font files needed, guaranteed glyph rendering.

    Arabic / Hindi use Noto TTF files from ./fonts/.
    Latin falls back to NotoSans TTF or Helvetica.
    """
    fc        = _get_font_config(language)
    font_name = fc['font_name']
    word_wrap = fc['word_wrap']
    alignment = fc['alignment']

    if not fc['ok']:
        print(f"WARNING: Falling back to Helvetica for lang='{language}'. "
              "Non-Latin characters may render as boxes.")

    lang_label = language.upper() if language and language not in ('auto', '') else ''

    # ── Paragraph styles ──────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "TFTitle",
        fontName=font_name,
        fontSize=18,
        leading=26,
        alignment=1,           # always centre
        spaceAfter=8,
    )
    heading_style = ParagraphStyle(
        "TFHeading",
        fontName=font_name,
        fontSize=13,
        leading=18,
        alignment=alignment,
        spaceBefore=12,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "TFBody",
        fontName=font_name,
        fontSize=10,
        leading=15,
        alignment=alignment,
        wordWrap=word_wrap,
        spaceAfter=3,
    )

    # ── Flowables ─────────────────────────────────────────────────────────
    story = []
    story.append(Paragraph(_xml_escape(title), title_style))
    story.append(HRFlowable(width="100%", thickness=1, color="#00aacc"))
    story.append(Spacer(1, 5 * mm))

    summary_heading = f"Summary ({lang_label})" if lang_label else "Summary"
    story.append(Paragraph(_xml_escape(summary_heading), heading_style))

    if summary and summary.strip():
        for para in summary.split('\n'):
            if para.strip():
                story.append(Paragraph(_xml_escape(para), body_style))
    else:
        story.append(Paragraph("No summary available.", body_style))

    story.append(Spacer(1, 6 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color="#cccccc"))

    transcript_heading = f"Transcript ({lang_label})" if lang_label else "Transcript"
    story.append(Paragraph(_xml_escape(transcript_heading), heading_style))

    if transcription and transcription.strip():
        for line in transcription.split('\n'):
            if line.strip():
                story.append(Paragraph(_xml_escape(line), body_style))
    else:
        story.append(Paragraph("No transcription available.", body_style))

    # ── Build ─────────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )
    doc.build(story)
    print(f"PDF generated: {output_path}  (font={font_name}, lang={language})")


# ============================================================
# DOWNLOAD ROUTE
# ============================================================

@app.route('/download/<filename>')
def download_file(filename):
    file_type   = request.args.get('type', 'txt')
    target_lang = request.args.get('lang', 'en')

    transcript_path = os.path.join(app.config['UPLOAD_FOLDER'], filename + ".txt")
    summary_path    = os.path.join(app.config['UPLOAD_FOLDER'], filename + "_summary.txt")

    if not os.path.exists(transcript_path):
        return "File data not found. Please wait for processing to complete.", 404

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = f.read()

    # Translate if needed
    if target_lang != 'en':
        try:
            translator = GoogleTranslator(source='auto', target=target_lang)
            summary = translator.translate(summary) if summary else ""
            chunks = [transcript[i:i+4500] for i in range(0, len(transcript), 4500)]
            translated_chunks = []
            for c in chunks:
                if c.strip():
                    translated_chunks.append(translator.translate(c))
                else:
                    translated_chunks.append(c)
            transcript = "".join(translated_chunks)
        except Exception as e:
            print(f"Translation Error during download: {e}")

    if file_type == 'pdf':
        try:
            safe_base    = secure_filename(filename)
            pdf_filename = f"{safe_base}_{target_lang}.pdf"
            pdf_path     = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)

            create_multilingual_pdf(
                output_path=pdf_path,
                title="TranscribeFlow Report",
                transcription=transcript,
                summary=summary,
                language=target_lang,
            )

            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()

            try:
                os.remove(pdf_path)
            except Exception:
                pass

            return Response(
                pdf_bytes,
                mimetype="application/pdf",
                headers={"Content-Disposition": f"attachment;filename={pdf_filename}"}
            )

        except Exception as e:
            print(f"PDF generation error: {e}")
            return f"Error generating PDF: {str(e)}", 500

    # Plain-text download
    txt_content = f"SUMMARY:\n{summary}\n\nTRANSCRIPT:\n{transcript}"
    return Response(
        txt_content.encode('utf-8'),
        mimetype="text/plain",
        headers={"Content-Disposition": f"attachment;filename={filename}_{target_lang}.txt"}
    )


@app.route('/serve_audio/<filename>')
def serve_audio(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        for ext in [".txt", "_summary.txt"]:
            extra_file = os.path.join(app.config['UPLOAD_FOLDER'], filename + ext)
            if os.path.exists(extra_file):
                os.remove(extra_file)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/clear_all', methods=['POST'])
def clear_all():
    if os.path.exists(UPLOAD_FOLDER):
        for f in os.listdir(UPLOAD_FOLDER):
            try:
                os.remove(os.path.join(UPLOAD_FOLDER, f))
            except Exception:
                pass
    return index()


# ============================================================
# CHATBOT ROUTE (Powered by Groq)
# ============================================================

@app.route('/chat', methods=['POST'])
@token_required
def chat_with_bot(current_user):
    data = request.get_json()
    user_message       = data.get('message', '').strip()
    transcript_context = data.get('context', '')

    if not user_message:
        return jsonify({"reply": "Message cannot be empty"}), 400

    if not groq_client:
        return jsonify({"reply": "Groq API key is missing. The Chatbot is currently offline."}), 500

    website_knowledge = """
TranscribeFlow is an AI audio transcription web app.

Features & Workflows:
- Supported Formats: ONLY local MP3 and WAV files (Max 50MB).
- Upload Process: Users must drag & drop or click the "Select Audio" pod in the main dashboard, then click "Transcribe Now".
- Cloud Integrations: None. We do NOT support YouTube, Tubi, Spotify, Google Drive, or Dropbox. Local files only.
- Free users: trial access managed via IP.
- Generates timestamped transcription using Whisper.
- Generates AI summary using BART.
- Supports dynamic translation via GoogleTranslator.
- Export as TXT or Multilingual PDF (correct fonts for Japanese, Chinese, Korean, Arabic, Hindi).
- Secure JWT authentication & Argon2 hashing.
"""

    history = session.get("bot_history", [])
    history.append({"role": "user", "content": user_message})
    history = history[-5:]
    session["bot_history"] = history

    system_prompt = f"""You are FlowBot, the AI assistant for TranscribeFlow.

Rules:
- Be friendly, helpful, and conversational.
- Answer general questions naturally, but prioritize helping with TranscribeFlow.
- STRICT RULE: NEVER invent features, cloud integrations, or supported platforms not listed in the Website Information.
- If a user has upload trouble, tell them to ensure it is an MP3 or WAV file and to click the 'Select Audio' pod.
- Keep responses concise (under 6 lines if possible).
- Use bullet points if helpful.
- Always end your response with:

Best regards,
FlowBot
transcribeflow.app@gmail.com

Website Information (STRICT TRUTH):
{website_knowledge}"""

    if transcript_context:
        system_prompt += f"\n\nHere is the user's current audio transcript for context:\n{transcript_context[:3000]}"

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=250,
            top_p=0.9
        )
        assistant_reply = chat_completion.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": assistant_reply})
        session["bot_history"] = history[-5:]
        return jsonify({"reply": assistant_reply})

    except Exception as e:
        print(f"Groq Chatbot Generation Error: {e}")
        return jsonify({"reply": "I'm having trouble connecting to my neural network right now. Try again in a moment!"}), 500


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
