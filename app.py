from dotenv import load_dotenv
load_dotenv()
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import time
import threading
import io
import random
import datetime
import uuid
import jwt
import json
import secrets
import re
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, redirect, url_for, session
from werkzeug.utils import secure_filename
import whisper
from deep_translator import GoogleTranslator
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display
from passlib.context import CryptContext
from transformers import BartForConditionalGeneration, BartTokenizer


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

# Google OAuth Flow (disable HTTPS check for local dev)
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# ============================================================
# AUTH CONFIGURATION
# ============================================================
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
ALGORITHM = "HS256"
DB_FILE = "users.json"

# In-memory stores
reset_tokens = {}  # {token: {email, expires}}
otp_store = {}     # {email: {otp, expires}}

# ============================================================
# DATABASE HELPERS
# ============================================================
def load_users():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r') as f:
                return json.load(f)            # Read and convert JSON to Python dict
        except Exception as e:
            print(f"Error loading database: {e}")
            return {}
    return {}

def save_users():
    try:
        with open(DB_FILE, 'w') as f:
            json.dump(users_db, f, indent=4)       # Convert Python dict to JSON and save
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
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=6)
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

# --- NEW: Send OTP Email ---
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
# AUTH ROUTES (EXISTING + NEW)
# ============================================================

# --- STEP 1: REQUEST REGISTRATION OTP ---
@app.route('/auth/register/request_otp', methods=['POST'])
def register_request_otp():
    """Send OTP to email before allowing registration"""
    data = request.get_json()

    if not data or 'email' not in data:
        return jsonify({"detail": "Missing email address"}), 400

    email = data['email'].strip().lower()

    # Check if email already exists
    if email in users_db:
        return jsonify({"detail": "This email is already registered. Please login instead."}), 400

    # Generate 6-digit OTP
    otp = str(random.randint(100000, 999999))
    otp_store[email] = {
        "otp": otp,
        "expires": datetime.datetime.utcnow() + datetime.timedelta(minutes=10),
        "verified": False  # Track verification status
    }

    sent = send_otp_email(email, otp)

    if not sent:
        otp_store.pop(email, None)
        return jsonify({"detail": "Could not send OTP. Please check SMTP configuration."}), 500

    return jsonify({"message": "OTP sent to your email. Please verify to continue registration."})


# --- STEP 2: VERIFY EMAIL & COMPLETE REGISTRATION ---
@app.route('/auth/register', methods=['POST'])
def register():
    """Complete registration after OTP verification"""
    data = request.get_json()

    if not data or 'email' not in data or 'password' not in data or 'name' not in data or 'otp' not in data:
        return jsonify({"detail": "Missing required fields (name, email, password, otp)"}), 400

    name     = data['name'].strip()
    email    = data['email'].strip().lower()
    password = data['password']
    phone    = normalise_phone(data.get('phone', ''))
    otp      = data['otp'].strip()

    # Verify OTP first
    otp_data = otp_store.get(email)
    
    if not otp_data:
        return jsonify({"detail": "No OTP found. Please request a new one."}), 400

    if datetime.datetime.utcnow() > otp_data['expires']:
        otp_store.pop(email, None)
        return jsonify({"detail": "OTP has expired. Please request a new one."}), 400

    if otp_data['otp'] != otp:
        return jsonify({"detail": "Incorrect OTP. Please try again."}), 401

    # OTP verified - proceed with registration
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

    # Clear OTP after successful registration
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


# --- LOGIN (email/phone + password) ---
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

    # Check if user has email (important for phone-only users from old system)
    if not user_record.get('email'):
        return jsonify({
            "detail": "Your account needs to be updated with an email address. Please contact support."
        }), 403

    return jsonify({
        "access_token": create_token(email_key),
        "user_id":      user_record['user_id'],
        "name":         user_record['name']
    })


# --- NEW: REQUEST OTP (Send 6-digit OTP to email) ---
@app.route('/auth/request_otp', methods=['POST'])
def request_otp():
    data = request.get_json()

    if not data or 'email' not in data:
        return jsonify({"detail": "Missing email address"}), 400

    email = data['email'].strip().lower()

    if email not in users_db:
        # Prevent user enumeration
        return jsonify({"message": "If that email is registered, an OTP has been sent."})

    # Generate 6-digit OTP
    otp = str(random.randint(100000, 999999))
    otp_store[email] = {
        "otp": otp,
        "expires": datetime.datetime.utcnow() + datetime.timedelta(minutes=5)
    }

    sent = send_otp_email(email, otp)

    if not sent:
        otp_store.pop(email, None)
        return jsonify({"detail": "Could not send OTP. Please check SMTP configuration."}), 500

    return jsonify({"message": "If that email is registered, an OTP has been sent."})


# --- NEW: VERIFY OTP (Login with OTP) ---
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

    if datetime.datetime.utcnow() > otp_data['expires']:
        otp_store.pop(email, None)
        return jsonify({"detail": "OTP has expired. Please request a new one."}), 400

    if otp_data['otp'] != otp:
        return jsonify({"detail": "Incorrect OTP."}), 401

    # OTP verified - clear it and log user in
    otp_store.pop(email, None)

    if email not in users_db:
        return jsonify({"detail": "User not found."}), 404

    user_record = users_db[email]

    return jsonify({
        "access_token": create_token(email),
        "user_id":      user_record['user_id'],
        "name":         user_record['name']
    })


# --- NEW: GOOGLE OAUTH LOGIN (Step 1: Redirect to Google) ---
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


# --- NEW: GOOGLE OAUTH CALLBACK (Step 2: Handle Google response) ---
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

    # Fetch token
    flow.fetch_token(authorization_response=request.url)

    credentials = flow.credentials
    request_session = google_requests.Request()

    # Verify ID token
    id_info = id_token.verify_oauth2_token(
        credentials.id_token, request_session, GOOGLE_CLIENT_ID
    )

    email = id_info.get('email')
    name  = id_info.get('name', email.split('@')[0])

    if not email:
        return "Could not retrieve email from Google", 400

    email = email.lower()

    # Check if user exists, if not create account
    if email not in users_db:
        new_user_obj = User(name, email, phone="")
        user_details = new_user_obj.register()
        user_details['password_hash'] = ""  # No password for OAuth users
        users_db[email] = user_details
        save_users()

    user_record = users_db[email]

    # Create JWT token
    token = create_token(email)

    # Redirect to frontend with token in URL (frontend will store it)
    return redirect(f"/?google_login=success&token={token}&name={user_record['name']}&user_id={user_record['user_id']}")


# --- FORGOT PASSWORD ---
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
        "expires": datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
    }

    sent = send_reset_email(email, token)

    if not sent:
        reset_tokens.pop(token, None)
        return jsonify({"detail": "Could not send reset email. Please check SMTP configuration."}), 500

    return jsonify({"message": "If that email is registered, a reset link has been sent."})


# --- RESET PASSWORD PAGE ---
@app.route('/auth/reset_password_page', methods=['GET'])
def reset_password_page():
    token = request.args.get('token', '')
    return render_template('reset_password.html', token=token)


# --- RESET PASSWORD ---
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

    if datetime.datetime.utcnow() > token_data['expires']:
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
# TRANSCRIPTION & APP FUNCTIONALITY (unchanged)
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
                files.append({'name': f, 'time': time.ctime(os.path.getctime(path))})
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
    data       = request.get_json()
    transcript = data.get('transcript', '')
    summary    = data.get('summary', '')
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
            pdf      = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            font_map = {
                'hi': 'NotoSansDevanagari-Regular.ttf',
                'ja': 'NotoSansJP-Regular.ttf',
                'ar': 'NotoSansArabic-Regular.ttf',
                'en': 'NotoSans-Regular.ttf'
            }
            desired_font = font_map.get(target_lang, 'NotoSans-Regular.ttf')

            if os.path.exists(desired_font):
                pdf.add_font("CustomFont", style="", fname=desired_font)
                family = "CustomFont"
            elif os.path.exists('NotoSans-Regular.ttf'):
                pdf.add_font("CustomFont", style="", fname='NotoSans-Regular.ttf')
                family = "CustomFont"
            else:
                family = "Arial"

            pdf.add_page()

            def format_for_pdf(text, lang_code):
                if not text: return ""
                if lang_code == 'ar':
                    return get_display(arabic_reshaper.reshape(text))
                return text

            pdf.set_font(family, size=20)
            pdf.cell(0, 15, txt="TranscribeFlow Report", ln=True, align='C')
            pdf.ln(5)
            pdf.set_font(family, size=14)
            pdf.cell(0, 10, txt=f"Summary ({target_lang.upper()}):", ln=True)
            pdf.set_font(family, size=11)
            pdf.multi_cell(0, 7, txt=format_for_pdf(summary, target_lang))
            pdf.ln(10)
            pdf.set_font(family, size=14)
            pdf.cell(0, 10, txt=f"Transcript ({target_lang.upper()}):", ln=True)
            pdf.set_font(family, size=10)
            pdf.multi_cell(0, 6, txt=format_for_pdf(transcript, target_lang))

            pdf_output = pdf.output()
            return Response(
                bytes(pdf_output),
                mimetype="application/pdf",
                headers={"Content-Disposition": f"attachment;filename={filename}_{target_lang}.pdf"}
            )
        except Exception as e:
            return f"Error generating PDF: {str(e)}", 500

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

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
