import os
import time
import threading
import io
import random
import datetime
import uuid
import jwt
import json  # --- NEW IMPORT FOR DATABASE ---
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename
import whisper
from deep_translator import GoogleTranslator
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display
from passlib.context import CryptContext
from transformers import BartForConditionalGeneration, BartTokenizer

app = Flask(__name__)
app.secret_key = "transcribe_flow_secret_key"

# --- AUTH CONFIGURATION ---
# Using Argon2 to avoid the "password too long" error
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
ALGORITHM = "HS256"
DB_FILE = "users.json"  # File to store user data

# --- DATABASE PERSISTENCE HELPER FUNCTIONS ---
def load_users():
    """Loads users from the JSON file on startup."""
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading database: {e}")
            return {}
    return {}

def save_users():
    """Saves the current users_db to the JSON file."""
    try:
        with open(DB_FILE, 'w') as f:
            json.dump(users_db, f, indent=4)
    except Exception as e:
        print(f"Error saving database: {e}")

# Initialize Database from file
users_db = load_users()
print(f"Database loaded. {len(users_db)} users found.")

# Configuration ⚙️
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXTENSIONS = {'mp3', 'wav'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Limit upload size to 50MB
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

processing_jobs = {}

# --- MODELS LOADING ---
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

# --- USER CLASS ---
class User:
    def __init__(self, name, email):
        self.user_id = str(uuid.uuid4())
        self.name = name
        self.email = email
        self.created_at = datetime.datetime.now()

    def register(self):
        return {
            "user_id": self.user_id,
            "name": self.name,
            "email": self.email,
            "registered_on": self.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }

# --- AUTH HELPER FUNCTIONS ---
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

# --- AUTH ROUTES ---
@app.route('/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    
    # 1. Validation
    if not data or 'email' not in data or 'password' not in data or 'name' not in data:
        return jsonify({"detail": "Missing name, email, or password"}), 400
        
    name = data['name']
    email = data['email']
    password = data['password']

    if email in users_db:
        return jsonify({"detail": "User already exists"}), 400

    # 2. Create User Object
    new_user_obj = User(name, email)
    user_details = new_user_obj.register()
    
    # 3. Hash Password
    user_details['password_hash'] = hash_password(password)
    
    # 4. Save to Memory DB
    users_db[email] = user_details
    
    # 5. Save to File (Persistence)
    save_users()
    
    return jsonify({
        "message": "User registered successfully",
        "user_details": {
            "user_id": user_details['user_id'],
            "name": user_details['name'],
            "email": user_details['email'],
            "registered_on": user_details['registered_on']
        }
    })

@app.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    
    if not data or 'email' not in data or 'password' not in data:
        return jsonify({"detail": "Missing email or password"}), 400
        
    email = data['email']
    password = data['password']
    
    # Retrieve user record
    user_record = users_db.get(email)
    
    # Verify existence and password
    if not user_record or not verify_password(password, user_record['password_hash']):
        return jsonify({"detail": "Invalid credentials"}), 401
        
    return jsonify({
        "access_token": create_token(email),
        "user_id": user_record['user_id'],
        "name": user_record['name']
    })

# --- TRANSCRIPTION & APP FUNCTIONALITY ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_timestamp(seconds):
    td = time.gmtime(seconds)
    return time.strftime("%H:%M:%S", td)

def summarize_chunk(text_chunk):
    if not summ_model or not summ_tokenizer:
        return text_chunk
    try:
        input_text = text_chunk
        input_ids = summ_tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = summ_model.generate(
            input_ids, max_length=150, min_length=40, num_beams=4, length_penalty=2.0, early_stopping=True, no_repeat_ngram_size=3
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
        segments = result.get('segments', [])
        formatted_transcript = ""
        
        for s in segments:
            start = format_timestamp(s['start'])
            end = format_timestamp(s['end'])
            formatted_transcript += f"[{start} - {end}] {s['text'].strip()}\n"

        processing_jobs[filename]['progress'] = 75
        summary_text = ""
        
        if len(full_text) > 50:
            if full_text[:50] == full_text[50:100]: 
                full_text = full_text[:50]
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
        summary_path = os.path.join(app.config['UPLOAD_FOLDER'], filename + "_summary.txt")
        
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(formatted_transcript)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        processing_jobs[filename].update({
            'status': 'completed', 'progress': 100, 'transcript': formatted_transcript, 'summary': summary_text
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
            filename = secure_filename(file.filename)
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
    data = request.get_json()
    transcript = data.get('transcript', '')
    summary = data.get('summary', '')
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
    file_type = request.args.get('type', 'txt')
    target_lang = request.args.get('lang', 'en')
    
    transcript_path = os.path.join(app.config['UPLOAD_FOLDER'], filename + ".txt")
    summary_path = os.path.join(app.config['UPLOAD_FOLDER'], filename + "_summary.txt")
    
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
                if c.strip(): translated_chunks.append(translator.translate(c))
                else: translated_chunks.append(c)
            transcript = "".join(translated_chunks)
        except Exception as e:
            print(f"Translation Error during download: {e}")

    if file_type == 'pdf':
        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            font_map = {'hi': 'NotoSansDevanagari-Regular.ttf', 'ja': 'NotoSansJP-Regular.ttf', 'ar': 'NotoSansArabic-Regular.ttf', 'en': 'NotoSans-Regular.ttf'}
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
                if lang_code == 'ar': return get_display(arabic_reshaper.reshape(text))
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
            return Response(bytes(pdf_output), mimetype="application/pdf", headers={"Content-Disposition": f"attachment;filename={filename}_{target_lang}.pdf"})
        except Exception as e:
            return f"Error generating PDF: {str(e)}", 500

    txt_content = f"SUMMARY:\n{summary}\n\nTRANSCRIPT:\n{transcript}"
    return Response(txt_content.encode('utf-8'), mimetype="text/plain", headers={"Content-Disposition": f"attachment;filename={filename}_{target_lang}.txt"})

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
            if os.path.exists(extra_file): os.remove(extra_file)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_all', methods=['POST'])
def clear_all():
    if os.path.exists(UPLOAD_FOLDER):
        for f in os.listdir(UPLOAD_FOLDER):
            try:
                os.remove(os.path.join(UPLOAD_FOLDER, f))
            except Exception: pass
    return index()

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
