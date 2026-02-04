import os
import time
import threading
import io
import random
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename
import whisper
from deep_translator import GoogleTranslator
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)
app.secret_key = "transcribe_flow_secret_key"

# Configuration ⚙️
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXTENSIONS = {'mp3', 'wav'} 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global dictionary to track job status
processing_jobs = {}

# 1. Load Whisper Model 🎙️
print("Loading Whisper Model...")
try:
    model = whisper.load_model("base")
    print("Whisper Model Loaded Successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load Whisper model. Details: {e}")

# 2. Load Summarization Model (T5-Small) 📝
print("Loading T5 Summarization Model...")
summ_model = None
summ_tokenizer = None
try:
    summ_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    summ_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    print("T5 Model Loaded Successfully.")
except Exception as e:
    print(f"WARNING: Could not load T5 model. Details: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_timestamp(seconds):
    td = time.gmtime(seconds)
    return time.strftime("%H:%M:%S", td)

def summarize_chunk(text_chunk):
    """Summarizes using Sampling to force rephrasing (Abstractive Summarization)."""
    if not summ_model or not summ_tokenizer:
        return text_chunk
    
    try:
        # T5 specific prefix
        input_text = "summarize: " + text_chunk
        
        input_ids = summ_tokenizer.encode(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        # --- NEW "CREATIVE" SETTINGS TO FIX REPETITION ---
        summary_ids = summ_model.generate(
            input_ids,
            max_length=100,
            min_length=20,
            
            # "Sample" mode forces rephrasing
            do_sample=True,      
            top_k=50,
            top_p=0.95,
            temperature=0.6,     # Higher = more rephrasing, Lower = more copying
            
            # Anti-repetition parameters
            repetition_penalty=2.5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
        return summ_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Chunk summarization error: {e}")
        return text_chunk

def run_transcription(file_path, filename):
    print(f"Starting transcription for: {filename}")
    try:
        # Step 1: Initialize
        processing_jobs[filename] = {'status': 'processing', 'progress': 10}
        
        # Step 2: Transcribe
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

        # Step 3: Summarize (T5)
        processing_jobs[filename]['progress'] = 75
        summary_text = ""
        
        if len(full_text) > 50:
            # Clean possible loops from source
            if full_text[:50] == full_text[50:100]: 
                full_text = full_text[:50]

            chunk_size = 2000 
            chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
            summarized_chunks = []
            
            for chunk in chunks:
                # Even short chunks get summarized to force rephrasing
                if len(chunk.strip()) > 30: 
                    clean_summary = summarize_chunk(chunk)
                    # Deduplication check
                    if clean_summary and clean_summary.lower() not in [s.lower() for s in summarized_chunks]:
                        summarized_chunks.append(clean_summary)
            
            summary_text = " ".join(summarized_chunks)
        else:
            # If the WHOLE file is tiny, just run it through the summarizer once
            summary_text = summarize_chunk(full_text)

        processing_jobs[filename]['progress'] = 85
        
        # Save files
        transcript_path = os.path.join(app.config['UPLOAD_FOLDER'], filename + ".txt")
        summary_path = os.path.join(app.config['UPLOAD_FOLDER'], filename + "_summary.txt")
        
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(formatted_transcript)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        # Step 4: Finalize
        processing_jobs[filename].update({
            'status': 'completed',
            'progress': 100,
            'transcript': formatted_transcript,
            'summary': summary_text
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
            
            # Start transcription thread
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
        # Handle large text by chunking
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

    # Translation for download
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
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # FONT CONFIGURATION
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
            return Response(bytes(pdf_output), mimetype="application/pdf",
                            headers={"Content-Disposition": f"attachment;filename={filename}_{target_lang}.pdf"})
        except Exception as e:
            print(f"PDF GENERATION ERROR: {e}")
            return f"Error generating PDF: {str(e)}. Check server logs for font details.", 500

    # TXT Download
    txt_content = f"SUMMARY:\n{summary}\n\nTRANSCRIPT:\n{transcript}"
    return Response(txt_content.encode('utf-8'), mimetype="text/plain",
                    headers={"Content-Disposition": f"attachment;filename={filename}_{target_lang}.txt"})

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
            except Exception as e:
                print(f"Error deleting {f}: {e}")
    return index()

if __name__ == '__main__':
    app.run(debug=True, threaded=True)