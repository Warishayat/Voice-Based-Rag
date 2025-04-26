import gradio as gr
import os
import tempfile
import time
import sys
from pathlib import Path
import shutil
import pygame
from gtts import gTTS
import logging
import threading

# Initialize pygame mixer
pygame.mixer.init()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your existing modules
from Pdf_Vectors import Pdf_parsing, textSplitter, qa_chain
from Speech_Query import Speech_Recognizer, transcribe_text

# Global state
current_qa_chain = None
processing_status = ""
temp_files = []

def TextToSpeech(respond_text):
    """Convert text to speech with proper file handling"""
    try:
        timestamp = int(time.time())
        audio_path = f"response_{timestamp}.mp3"
        temp_files.append(audio_path)
        
        tts = gTTS(text=respond_text, lang="en")
        tts.save(audio_path)
        
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        
    except Exception as e:
        logging.error(f"TextToSpeech error: {e}")
        raise

def process_pdf(pdf_file):
    """Handle PDF upload and processing"""
    global current_qa_chain, processing_status, temp_files
    
    processing_status = "Processing PDF. Please wait..."
    yield processing_status
    
    try:
        temp_dir = os.path.join(tempfile.gettempdir(), "pdf_qa")
        os.makedirs(temp_dir, exist_ok=True)
        
        timestamp = int(time.time())
        pdf_path = os.path.join(temp_dir, f"upload_{timestamp}.pdf")
        temp_files.append(pdf_path)
        
        if isinstance(pdf_file, bytes):
            with open(pdf_path, "wb") as f:
                f.write(pdf_file)
        elif hasattr(pdf_file, "read"):
            with open(pdf_path, "wb") as f:
                f.write(pdf_file.read())
        else:
            raise ValueError("Unsupported file input type")
        
        docs = Pdf_parsing(pdf_path)
        retriever = textSplitter(docs)
        current_qa_chain = qa_chain(retriever)
        
        processing_status = "PDF processed successfully. You can now ask your questions."
        yield processing_status
        
    except Exception as e:
        processing_status = f"PDF Processing Error: {str(e)}"
        yield processing_status
        raise gr.Error(f"Failed to process PDF: {e}")

def handle_voice_query():
    """Handle voice recording and question answering"""
    global current_qa_chain, processing_status, temp_files
    
    if not current_qa_chain:
        yield "Please upload and process a PDF first.", None
        return
    
    try:
        processing_status = "Listening... Speak your question."
        yield processing_status, None
        
        temp_dir = os.path.join(tempfile.gettempdir(), "pdf_qa")
        os.makedirs(temp_dir, exist_ok=True)
        
        timestamp = int(time.time())
        audio_path = os.path.join(temp_dir, f"query_{timestamp}.mp3")
        temp_files.append(audio_path)
        
        # No time limit on recording
        Speech_Recognizer(audio_path, phrase_time_limit=None)
        
        processing_status = "Transcribing your question..."
        yield processing_status, None
        
        question = transcribe_text(audio_path)
        if not question or len(question.strip()) < 3:
            raise ValueError("No valid speech detected.")
        
        processing_status = "Generating answer..."
        yield processing_status, None
        
        response = current_qa_chain.invoke(question)
        answer = response.get("result", "Sorry, no answer could be generated.")
        
        # Clean and concise output
        clean_output = f"Question: {question.strip()}\n\nAnswer: {answer.strip()}"
        
        voice_thread = threading.Thread(target=TextToSpeech, args=(answer,))
        voice_thread.start()
        
        yield clean_output, answer
        
    except Exception as e:
        processing_status = f"Voice Processing Error: {str(e)}"
        yield processing_status, None
        raise gr.Error(f"Voice query failed: {e}")

def cleanup_temp_files():
    """Clean up temporary files"""
    global temp_files
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass
    temp_files = []

#Gradio
with gr.Blocks(
    title="Voice-Powered PDF Assistant",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 100% !important;
        padding: 20px;
        height: 100vh;
        background: linear-gradient(135deg, #f9f9f9 0%, #e3e3e3 100%);
        font-family: 'Poppins', sans-serif;
    }
    .container {
        max-width: 1100px;
        margin: 0 auto;
        height: 100%;
    }
    h1, p {
        color: #333;
    }
    .gr-button-primary {
        background-color: #4CAF50;
        border: none;
    }
    .gr-button-secondary {
        background-color: #2196F3;
        border: none;
    }
    """
) as app:
    
    with gr.Column(elem_classes=["container"]):
        gr.Markdown("""
        <h1 style="text-align: center;">Voice-Powered PDF Assistant</h1>
        <p style="text-align: center;">Upload a PDF and interact using your voice</p>
        """)
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Upload PDF")
                pdf_upload = gr.File(
                    label="Upload PDF Document",
                    type="binary",
                    file_types=[".pdf"],
                    height=100
                )
                process_btn = gr.Button(
                    "Process Document", 
                    variant="primary",
                    size="lg"
                )
                status_display = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3
                )
                
            with gr.Column(scale=2):
                gr.Markdown("### Ask Questions by Voice")
                voice_btn = gr.Button(
                    "Press to Speak", 
                    variant="secondary",
                    size="lg"
                )
                question_answer = gr.Textbox(
                    label="Interaction",
                    interactive=False,
                    lines=10,
                    autoscroll=True
                )
                gr.Markdown("""
                **Steps:**
                - Upload your PDF
                - Click "Process Document"
                - Press the button and ask your question
                """)
    
    # Event handlers
    process_btn.click(
        process_pdf,
        inputs=pdf_upload,
        outputs=status_display
    )
    
    voice_btn.click(
        handle_voice_query,
        outputs=[status_display, question_answer]
    )
    
    # Cleanup on close
    app.unload(cleanup_temp_files)

# ======================
# APPLICATION LAUNCH
# ======================
if __name__ == "__main__":
    temp_dir = os.path.join(tempfile.gettempdir(), "pdf_qa")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    try:
        app.launch(
            server_port=7860,
            show_error=True,
            share=False,
            debug=True,
            inbrowser=True
        )
    finally:
        cleanup_temp_files()
