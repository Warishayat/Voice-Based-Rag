import gradio as gr
import os
import pygame
from gtts import gTTS
from Pdf_Vectors import textSplitter, qa_chain, Pdf_parsing
from speech_query import Speech_Recognizer, transcribe_text  # Assuming these are in 'speech_query.py'
import warnings

warnings.filterwarnings("ignore")

def TextToSpeech(output_path, respond_text):
    """
    Convert the text into an MP3 format and play that file automatically.

    Args:
    - respond_text (str): The text that will be converted to speech.
    - output_path (str): The path where the MP3 file will be saved.
    """
    try:
        print("Now text-to-speech is working...")

        # Remove the file if it exists to avoid permission issues
        if os.path.exists(output_path):
            os.remove(output_path)

        # Convert text to speech and save as MP3
        tts = gTTS(text=respond_text, lang="en")
        tts.save(output_path)
        print(f"MP3 file is saved at: {output_path}")

        # Check if the file was saved correctly
        if not os.path.exists(output_path):
            print("Error: MP3 file not saved properly.")
            return

        # Initialize pygame mixer for audio playback
        pygame.mixer.init()

        # Load and play the MP3 file
        pygame.mixer.music.load(output_path)
        pygame.mixer.music.play()

        # Block until the music finishes playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    except Exception as e:
        print(f"Error in text-to-speech: {e}")

# Function to process the PDF, generate response, and convert to speech
def process_pdf_and_query(pdf_file, query_text):
    """
    Process the uploaded PDF, generate response for the query, and convert to speech.

    Args:
    - pdf_file (UploadedFile): The uploaded PDF file.
    - query_text (str): The text query to ask after PDF processing.
    """
    try:
        # Save the PDF file
        pdf_path = "temp_uploaded_pdf.pdf"
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.read())

        # Parse the PDF and create retriever
        docs = Pdf_parsing(pdf_path=pdf_path)
        retriever = textSplitter(documents=docs)
        chain = qa_chain(retriever=retriever)

        # Get the response
        response = chain.invoke(query_text)
        response_text = response["result"]

        # Create output path for MP3
        output_path = "response.mp3"

        # Convert the response to speech and play it
        TextToSpeech(output_path, respond_text=response_text)

        return response_text, output_path

    except Exception as e:
        return str(e), ""


# Define the Gradio interface
def query_via_voice(pdf_file):
    # This function allows the user to ask the question through voice and process the PDF
    file_path = "user_query.mp3"
    file_path_ex = Speech_Recognizer(file_path=file_path)  # Record voice from microphone

    if file_path_ex:
        query_text = transcribe_text(file_path=file_path_ex)  # Transcribe the recorded voice to text
        print(f"User Query: {query_text}")
        return process_pdf_and_query(pdf_file, query_text)
    else:
        return "Error in recording or transcribing query.", ""

iface = gr.Interface(
    fn=query_via_voice,  # This function allows the user to pass a query via voice
    inputs=[gr.File(label="Upload your PDF file")],  # PDF file input
    outputs=[
        gr.Textbox(label="Generated Answer", placeholder="The answer will appear here"),  # Text output
        gr.Audio(label="Response Audio", type="filepath")  # Audio output
    ],
    live=True,
    allow_flagging="never",
    title="Voice-Based PDF Query System",
    description="Upload a PDF, ask a query through your voice, and the system will respond with an audio answer."
)

# Launch the Gradio interface
if __name__ == "__main__":
    iface.launch()
