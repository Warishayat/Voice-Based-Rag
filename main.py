import gradio as gr
import os
import pygame
from gtts import gTTS
from Pdf_Vectors import qa_chain, Pdf_parsing, textSplitter
from speech_recognition import Recognizer, Microphone
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# State Management for PDF and responses
def process_pdf(uploaded_file):
    """Handle the PDF upload and processing"""
    if uploaded_file is not None:
        # Save the file locally
        pdf_path = "temp_uploaded_pdf.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Parse the PDF and create retriever
        docs = Pdf_parsing(pdf_path=pdf_path)
        retriever = textSplitter(documents=docs)
        return retriever
    return None

# Function to record user query
def record_query():
    """Record the user's query using the microphone"""
    recognizer = Recognizer()
    mic = Microphone()

    with mic as source:
        print("Please speak now...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        query = recognizer.recognize_google(audio)
        print(f"You said: {query}")
        return query
    except Exception as e:
        print(f"Error: {e}")
        return ""

# Function to convert text to speech and play it
def TextToSpeech(output_path, respond_text):
    """Convert text to speech and play it automatically"""
    try:
        tts = gTTS(text=respond_text, lang="en")
        tts.save(output_path)

        # Play the audio file using pygame
        pygame.mixer.init()
        pygame.mixer.music.load(output_path)
        pygame.mixer.music.play()

        # Wait until audio finishes
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

# Function to handle Gradio interface inputs and outputs
def gradio_interface(uploaded_file, user_query):
    retriever = None

    # Process PDF if uploaded
    if uploaded_file is not None:
        retriever = process_pdf(uploaded_file)
        if retriever:
            print("PDF is ready for queries!")
    
    # Process the user's query and generate response
    if user_query:
        with gr.progress_bar():
            # Use the retriever to generate a response
            chain = qa_chain(retriever=retriever)
            response = chain.invoke(user_query)
            output = response["result"]
            print(f"Generated Answer: {output}")
        
            # Convert the response to speech and play it
            output_path = "response.mp3"
            TextToSpeech(output_path=output_path, respond_text=output)

            # Return the text response and the path to the audio file
            return output, output_path  # Return the file path for download

# Gradio interface function
def main():
    interface = gr.Interface(
        fn=gradio_interface,
        inputs=[gr.File(label="Upload PDF"), gr.Textbox(label="Ask a Question")],
        outputs=[gr.Textbox(label="Generated Answer"), gr.Audio(label="Download Response Audio", type="filepath")]
    )

    interface.launch()

if __name__ == "__main__":
    main()
