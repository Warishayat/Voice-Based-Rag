import os
import pygame
from gtts import gTTS
import logging
import time

# Set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def TextToSpeech(respond_text, output_path="studentvoice.mp3"):
    """
    Convert response text to speech, save as mp3, play it, and delete after playing.
    """
    try:
        # Delete old file if it exists
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                logging.info(f"Old file {output_path} deleted successfully before saving new one.")
            except Exception as e:
                logging.warning(f"Could not delete old file {output_path}: {e}")
                # Try renaming if delete failed
                output_path = f"studentvoice_{int(time.time())}.mp3"
                logging.info(f"Using new output path: {output_path}")

        # Convert the response text to speech and save
        tts = gTTS(text=respond_text, lang="en")
        tts.save(output_path)
        logging.info(f"Saved new response at {output_path}")

        # Initialize pygame mixer
        pygame.mixer.init()
        pygame.mixer.music.load(output_path)
        pygame.mixer.music.play()

        # Wait until the audio finishes playing
        clock = pygame.time.Clock()
        while pygame.mixer.music.get_busy():
            clock.tick(10)

        # Properly stop and unload the music
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        pygame.mixer.quit()

        # After playing, delete the mp3 file
        if os.path.exists(output_path):
            os.remove(output_path)
            logging.info(f"File {output_path} deleted after playing.")

    except Exception as e:
        logging.error(f"Error in TextToSpeech: {e}")

if __name__ == "__main__":
    # Example usage
    TextToSpeech("Hello! This is a test message. i am fine what about you")
