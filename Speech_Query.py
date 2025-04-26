import speech_recognition as sr
import logging                   #for logs 
from pydub import AudioSegment   #to convert the audio to mp3
from pydub.utils import which
import os
from groq import Groq
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import warnings
from io import BytesIO
warnings.filterwarnings("ignore")


#set tne env
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
FFMPEG_PATH = os.environ.get("FFMPEG_PATH")

if FFMPEG_PATH:
    AudioSegment.converter = FFMPEG_PATH
else:
    print("There is some Issue witht the FFMPEG-PATH")

#Set the logs
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

def Speech_Recognizer(file_path,timeout=20,phrase_time_limit=None):
    """
    Record audio from the microphone and save it as an MP3 file.

    Args:
        file_path (str): Path to save the recorded audio file.
        timeout (int): Max time to wait for speech to start (in seconds).
        phrase_time_limit (int or None): Max length of the speech (in seconds).
    """
    #intialize the recognizer
    speech_recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            logging.info("Adjusting the ambeint noise..........")
            speech_recognizer.adjust_for_ambient_noise(source=source,duration=1)
            logging.info("Start Speaking Now.........")

            #recognize speech
            audio_data = speech_recognizer.listen(source,phrase_time_limit=phrase_time_limit,timeout=timeout)
            logging.info("Recording is complete................")

            #now convert this wav speech into mp3 right format
            wav_data = audio_data.get_wav_data()
            Audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            Audio_segment.export(file_path,format="mp3",bitrate="128k")

            logging.info(f"Audio is saved at: {file_path}")
            
            return file_path

    except Exception as e: 
        print(f"An error has occured {e}")

#setup the ehisper for transcribe the text
client = Groq(api_key=GROQ_API_KEY)
def transcribe_text(file_path):
    with open(file_path,"rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3",
            language="en"
        )
        return transcription.text

if __name__ == "__main__":
    file_path = "SpeechQuery.mp3"
    file_path_ex=Speech_Recognizer(file_path=file_path)
    if file_path_ex:
        text=transcribe_text(file_path=file_path_ex)
        print(text)
    else:
        print("You may got some error")