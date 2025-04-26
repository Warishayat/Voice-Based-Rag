# 🎤 Voice-Powered PDF Assistant

A Gradio web app that lets you **upload a PDF** and then **ask questions using your voice**.  
The app will **instantly generate answers** and **speak them out loud** — no delay!

---

## ✨ Features

- 📄 Upload any PDF document
- 🎤 Ask questions via your **microphone**
- 🤖 Powered by **local PDF retrieval** and **RAG (Retrieval-Augmented Generation)**
- 🗣️ **Instant text-to-speech** answers
- ⚡ Built using **Gradio**, **Threading**, and **local LLM chains**
- 🧹 Automatic **cleanup of temp files**

---

## 🛠 How It Works

1. Upload a PDF.
2. Click **Process Document**.
3. Click **🎤 Speak Question**, ask your question using voice.
4. View the **instant text** response **and** listen to the **spoken** answer immediately!

---

## 🖥️ Tech Stack

- **Python 3.10+**
- **Gradio** (for the web interface)
- **Threading** (for instant voice + text)
- **Tempfile / Shutil** (for handling file uploads safely)
- **Speech Recognition** (custom Speech_Recognizer)
- **PDF Parsing & Text Splitting** (custom Pdf_Vectors)
- **Text-to-Speech** (custom Rag_voice)

---

## 📂 Project Structure

```
├── App.py               # Gradio app (this file)
├── Speech_Query.py        # Speech recording + transcription functions
├── Pdf_Vectors.py         # PDF parsing + retriever + QA chain setup
├── Rag_voice.py           # Text-to-Speech functionality
```


---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone (https://github.com/Warishayat/Voice-Based-Rag)
cd Voice-Based-Rag
```

2. Install dependencies:

```bash
pip install gradio
```
(Also install any other dependencies used inside `Speech_Query.py`, `Pdf_Vectors.py`, and `Rag_voice.py`)

3. Run the app:

```bash
python app.py
```

The app will be available at:  
**http://localhost:7860**

---

## 💬 Future Improvements (optional)

- Add support for multiple PDFs
- Add loading animations ("Thinking...")
- Auto-stop previous audio when new response comes

---

## 📜 License

This project is licensed under the [Apache License](LICENSE).

---

## 🙌 Acknowledgements

Thanks to Gradio, Python community, and local LLM open-source projects for making this possible!

---
