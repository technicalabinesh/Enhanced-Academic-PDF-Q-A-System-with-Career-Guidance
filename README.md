# 📚 Enhanced Academic PDF Q&A System with Career Guidance

A comprehensive **Streamlit application** that combines **document analysis, AI-powered question answering, and career guidance** with roadmap-style learning paths.

---

## 🌟 Features

### 📤 Document Processing
- **PDF Text Extraction** – Extract and process text from PDF documents  
- **Image OCR** – Extract text from images using **Tesseract OCR**  
- **Multi-language Support** – Process documents in **12+ languages**  
- **Chunking & Indexing** – Intelligent text chunking with overlap for better context  

### ❓ AI-Powered Q&A
- **Document-based Answers** – Get answers from your uploaded documents  
- **General Knowledge** – AI answers using **IBM Watsonx** when no documents are available  
- **Multi-language Support** – Answers in **12+ languages**  
- **Voice Questions** – Ask questions using your microphone  
- **Audio Transcription** – Upload audio files for transcription  

### 🧭 Career Guidance
- **25+ Career Paths** – Comprehensive roadmaps for tech careers  
- **YouTube Integration** – Embedded curated videos for each career  
- **Personalized Plans** – Generate customized learning plans  
- **Skill Level Tracking** – Progress from Beginner ➝ Advanced  

### 📅 Study Tools
- **Smart Study Planner** – Personalized study schedules  
- **Quiz Generator** – Create MCQs from documents  
- **Question Generator** – Generate practice questions  
- **Progress Tracking** – Track your learning journey  

---

## 🚀 Installation

### ✅ Prerequisites
- Python **3.8+**  
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (for image text extraction)  
- IBM **Watsonx** credentials (optional but recommended for AI features)  

### ⚙️ Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd academic-qa-system

   Install dependencies

pip install -r requirements.txt


Install Tesseract OCR

Windows: Download here

macOS:

brew install tesseract


Linux (Ubuntu/Debian):

sudo apt-get install tesseract-ocr


Configure IBM Watsonx (Optional)

Create an IBM Cloud account

Enable Watsonx service

Get API Key, Project ID, and URL

Enter them in the app sidebar

Run the application

streamlit run app.py

🛠️ Configuration

Edit Config class in app.py if needed:

class Config:
    CHUNK_SIZE = 600  # Text chunk size
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Sentence embedding model
    TESSERACT_CMD = None  # Set path if Tesseract is not in PATH
    # Example: r'C:\Program Files\Tesseract-OCR\tesseract.exe'

📊 Supported Career Paths

Includes 25+ roadmaps:

🖥️ Development: Frontend, Backend, Full Stack, Android, iOS, Blockchain

📊 Data & AI: Data Analyst, Data Scientist, Data Engineer, ML Engineer, AI Engineer, MLOps Engineer

☁️ DevOps & Infra: DevOps Engineer, Software Architect

🔒 Other Roles: QA Engineer, Cyber Security, UX Designer, Technical Writer, Game Developer, BI Analyst

👨‍💼 Management: Product Manager, Engineering Manager, Developer Relations

🌐 Supported Languages

Supports 12 languages for answers:

English, Tamil, Hindi, Telugu, Malayalam, Kannada, Spanish, French, German, Chinese, Japanese, Arabic

🎥 YouTube Integration

Each career roadmap has curated videos. Example:

Frontend: Video 1
, Video 2

Backend: Video

Data Analysis: Video

AI & ML: Video
, Video

📁 Project Structure
academic-qa-system/
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── README.md              # Documentation
├── cache/                 # Index cache
│   ├── index.faiss
│   └── metadata.pkl
└── assets/                # Optional assets

💡 Usage Guide

Document Upload – Upload PDFs/images, click Process documents

Ask Questions – Type your question, choose doc-based or AI mode

Career Exploration – Browse roadmaps, watch videos, get learning plans

Study Planning – Generate personalized schedules

Assessment Tools – Create quizzes and practice questions

🔧 Technical Architecture

PDF Processor – Extracts and cleans text

Vector Store – Embeddings + similarity search using FAISS

AI Integration – IBM Watsonx for LLM capabilities

Translation Service – Multi-language support (Google Translate)

Career Guidance – Roadmaps + YouTube integration

Study Tools – Quiz + Planner

🚦 Performance Tips

Process documents in batches

Adjust chunk size & overlap based on content

Clear cache periodically

For large PDFs, increase overlap for better answers

🤝 Contributing

We welcome contributions!

Fork the repository

Create a feature branch

Make your changes

Submit a pull request

📝 License

This project is licensed under the MIT License – see the LICENSE file.

🆘 Support

If you face issues:

Check troubleshooting section

Ensure all dependencies installed

Verify IBM Watsonx credentials

Confirm Tesseract OCR is configured

🔮 Future Enhancements

📱 Mobile App version

👥 Collaborative study groups

📊 Advanced analytics

🔗 Integration with LMS platforms

🎬 Better curated video content

🙏 Acknowledgments

IBM Watsonx for AI

Streamlit for web framework

Hugging Face for embeddings

Google Translate for multi-language support

Tesseract OCR for text extraction

All YouTube creators for learning content

Dependencies for your Academic PDF Q&A System with Career Guidance:


# Core framework
pip install streamlit

# PDF processing
pip install PyMuPDF   # fitz
pip install pdfplumber

# OCR (requires Tesseract installed separately)
pip install pytesseract
pip install Pillow

# AI/Embeddings & Vector search
pip install sentence-transformers
pip install faiss-cpu

# Audio handling (speech-to-text & MP3/WAV support)
pip install SpeechRecognition
pip install pydub

# Translation & multilingual support
pip install deep-translator
pip install googletrans==4.0.0rc1

# IBM Watsonx integration
pip install ibm-watsonx-ai

# Data handling
pip install numpy
pip install pandas

# File caching & persistence
pip install pickle-mixin

# Optional (if using YouTube integration, web requests, etc.)
pip install requests
pip install youtube-search-python
