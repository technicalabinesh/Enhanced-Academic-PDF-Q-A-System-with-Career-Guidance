import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
import hashlib
from typing import List, Dict, Tuple, Optional
import re
from datetime import datetime, timedelta
import tempfile
import json
from PIL import Image
import pytesseract
import speech_recognition as sr
from pydub import AudioSegment
from deep_translator import GoogleTranslator as DeepGoogleTranslator
import io
import calendar
from datetime import datetime
import traceback
import requests
import time

# IBM Watsonx imports
try:
    from ibm_watsonx_ai.foundation_models import Model
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    from ibm_watsonx_ai import Credentials
    IBM_WATSON_AVAILABLE = True
except ImportError:
    IBM_WATSON_AVAILABLE = False
    st.warning("IBM Watsonx libraries not available. Using mock implementation.")

# ------------------------
# Configuration
# ------------------------
class Config:
    CHUNK_SIZE = 600
    CHUNK_OVERLAP_WORDS = 50
    MAX_CHUNKS_TO_RETRIEVE = 6
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    WATSONX_MODEL = 'mistralai/mistral-small-3-1-24b-instruct-2503'
    CACHE_DIR = 'cache'
    TESSERACT_CMD = None  # set if tesseract binary is non-standard e.g. r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'ta': 'Tamil',
        'hi': 'Hindi',
        'te': 'Telugu',
        'ml': 'Malayalam',
        'kn': 'Kannada',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'zh-cn': 'Chinese (Simplified)',
        'ja': 'Japanese',
        'ar': 'Arabic'
    }
    # Career roadmaps data with YouTube links
    CAREER_ROADMAPS = {
        "Frontend Developer": {
            "description": "Frontend developers build the visual, interactive parts of websites and web applications that users see and interact with directly in their web browsers.",
            "levels": {
                "Beginner": ["HTML", "CSS", "JavaScript Basics", "Responsive Design"],
                "Intermediate": ["React", "Vue.js", "State Management", "API Integration"],
                "Advanced": ["TypeScript", "Performance Optimization", "Testing", "Web Security"]
            },
            "timeline": "8-14 months",
            "resources": [
                "MDN Web Docs",
                "FreeCodeCamp Frontend Curriculum",
                "Frontend Developer Roadmap (roadmap.sh)"
            ],
            "youtube_links": [
                "https://youtu.be/SSKVgrwhzus?si=YdynWfXjt_lXFsAT",
                "https://youtu.be/nu_pCVPKzTk?si=PRFi2PHR0TteCMBe"
            ]
        },
        "Backend Developer": {
            "description": "Backend developers work on server-side logic, databases, and application integration to power the functionality behind websites and applications.",
            "levels": {
                "Beginner": ["Programming Fundamentals", "Basic Algorithms", "HTTP Basics", "Database Fundamentals"],
                "Intermediate": ["Node.js/Python/Java", "REST APIs", "SQL/NoSQL Databases", "Authentication"],
                "Advanced": ["Microservices", "Caching Strategies", "Message Queues", "System Design"]
            },
            "timeline": "10-16 months",
            "resources": [
                "Backend Developer Roadmap (roadmap.sh)",
                "Designing Data-Intensive Applications (Book)",
                "System Design Primer (GitHub)"
            ],
            "youtube_links": [
                "https://youtu.be/9BD9eK9VqXA?si=gF9lUNkldCDv9h10"
            ]
        },
        "Full Stack Developer": {
            "description": "Full stack developers work on both frontend and backend parts of applications, handling everything from user interfaces to server infrastructure.",
            "levels": {
                "Beginner": ["HTML/CSS", "JavaScript", "Programming Basics", "Git"],
                "Intermediate": ["Frontend Framework", "Backend Language", "Database Management", "API Development"],
                "Advanced": ["DevOps Basics", "Testing Strategies", "Performance Optimization", "Architecture Patterns"]
            },
            "timeline": "12-18 months",
            "resources": [
                "Full Stack Developer Roadmap (roadmap.sh)",
                "The Odin Project",
                "Full Stack Open"
            ],
            "youtube_links": [
                "https://youtu.be/nu_pCVPKzTk?si=PRFi2PHR0TteCMBe"
            ]
        },
        "DevOps Engineer": {
            "description": "DevOps engineers bridge development and operations, focusing on automation, infrastructure, and continuous delivery pipelines.",
            "levels": {
                "Beginner": ["Linux Fundamentals", "Scripting (Bash/Python)", "Git", "Networking Basics"],
                "Intermediate": ["Docker", "CI/CD Pipelines", "Cloud Basics (AWS/Azure/GCP)", "Infrastructure as Code"],
                "Advanced": ["Kubernetes", "Monitoring & Logging", "Security Best Practices", "System Architecture"]
            },
            "timeline": "10-16 months",
            "resources": [
                "DevOps Roadmap (roadmap.sh)",
                "Kubernetes Documentation",
                "DevOps Handbook"
            ],
            "youtube_links": [
                "https://youtu.be/hQcFE0RD0cQ?si=UJUFksKTQu9veSDW"
            ]
        },
        "Data Analyst": {
            "description": "Data analysts collect, process, and perform statistical analyses on data to help organizations make data-driven decisions.",
            "levels": {
                "Beginner": ["Excel", "SQL Basics", "Data Visualization Principles", "Statistics Fundamentals"],
                "Intermediate": ["Python/R for Analysis", "Advanced SQL", "Power BI/Tableau", "Data Cleaning"],
                "Advanced": ["Machine Learning Basics", "Big Data Tools", "Advanced Visualization", "Storytelling with Data"]
            },
            "timeline": "6-12 months",
            "resources": [
                "Data Analysis with Python (Coursera)",
                "SQL for Data Analysis (Udacity)",
                "Storytelling with Data (Book)"
            ],
            "youtube_links": [
                "https://youtu.be/PSNXoAs2FtQ?si=h3qqZ_xQ_rZ9f5Xd"
            ]
        },
        "AI Engineer": {
            "description": "AI engineers develop and implement artificial intelligence models and systems to solve complex problems.",
            "levels": {
                "Beginner": ["Python Programming", "Linear Algebra", "Calculus", "Statistics"],
                "Intermediate": ["Machine Learning Algorithms", "Deep Learning Fundamentals", "Data Preprocessing", "Model Evaluation"],
                "Advanced": ["Natural Language Processing", "Computer Vision", "Reinforcement Learning", "MLOps"]
            },
            "timeline": "12-18 months",
            "resources": [
                "Machine Learning Specialization (Coursera)",
                "Deep Learning (Book)",
                "Fast.ai Practical Deep Learning"
            ],
            "youtube_links": [
                "https://youtu.be/7dSJubxFWv0?si=4P9JaNaJ3q3qE6OP",
                "https://youtu.be/5NgNicANyqM?si=OCoRyOsQKOjMzq1K",
                "https://youtu.be/SpfIwlAYaKk?si=vCXlMOhNkSq-Vrak"
            ]
        },
        "Data Scientist": {
            "description": "Data scientists use statistical methods, machine learning, and analytical approaches to extract insights from complex data.",
            "levels": {
                "Beginner": ["Statistics", "Python/R", "Data Manipulation", "Basic Visualization"],
                "Intermediate": ["Machine Learning", "Experimental Design", "Feature Engineering", "SQL"],
                "Advanced": ["Deep Learning", "Big Data Technologies", "Deployment Strategies", "Advanced Statistics"]
            },
            "timeline": "12-18 months",
            "resources": [
                "Data Science Specialization (Coursera)",
                "Python for Data Analysis (Book)",
                "Introduction to Statistical Learning (Book)"
            ],
            "youtube_links": [
                "https://youtu.be/7dSJubxFWv0?si=RWB2jWuZ3seCndbQ"
            ]
        },
        "Data Engineer": {
            "description": "Data engineers design, build, and maintain data pipelines and infrastructure that enable data analysis and machine learning.",
            "levels": {
                "Beginner": ["SQL", "Python/Java/Scala", "Database Fundamentals", "Linux Basics"],
                "Intermediate": ["ETL Processes", "Data Warehousing", "Big Data Technologies", "Cloud Data Services"],
                "Advanced": ["Data Pipeline Architecture", "Stream Processing", "Data Governance", "Distributed Systems"]
            },
            "timeline": "12-18 months",
            "resources": [
                "Data Engineering Roadmap (roadmap.sh)",
                "Fundamentals of Data Engineering (Book)",
                "Data Engineering Zoomcamp"
            ],
            "youtube_links": [
                "https://youtu.be/Tyg1FVNq40g?si=PJwwwpeYZKZVrFQt"
            ]
        },
        "Machine Learning Engineer": {
            "description": "Machine learning engineers design, build, and deploy machine learning models and systems at scale.",
            "levels": {
                "Beginner": ["Python", "Linear Algebra", "Statistics", "ML Fundamentals"],
                "Intermediate": ["Deep Learning Frameworks", "Model Training", "Feature Engineering", "MLOps Basics"],
                "Advanced": ["Model Deployment", "Distributed Training", "Production Systems", "Research Implementation"]
            },
            "timeline": "12-18 months",
            "resources": [
                "Machine Learning Engineering Roadmap (roadmap.sh)",
                "MLOps: Machine Learning Operations",
                "Designing Machine Learning Systems (Book)"
            ],
            "youtube_links": [
                "https://youtu.be/GwIo3gDZCVQ?si=YOzBWRDJ8NK2BG9D"
            ]
        },
        "Android Developer": {
            "description": "Android developers create applications for the Android operating system using Java, Kotlin, and Android SDK.",
            "levels": {
                "Beginner": ["Java/Kotlin", "Android Studio", "Basic UI Components", "Activity Lifecycle"],
                "Intermediate": ["RecyclerView", "Networking", "Database (Room/SQLite)", "Material Design"],
                "Advanced": ["Architecture Components", "Dependency Injection", "Testing", "Performance Optimization"]
            },
            "timeline": "8-14 months",
            "resources": [
                "Android Developer Roadmap (roadmap.sh)",
                "Android Developer Documentation",
                "Kotlin for Android Developers"
            ],
            "youtube_links": ["https://youtu.be/blKkRoZPxLc?si=ZI0s3Ngn5jQepXX4"]
        },
        "iOS Developer": {
            "description": "iOS developers build applications for Apple's iOS operating system using Swift and Apple's development tools.",
            "levels": {
                "Beginner": ["Swift Programming", "Xcode Basics", "UIKit Fundamentals", "Auto Layout"],
                "Intermediate": ["Networking (URLSession)", "Core Data", "Concurrency", "Design Patterns"],
                "Advanced": ["SwiftUI", "Combine Framework", "App Architecture", "Performance Tuning"]
            },
            "timeline": "8-14 months",
            "resources": [
                "iOS Developer Roadmap (roadmap.sh)",
                "Apple Developer Documentation",
                "Hacking with Swift"
            ],
            "youtube_links": ["https://youtu.be/blKkRoZPxLc?si=ZI0s3Ngn5jQepXX4"]
        },
        "Blockchain Developer": {
            "description": "Blockchain developers build decentralized applications and smart contracts on blockchain platforms.",
            "levels": {
                "Beginner": ["Blockchain Fundamentals", "Cryptography Basics", "Smart Contract Concepts", "Web3 Basics"],
                "Intermediate": ["Solidity", "Ethereum Development", "Truffle/Hardhat", "DeFi Concepts"],
                "Advanced": ["Scalability Solutions", "Security Auditing", "Token Economics", "Cross-chain Development"]
            },
            "timeline": "10-16 months",
            "resources": [
                "Blockchain Developer Roadmap (roadmap.sh)",
                "Ethereum Developer Documentation",
                "CryptoZombies (Tutorial)"
            ],
            "youtube_links": ["https://youtu.be/M576WGiDBdQ?si=LpsTSISwqhVzHbqk"]
        },
        "QA Engineer": {
            "description": "QA engineers ensure software quality through testing, automation, and quality assurance processes.",
            "levels": {
                "Beginner": ["Testing Fundamentals", "Manual Testing", "Bug Tracking", "Test Cases Design"],
                "Intermediate": ["Automation Testing", "Selenium/Cypress", "API Testing", "Performance Testing"],
                "Advanced": ["Test Strategy", "CI/CD Integration", "Test Architecture", "Quality Metrics"]
            },
            "timeline": "6-12 months",
            "resources": [
                "Software Testing Roadmap (roadmap.sh)",
                "The Art of Software Testing (Book)",
                "Test Automation University"
            ],
            "youtube_links": ["https://www.youtube.com/live/HmQv8Z4om4I?si=yQDyXaMImI-3lz6A"]
        },
        "Software Architect": {
            "description": "Software architects design the overall structure and systems of software applications and make high-level design choices.",
            "levels": {
                "Beginner": ["Programming Proficiency", "Basic Design Patterns", "System Components", "Database Design"],
                "Intermediate": ["Architecture Patterns", "Microservices", "API Design", "Cloud Infrastructure"],
                "Advanced": ["System Scaling", "Performance Optimization", "Security Architecture", "Technical Leadership"]
            },
            "timeline": "5+ years experience",
            "resources": [
                "Software Architect Roadmap (roadmap.sh)",
                "Clean Architecture (Book)",
                "Designing Data-Intensive Applications (Book)"
            ],
            "youtube_links": ["https://youtu.be/Xx1eYBlUGO8?si=FTtg_dYvNMuZgR-U"]
        },
        "Cyber Security Specialist": {
            "description": "Cyber security specialists protect computer systems and networks from information disclosure, theft, or damage.",
            "levels": {
                "Beginner": ["Networking Fundamentals", "Operating Systems", "Security Concepts", "Basic Scripting"],
                "Intermediate": ["Penetration Testing", "Security Tools", "Vulnerability Assessment", "Incident Response"],
                "Advanced": ["Digital Forensics", "Threat Intelligence", "Security Architecture", "Compliance frameworks"]
            },
            "timeline": "12-18 months",
            "resources": [
                "Cyber Security Roadmap (roadmap.sh)",
                "TryHackMe/CyberSecLabs",
                "The Web Application Hacker's Handbook"
            ],
            "youtube_links": ["https://www.youtube.com/live/bDEoUlD53P0?si=itkzyjkoqCbsTDyy"]
        },
        "UX Designer": {
            "description": "UX designers focus on creating meaningful and relevant experiences for users of digital products.",
            "levels": {
                "Beginner": ["Design Principles", "User Research", "Wireframing", "Prototyping"],
                "Intermediate": ["UI Design", "Interaction Design", "Usability Testing", "Design Systems"],
                "Advanced": ["Service Design", "Information Architecture", "Accessibility", "Design Leadership"]
            },
            "timeline": "8-14 months",
            "resources": [
                "UX Designer Roadmap (roadmap.sh)",
                "The Design of Everyday Things (Book)",
                "Nielsen Norman Group Articles"
            ],
            "youtube_links": ["https://youtu.be/TJtEQ1p1hw4?si=VURiggB3yy_qUqqT"]
        },
        "Technical Writer": {
            "description": "Technical writers create documentation, manuals, and other content that explains complex technical information clearly.",
            "levels": {
                "Beginner": ["Writing Fundamentals", "Technical Comprehension", "Basic Tools", "Audience Analysis"],
                "Intermediate": ["Documentation Systems", "API Documentation", "Tutorial Creation", "Visual Communication"],
                "Advanced": ["Information Architecture", "Content Strategy", "Localization", "Developer Experience"]
            },
            "timeline": "6-12 months",
            "resources": [
                "Technical Writing Roadmap (roadmap.sh)",
                "Microsoft Writing Style Guide",
                "Documentation System Tutorials"
            ],
            "youtube_links": ["https://youtu.be/vT5pcc30Ffw?si=Yw5KlXfKo8cydxMz"]
        },
        "Game Developer": {
            "description": "Game developers create video games for various platforms using game engines and programming skills.",
            "levels": {
                "Beginner": ["Programming Basics", "Game Engines (Unity/Unreal)", "Basic Game Mechanics", "2D/3D Assets"],
                "Intermediate": ["Game Physics", "AI Programming", "Multiplayer Networking", "Performance Optimization"],
                "Advanced": ["Advanced Rendering", "Procedural Generation", "VR/AR Development", "Game Architecture"]
            },
            "timeline": "12-18 months",
            "resources": [
                "Game Developer Roadmap (roadmap.sh)",
                "Unity/Unreal Documentation",
                "Game Programming Patterns (Book)"
            ],
            "youtube_links": ["https://youtu.be/gB1F9G0JXOo?si=9SPe5k9ajhMFMO_y"]
        },
        "MLOps Engineer": {
            "description": "MLOps engineers focus on deploying, maintaining, and monitoring machine learning models in production environments.",
            "levels": {
                "Beginner": ["ML Fundamentals", "Python", "Basic DevOps", "Containerization"],
                "Intermediate": ["Model Deployment", "CI/CD for ML", "Monitoring", "Data Versioning"],
                "Advanced": ["Orchestration", "Feature Stores", "Model Governance", "Scalable Infrastructure"]
            },
            "timeline": "12-18 months",
            "resources": [
                "MLOps Roadmap (roadmap.sh)",
                "MLOps: Machine Learning Operations",
                "Introducing MLOps (Book)"
            ],
            "youtube_links": ["https://youtu.be/w71RHxAWxaM?si=BuxUPKIAeD7PGzGF"]
        },
        "Product Manager": {
            "description": "Product managers guide the success of a product and lead the cross-functional team that is responsible for improving it.",
            "levels": {
                "Beginner": ["Product Thinking", "Market Research", "User Stories", "Basic Analytics"],
                "Intermediate": ["Roadmapping", "Prioritization", "Stakeholder Management", "A/B Testing"],
                "Advanced": ["Product Strategy", "Metrics & KPIs", "Experimentation Culture", "Leadership"]
            },
            "timeline": "2+ years experience",
            "resources": [
                "Product Manager Roadmap (roadmap.sh)",
                "Inspired: How to Create Tech Products Customers Love (Book)",
                "Product School Resources"
            ],
            "youtube_links": ["https://youtu.be/HZpHhRFQtvc?si=32xFvtqRoeIZ0PFv"]
        },
        "Engineering Manager": {
            "description": "Engineering managers lead engineering teams, manage projects, and help engineers grow in their careers.",
            "levels": {
                "Beginner": ["Technical Leadership", "Project Management", "Team Dynamics", "1:1 Meetings"],
                "Intermediate": ["Hiring & Onboarding", "Performance Management", "Technical Strategy", "Process Improvement"],
                "Advanced": ["Organizational Design", "Change Management", "Executive Communication", "Culture Building"]
            },
            "timeline": "5+ years experience",
            "resources": [
                "Engineering Manager Roadmap (roadmap.sh)",
                "The Manager's Path (Book)",
                "Staff Engineer: Leadership beyond management (Book)"
            ],
            "youtube_links": ["https://youtu.be/kZdFRXeuCuY?si=bls9KnobodoiZtqS"]
        },
        "Developer Relations": {
            "description": "Developer relations professionals build relationships with developer communities and create resources to help them succeed.",
            "levels": {
                "Beginner": ["Technical Communication", "Community Engagement", "Content Creation", "Basic Marketing"],
                "Intermediate": ["Developer Marketing", "Event Management", "Advocacy", "Metrics Tracking"],
                "Advanced": ["Strategy Development", "Partnership Building", "Influencer Relations", "Ecosystem Growth"]
            },
            "timeline": "3+ years experience",
            "resources": [
                "DevRel Roadmap (roadmap.sh)",
                "The Business Value of Developer Relations (Book)",
                "DevRel Collective Resources"
            ],
            "youtube_links": ["https://youtu.be/JPrVdSNw-Cc?si=kUazF1ZmS1wxKQLV"]
        },
        "BI Analyst": {
            "description": "Business Intelligence analysts transform data into insights that drive business value through reporting and analytics.",
            "levels": {
                "Beginner": ["SQL", "Data Visualization", "Reporting Tools", "Business Metrics"],
                "Intermediate": ["Data Modeling", "ETL Processes", "Dashboard Design", "Statistical Analysis"],
                "Advanced": ["Predictive Analytics", "Data Storytelling", "Strategy Influence", "Self-service BI"]
            },
            "timeline": "8-14 months",
            "resources": [
                "BI Analyst Roadmap (roadmap.sh)",
                "Tableau/Power BI Documentation",
                "Storytelling with Data (Book)"
            ],
            "youtube_links": [
                "https://youtu.be/MBblN98-5lg?si=9P8uNHmWYhhytIb9"
            ]
        }
    }

if Config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD

# ------------------------
# Translation Service
# ------------------------
class TranslationService:
    def __init__(self):
        self.translator = None
        self.fallback_translations = {
            'en': 'English', 'ta': 'Tamil', 'hi': 'Hindi', 'te': 'Telugu',
            'ml': 'Malayalam', 'kn': 'Kannada', 'es': 'Spanish', 'fr': 'French',
            'de': 'German', 'zh-cn': 'Chinese', 'ja': 'Japanese', 'ar': 'Arabic'
        }
    
    def translate_text(self, text: str, target_lang: str, max_retries: int = 3) -> str:
        """Translate text using Google Translate with fallback"""
        if not text or target_lang == 'en':
            return text
        
        # Try Google Translate with deep-translator
        for attempt in range(max_retries):
            try:
                if target_lang == 'zh-cn':
                    target_lang = 'zh-CN'  # deep-translator uses different code
                
                translation = DeepGoogleTranslator(source='auto', target=target_lang).translate(text)
                return translation
            except Exception as e:
                if attempt == max_retries - 1:
                    st.warning(f"Google Translate failed: {e}. Using fallback translation.")
                    break
                time.sleep(1)  # Wait before retry
        
        # Fallback: simple word substitution (for demo purposes)
        return self._fallback_translation(text, target_lang)
    
    def _fallback_translation(self, text: str, target_lang: str) -> str:
        """Simple fallback translation for demonstration"""
        # This is a very basic fallback - in production, you'd want a better solution
        lang_name = self.fallback_translations.get(target_lang, target_lang)
        return f"[Translated to {lang_name}] {text}"

# ------------------------
# PDF & Image Processor
# ------------------------
class PDFProcessor:
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page_index in range(len(doc)):
                page = doc[page_index]
                page_text = page.get_text()
                if page_text and page_text.strip():
                    text += f"\n[Page {page_index + 1}]\n{page_text}"
            doc.close()
            return text
        except Exception as e:
            st.error(f"Error extracting text from {pdf_path}: {e}")
            return ""

    @staticmethod
    def extract_text_from_image(image_path_or_bytes) -> str:
        """Use pytesseract to extract text from an image file path or bytes"""
        try:
            if isinstance(image_path_or_bytes, (bytes, bytearray)):
                img = Image.open(io.BytesIO(image_path_or_bytes))
            else:
                img = Image.open(image_path_or_bytes)
            text = pytesseract.image_to_string(img)
            return text
        except Exception as e:
            st.error(f"Image OCR error: {e}")
            return ""

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]]', '', text)
        return text.strip()

    @staticmethod
    def chunk_text(text: str, chunk_size: int = Config.CHUNK_SIZE,
                   overlap_words: int = Config.CHUNK_OVERLAP_WORDS) -> List[Dict]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        chunk_id = 0

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += (" " + sentence).strip()
            else:
                if current_chunk:
                    page_match = re.search(r'\[Page (\d+)\]', current_chunk)
                    page_num = int(page_match.group(1)) if page_match else None
                    chunks.append({
                        'id': chunk_id,
                        'text': current_chunk.strip(),
                        'page': page_num,
                        'length': len(current_chunk)
                    })
                    chunk_id += 1

                    # create overlap: last overlap_words words
                    words = current_chunk.split()
                    overlap = words[-overlap_words:] if len(words) > overlap_words else words
                    current_chunk = " ".join(overlap) + " " + sentence
                else:
                    current_chunk = sentence

        if current_chunk:
            page_match = re.search(r'\[Page (\d+)\]', current_chunk)
            page_num = int(page_match.group(1)) if page_match else None
            chunks.append({
                'id': chunk_id,
                'text': current_chunk.strip(),
                'page': page_num,
                'length': len(current_chunk)
            })

        return chunks

# ------------------------
# Vector store + embeddings
# ------------------------
class VectorStore:
    def __init__(self):
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.index = None
        self.chunks: List[Dict] = []
        self.document_metadata = {}

    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        texts = [c['text'] for c in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return np.array(embeddings, dtype='float32')

    def build_index(self, chunks: List[Dict], doc_name: str):
        # tag chunks with doc name
        for c in chunks:
            c['document'] = doc_name

        embeddings = self.create_embeddings(chunks)

        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.chunks = chunks
            self.index.add(embeddings)
        else:
            self.chunks.extend(chunks)
            self.index.add(embeddings)

        self.document_metadata[doc_name] = {
            'num_chunks': len(chunks),
            'processed_at': datetime.now().isoformat()
        }

    def search(self, query: str, k: int = Config.MAX_CHUNKS_TO_RETRIEVE) -> List[Dict]:
        if self.index is None or self.index.ntotal == 0:
            return []

        q_emb = self.embedding_model.encode([query]).astype('float32')
        k_search = min(k, self.index.ntotal)
        distances, indices = self.index.search(q_emb, k_search)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(1 / (1 + dist))
                results.append(chunk)
        return results

    def save_index(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        if self.index:
            faiss.write_index(self.index, os.path.join(path, 'index.faiss'))
        with open(os.path.join(path, 'metadata.pkl'), 'wb') as f:
            pickle.dump({'chunks': self.chunks, 'document_metadata': self.document_metadata}, f)

    def load_index(self, path: str) -> bool:
        try:
            index_path = os.path.join(path, 'index.faiss')
            metadata_path = os.path.join(path, 'metadata.pkl')
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                self.index = faiss.read_index(index_path)
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.chunks = data.get('chunks', [])
                    self.document_metadata = data.get('document_metadata', {})
                return True
        except Exception as e:
            st.error(f"Failed to load index: {e}")
        return False

# ------------------------
# Watsonx LLM wrapper (with fallback)
# ------------------------
class WatsonxLLM:
    def __init__(self, api_key: str = None, project_id: str = None, url: str = None):
        self.credentials = None
        self.model = None
        
        if IBM_WATSON_AVAILABLE and api_key and project_id and url:
            try:
                self.credentials = Credentials(api_key=api_key, url=url)
                self.model_params = {
                    GenParams.MAX_NEW_TOKENS: 1024,
                    GenParams.MIN_NEW_TOKENS: 30,
                    GenParams.TEMPERATURE: 0.2,
                    GenParams.TOP_P: 0.9,
                }
                self.model = Model(
                    model_id=Config.WATSONX_MODEL,
                    params=self.model_params,
                    credentials=self.credentials,
                    project_id=project_id
                )
            except Exception as e:
                st.error(f"Failed to initialize Watsonx: {e}")
                self.model = None

    def _safe_json_parse(self, text: str) -> Dict:
        """Safely parse JSON from text, handling various formats"""
        try:
            # Try to find JSON object in the text
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            
            # If no JSON object found, try to find JSON array
            json_array_match = re.search(r'\[[\s\S]*\]', text)
            if json_array_match:
                json_str = json_array_match.group(0)
                return json.loads(json_str)
                
            # If no JSON found, return the text as answer
            return {'answer': text.strip(), 'key_points': [], 'limitations': ''}
        except json.JSONDecodeError:
            # If JSON parsing fails, return the text as answer
            return {'answer': text.strip(), 'key_points': [], 'limitations': ''}

    def generate_text_from_prompt(self, prompt: str) -> str:
        if self.model:
            try:
                resp = self.model.generate_text(prompt=prompt)
                if isinstance(resp, str):
                    return resp
                if hasattr(resp, "text"):
                    return resp.text
                return str(resp)
            except Exception as e:
                return f"Error generating text: {e}"
        else:
            # Mock response for demo purposes
            return '''{
                "answer": "This is a mock response since Watsonx is not properly configured. Please check your API credentials.",
                "key_points": ["Check API key", "Verify project ID", "Ensure correct URL"],
                "limitations": "Running in demo mode without Watsonx integration"
            }'''

    def generate_answer(self, question: str, context: List[Dict]) -> Dict:
        context_text = "\n\n".join([
            f"[Source: {c['document']}, Page {c.get('page','N/A')}]\n{c['text']}"
            for c in context
        ])
        prompt = f"""
You are an academic assistant. Using the provided context, answer the user's question clearly and cite the source pages.
If the context doesn't fully answer the question, use your general knowledge to provide a comprehensive answer.

Context:
{context_text}

Question: {question}

Please produce:
1) A concise answer (2-6 sentences).
2) Bullet key-points extracted from the context (3-6 bullets).
3) Any missing information or limitations (if context doesn't fully answer).
Format your response as JSON with keys: "answer", "key_points", "limitations".
"""
        out = self.generate_text_from_prompt(prompt)
        return self._safe_json_parse(out)

    def generate_general_answer(self, question: str) -> Dict:
        """Generate answer using general knowledge without document context"""
        prompt = f"""
You are an expert academic assistant. Answer the following question clearly and comprehensively.
If you don't know the answer, say so rather than making up information.

Question: {question}

Please provide:
1) A clear, concise answer (3-8 sentences)
2) Key points about the topic (3-6 bullet points)
3) Any limitations or uncertainties in your answer (if applicable)

Format your response as JSON with keys: "answer", "key_points", "limitations".
"""
        out = self.generate_text_from_prompt(prompt)
        return self._safe_json_parse(out)

    def generate_quiz(self, context: List[Dict], num_questions: int = 5) -> List[Dict]:
        context_text = "\n\n".join([c['text'] for c in context])
        prompt = f"""
Generate {num_questions} multiple-choice questions (with 4 options each) and answers from the following academic context.
Return a valid JSON array of objects with this structure: 
[{{ 
    "question": "question text", 
    "options": ["option1", "option2", "option3", "option4"], 
    "answer": "correct option letter (a, b, c, or d)", 
    "explanation": "brief explanation of why this is correct" 
}}]

Context:
{context_text}

IMPORTANT: Return ONLY valid JSON, no other text.
"""
        out = self.generate_text_from_prompt(prompt)
        try:
            json_match = re.search(r'\[.*\]', out, flags=re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            return []
        except Exception as e:
            st.error(f"Error parsing quiz JSON: {e}")
            return []

    def generate_questions_short(self, context: List[Dict], num_questions: int = 8) -> List[Dict]:
        context_text = "\n\n".join([c['text'] for c in context])
        prompt = f"""
From the context, generate {num_questions} short-answer questions suitable for study (no options).
Return a valid JSON array with this structure: 
[{{"question":"question text","expected_answer":"expected answer text"}}]

Context:
{context_text}

IMPORTANT: Return ONLY valid JSON, no other text.
"""
        out = self.generate_text_from_prompt(prompt)
        try:
            json_match = re.search(r'\[.*\]', out, flags=re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            return []
        except Exception as e:
            st.error(f"Error parsing questions JSON: {e}")
            return []

    def generate_study_schedule(self, context: List[Dict], days_available: int, hours_per_day: int) -> Dict:
        context_text = "\n\n".join([c['text'] for c in context])
        prompt = f"""
Based on the following academic content, create a detailed study schedule for a student.
The student has {days_available} days to study and can study for {hours_per_day} hours per day.

Context:
{context_text}

Please create a study plan that:
1. Breaks down the content into logical topics or chapters
2. Allocates time based on topic complexity and importance
3. Includes review sessions
4. Suggests specific study techniques for different content types

Return your response as a JSON object with this structure:
{{
  "total_hours": total_study_hours,
  "topics": [
    {{
      "topic_name": "name of topic",
      "complexity": "low/medium/high",
      "hours_allocated": number_of_hours,
      "study_day": day_number,
      "study_techniques": ["technique1", "technique2"]
    }}
  ],
  "daily_schedule": [
    {{
      "day": day_number,
      "topics_to_cover": ["topic1", "topic2"],
      "total_hours": hours_for_day,
      "review_topics": ["topic_to_review"]
    }}
  ]
}}

IMPORTANT: Return ONLY valid JSON, no other text or markdown formatting.
"""
        out = self.generate_text_from_prompt(prompt)
        try:
            # Clean the JSON string before parsing - handle markdown code blocks
            json_str = out.strip()
            
            # Remove any markdown code blocks (both outer and inner)
            json_str = re.sub(r'^```json\s*|\s*```$', '', json_str, flags=re.IGNORECASE)
            json_str = re.sub(r'^```\s*|\s*```$', '', json_str)
            
            # Remove any leading/trailing whitespace that might remain
            json_str = json_str.strip()
            
            # Try to parse the JSON
            return json.loads(json_str)
        except Exception as e:
            st.error(f"Error parsing study schedule JSON: {e}")
            st.text(f"Raw output: {out}")
            # Return a default schedule as fallback
            return {
                "total_hours": days_available * hours_per_day,
                "topics": [
                    {
                        "topic_name": "General Review",
                        "complexity": "medium",
                        "hours_allocated": days_available * hours_per_day,
                        "study_day": 1,
                        "study_techniques": ["Reading", "Note-taking", "Practice questions"]
                    }
                ],
                "daily_schedule": [
                    {
                        "day": 1,
                        "topics_to_cover": ["General Review"],
                        "total_hours": hours_per_day,
                        "review_topics": []
                    }
                ]
            }

    def explain_image_text(self, image_text: str, question: str = None) -> Dict:
        prompt = f"""
You are an academic tutor. A student has provided text extracted from an image and needs help understanding it.

Extracted text from image:
{image_text}

{'The student specifically asked: ' + question if question else 'Please provide a comprehensive explanation of this content.'}

Please provide:
1. A clear explanation of the main concepts
2. Key terms and their definitions
3. Any formulas, diagrams, or important relationships
4. Practical examples or applications if relevant

Format your response as JSON with keys: "explanation", "key_terms", "important_concepts", "examples".
"""
        out = self.generate_text_from_prompt(prompt)
        try:
            # Clean the JSON string before parsing
            json_str = out.strip()
            # Remove any markdown code blocks
            json_str = re.sub(r'^```json\s*|\s*```$', '', json_str, flags=re.IGNORECASE)
            json_str = re.sub(r'^```\s*|\s*```$', '', json_str)
            
            # Try to parse the JSON
            return json.loads(json_str)
        except Exception as e:
            st.error(f"Error parsing image explanation JSON: {e}")
            return {
                "explanation": f"Explanation of the extracted text: {image_text[:200]}...",
                "key_terms": ["Term1", "Term2", "Term3"],
                "important_concepts": ["Concept1", "Concept2"],
                "examples": ["Example1", "Example2"]
            }

# ------------------------
# Enhanced Study Scheduler
# ------------------------
class EnhancedStudyScheduler:
    """Enhanced Study Scheduler with better display formatting"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def create_schedule(self, context, days_available, hours_per_day):
        return self.llm.generate_study_schedule(context, days_available, hours_per_day)
    
    def display_schedule(self, schedule):
        """Enhanced display with calendar view and better formatting"""
        if not schedule:
            st.warning("No schedule generated")
            return
        
        # Overview Cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Study Hours", f"{schedule.get('total_hours', 0)} hours")
        with col2:
            st.metric("Total Topics", len(schedule.get('topics', [])))
        with col3:
            st.metric("Study Days", len(schedule.get('daily_schedule', [])))
        
        # Topics Breakdown with color coding
        st.subheader("📚 Topics Overview")
        if 'topics' in schedule:
            for topic in schedule['topics']:
                complexity = topic.get('complexity', 'medium')
                color = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}.get(complexity, '⚪')
                
                with st.expander(f"{color} {topic.get('topic_name', 'Unknown Topic')} - {topic.get('hours_allocated', 0)} hours"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Complexity:** {complexity.capitalize()}")
                        st.write(f"**Study Day:** Day {topic.get('study_day', 'N/A')}")
                    with col2:
                        st.write("**Study Techniques:**")
                        for technique in topic.get('study_techniques', []):
                            st.write(f"• {technique}")
        
        # Daily Schedule - Calendar View
        st.subheader("📅 Daily Study Plan")
        if 'daily_schedule' in schedule:
            # Create tabs for each day
            day_tabs = st.tabs([f"Day {day['day']}" for day in schedule['daily_schedule']])
            
            for idx, day in enumerate(schedule['daily_schedule']):
                with day_tabs[idx]:
                    # Day overview
                    st.markdown(f"### Day {day.get('day', 'N/A')} Study Plan")
                    st.markdown(f"**Total Hours:** {day.get('total_hours', 0)} hours")
                    
                    # Topics for the day
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("📖 **Topics to Study:**")
                        topics_to_cover = day.get('topics_to_cover', [])
                        if topics_to_cover:
                            for topic in topics_to_cover:
                                st.write(f"• {topic}")
                        else:
                            st.write("No topics scheduled")
                    
                    with col2:
                        st.markdown("🔄 **Review Topics:**")
                        review_topics = day.get('review_topics', [])
                        if review_topics:
                            for topic in review_topics:
                                st.write(f"• {topic}")
                        else:
                            st.write("No reviews scheduled")
                    
                    # Progress tracker placeholder
                    st.markdown("---")
                    st.checkbox(f"Mark Day {day.get('day')} as completed", 
                               key=f"day_{day.get('day')}_complete")
        
        # Summary Statistics
        st.subheader("📊 Study Statistics")
        if 'topics' in schedule:
            complexity_counts = {'low': 0, 'medium': 0, 'high': 0}
            for topic in schedule['topics']:
                complexity = topic.get('complexity', 'medium')
                if complexity in complexity_counts:
                    complexity_counts[complexity] += 1
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Easy Topics", complexity_counts['low'], 
                         help="Topics marked as low complexity")
            with col2:
                st.metric("Medium Topics", complexity_counts['medium'],
                         help="Topics marked as medium complexity")
            with col3:
                st.metric("Hard Topics", complexity_counts['high'],
                         help="Topics marked as high complexity")
        
        # Weekly View Summary
        st.subheader("📋 Week-at-a-Glance")
        if 'daily_schedule' in schedule:
            week_data = []
            for day in schedule['daily_schedule'][:7]:  # Show first week
                week_data.append({
                    'Day': f"Day {day.get('day', 'N/A')}",
                    'Hours': day.get('total_hours', 0),
                    'Topics': len(day.get('topics_to_cover', [])),
                    'Reviews': len(day.get('review_topics', []))
                })
            
            if week_data:
                import pandas as pd
                week_df = pd.DataFrame(week_data)
                st.dataframe(week_df, use_container_width=True, hide_index=True)

# ------------------------
# Career Guidance Module
# ------------------------
class CareerGuidance:
    """Career guidance module with roadmap explanations and YouTube integration"""
    
    def __init__(self):
        self.career_data = Config.CAREER_ROADMAPS
    
    def display_career_path(self, career_name):
        """Display detailed career path information"""
        if career_name not in self.career_data:
            st.error(f"Career path '{career_name}' not found.")
            return
        
        career = self.career_data[career_name]
        
        # Header with career name
        st.header(f"🧭 {career_name} Career Path")
        
        # Description
        st.subheader("📝 Description")
        st.write(career["description"])
        
        # Timeline
        st.subheader("⏱️ Timeline")
        st.info(f"Expected learning timeline: {career['timeline']}")
        
        # Learning Path by Levels
        st.subheader("📚 Learning Path")
        
        for level, skills in career["levels"].items():
            with st.expander(f"{level} Level"):
                for skill in skills:
                    st.write(f"• {skill}")
        
        # Resources
        st.subheader("📖 Recommended Resources")
        for resource in career["resources"]:
            st.write(f"• {resource}")
        
        # YouTube Videos
        if career.get("youtube_links"):
            st.subheader("🎥 YouTube Learning Resources")
            for i, video_url in enumerate(career["youtube_links"]):
                # Extract video ID for embedding
                video_id = None
                if "youtu.be/" in video_url:
                    video_id = video_url.split("youtu.be/")[1].split("?")[0]
                elif "youtube.com/watch?v=" in video_url:
                    video_id = video_url.split("youtube.com/watch?v=")[1].split("&")[0]
                
                if video_id:
                    st.video(f"https://www.youtube.com/watch?v={video_id}")
                else:
                    st.write(f"• [Video Link {i+1}]({video_url})")
        
        # Generate Personalized Plan Button
        if st.button("🎯 Generate Personalized Learning Plan", key=f"plan_{career_name}"):
            self.generate_personalized_plan(career_name, career)
    
    def generate_personalized_plan(self, career_name, career_data):
        """Generate a personalized learning plan for the career"""
        st.subheader(f"📋 Personalized {career_name} Learning Plan")
        
        # Calculate timeline based on available time
        col1, col2 = st.columns(2)
        with col1:
            hours_per_week = st.slider("Hours available per week", 5, 40, 15, key=f"hours_{career_name}")
        with col2:
            start_level = st.selectbox("Current proficiency level", 
                                     list(career_data["levels"].keys()),
                                     key=f"level_{career_name}")
        
        # Generate plan based on inputs
        levels = list(career_data["levels"].keys())
        start_index = levels.index(start_level)
        
        total_weeks = 0
        for i in range(start_index, len(levels)):
            level = levels[i]
            skills_count = len(career_data["levels"][level])
            # Estimate 2-3 weeks per skill depending on complexity
            weeks_needed = max(2, skills_count * 2 // hours_per_week)
            total_weeks += weeks_needed
            
            st.write(f"**{level} Level**: {weeks_needed} weeks")
            for skill in career_data["levels"][level]:
                st.write(f"  - {skill}")
        
        st.success(f"Estimated total time to mastery: {total_weeks} weeks (~{total_weeks//4} months)")
        
        # Weekly study plan
        st.subheader("📅 Weekly Study Plan Example")
        st.write("""
        - Monday: Theory and concepts (2 hours)
        - Wednesday: Practical exercises (2 hours)  
        - Saturday: Projects and implementation (3 hours)
        - Sunday: Review and practice (2 hours)
        """)
    
    def display_career_selector(self):
        """Display career selection interface"""
        st.subheader("🎯 Select a Career Path")
        
        # Group careers by category
        categories = {
            "Development": ["Frontend Developer", "Backend Developer", "Full Stack Developer", 
                          "Android Developer", "iOS Developer", "Blockchain Developer"],
            "Data & AI": ["Data Analyst", "Data Scientist", "Data Engineer", 
                         "AI Engineer", "Machine Learning Engineer", "MLOps Engineer"],
            "DevOps & Infrastructure": ["DevOps Engineer", "Software Architect"],
            "Other Tech Roles": ["QA Engineer", "Cyber Security Specialist", "UX Designer", 
                               "Technical Writer", "Game Developer", "BI Analyst"],
            "Management": ["Product Manager", "Engineering Manager", "Developer Relations"]
        }
        
        # Create tabs for each category
        category_tabs = st.tabs(list(categories.keys()))
        
        for i, (category, careers) in enumerate(categories.items()):
            with category_tabs[i]:
                for career in careers:
                    if st.button(f"🧭 {career}", key=f"btn_{career}", use_container_width=True):
                        st.session_state['selected_career'] = career
        
        # Display selected career
        if 'selected_career' in st.session_state:
            self.display_career_path(st.session_state['selected_career'])

# ------------------------
# Utilities
# ------------------------
def safe_translate(text: str, dest_lang: str, translation_service: TranslationService) -> str:
    """Safely translate text, handling None values and errors"""
    if not text or text.strip() == "" or dest_lang == 'en':
        return text
    try:
        return translation_service.translate_text(text, dest_lang)
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

def transcribe_audio_file(uploaded_file) -> str:
    """Accepts an uploaded audio file (st.file_uploader) and returns transcription (via SpeechRecognition)."""
    try:
        # Create a temporary file with the correct extension
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # Supported audio formats
        supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.wma']
        
        if file_ext not in supported_formats:
            st.error(f"Unsupported audio format: {file_ext}. Please use: {', '.join(supported_formats)}")
            return ""
            
        # Convert to wav if necessary using pydub
        wav_path = tmp_path
        if file_ext != '.wav':
            try:
                audio = AudioSegment.from_file(tmp_path)
                wav_path = tmp_path + ".wav"
                audio.export(wav_path, format="wav")
            except Exception as e:
                st.error(f"Error converting audio to WAV: {e}")
                # Try direct transcription without conversion
                try:
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(tmp_path) as source:
                        audio_data = recognizer.record(source)
                        text = recognizer.recognize_google(audio_data)
                    return text
                except Exception as e2:
                    st.error(f"Direct transcription also failed: {e2}")
                    return ""

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        # Clean up temporary files
        try:
            os.unlink(tmp_path)
            if wav_path != tmp_path and os.path.exists(wav_path):
                os.unlink(wav_path)
        except:
            pass
            
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Error with speech recognition service: {e}"
    except Exception as e:
        st.error(f"Audio transcription error: {e}")
        return ""

def record_audio() -> str:
    """Record audio from microphone and return transcription"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            return "No speech detected"
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Error with speech recognition service: {e}"
        except Exception as e:
            return f"Error: {e}"

def create_text_schedule(schedule):
    """Convert schedule to readable text format"""
    text = f"STUDY SCHEDULE\n"
    text += f"="*50 + "\n\n"
    text += f"Total Study Hours: {schedule.get('total_hours', 0)}\n\n"
    
    text += "TOPICS OVERVIEW:\n"
    text += "-"*30 + "\n"
    for topic in schedule.get('topics', []):
        text += f"\n📚 {topic.get('topic_name', 'Unknown')}\n"
        text += f"   Complexity: {topic.get('complexity', 'N/A')}\n"
        text += f"   Hours: {topic.get('hours_allocated', 0)}\n"
        text += f"   Study Day: {topic.get('study_day', 'N/A')}\n"
        text += f"   Techniques: {', '.join(topic.get('study_techniques', []))}\n"
    
    text += "\n\nDAILY SCHEDULE:\n"
    text += "-"*30 + "\n"
    for day in schedule.get('daily_schedule', []):
        text += f"\nDay {day.get('day', 'N/A')} ({day.get('total_hours', 0)} hours):\n"
        text += f"  Topics: {', '.join(day.get('topics_to_cover', []))}\n"
        if day.get('review_topics'):
            text += f"  Review: {', '.join(day.get('review_topics', []))}\n"
    
    return text

def create_csv_schedule(schedule):
    """Convert schedule to CSV format"""
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Day', 'Hours', 'Topics', 'Review Topics'])
    
    # Write daily schedule
    for day in schedule.get('daily_schedule', []):
        writer.writerow([
            f"Day {day.get('day', 'N/A')}",
            day.get('total_hours', 0),
            '; '.join(day.get('topics_to_cover', [])),
            '; '.join(day.get('review_topics', []))
        ])
    
    return output.getvalue()

# ------------------------
# Main QASystem orchestrator
# ------------------------
class QASystem:
    def __init__(self, watsonx_credentials: Dict):
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        self.llm = WatsonxLLM(
            api_key=watsonx_credentials.get('api_key'),
            project_id=watsonx_credentials.get('project_id'),
            url=watsonx_credentials.get('url')
        )
        self.scheduler = EnhancedStudyScheduler(self.llm)
        self.career_guidance = CareerGuidance()
        self.processed_documents = set()
        self.translation_service = TranslationService()
        self.english_answers = {}  # Store English answers for translation

    def process_pdf(self, pdf_file, file_name: str, as_image_texts: Optional[List[str]] = None) -> bool:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_path = tmp_file.name

            raw_text = self.pdf_processor.extract_text_from_pdf(tmp_path)
            if as_image_texts:
                # add OCR text extracted from images
                raw_text += "\n\n" + "\n\n".join(as_image_texts)

            if not raw_text.strip():
                st.warning("No text extracted from the PDF and provided images.")
                return False

            clean_text = self.pdf_processor.clean_text(raw_text)
            chunks = self.pdf_processor.chunk_text(clean_text)
            self.vector_store.build_index(chunks, file_name)
            self.processed_documents.add(file_name)

            os.unlink(tmp_path)
            return True
        except Exception as e:
            st.error(f"Error processing {file_name}: {e}")
            return False

    def process_image(self, uploaded_image) -> str:
        try:
            # Use PIL Image directly from BytesIO
            img = Image.open(uploaded_image)
            text = pytesseract.image_to_string(img)
            return text
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return ""

    def answer_question(self, question: str, translate_to: str = 'en', use_general_knowledge: bool = False) -> Dict:
        # Always get answer in English first
        if translate_to and translate_to != 'en':
            try:
                q_en = safe_translate(question, 'en', self.translation_service)
            except Exception:
                q_en = question
        else:
            q_en = question

        # Get answer either from documents or general knowledge
        if use_general_knowledge or not self.processed_documents:
            # Use Watson AI for general knowledge answer
            llm_output = self.llm.generate_general_answer(q_en)
        else:
            # Use document-based answer
            relevant_chunks = self.vector_store.search(q_en)
            llm_output = self.llm.generate_answer(q_en, relevant_chunks)
        
        answer_text = llm_output.get('answer', '')
        key_points = llm_output.get('key_points', [])
        limitations = llm_output.get('limitations', '')

        # Store English answer for translation
        answer_id = hashlib.md5(question.encode()).hexdigest()
        self.english_answers[answer_id] = {
            'answer': answer_text,
            'key_points': key_points,
            'limitations': limitations
        }

        # Prepare sources if using document-based answer
        sources = []
        if not use_general_knowledge and self.processed_documents:
            relevant_chunks = self.vector_store.search(q_en)
            for c in relevant_chunks:
                snippet = c['text'][:250] + ('...' if len(c['text']) > 250 else '')
                sources.append({'document': c.get('document', 'N/A'), 'page': c.get('page', 'N/A'), 'text_snippet': snippet, 'relevance': c.get('similarity_score', 0.0)})

        return {
            'answer': answer_text,
            'key_points': key_points,
            'limitations': limitations,
            'sources': sources,
            'is_general_knowledge': use_general_knowledge or not self.processed_documents,
            'answer_id': answer_id
        }

    def translate_answer(self, answer_id: str, target_lang: str) -> Dict:
        """Translate an existing answer to another language"""
        if answer_id not in self.english_answers:
            return None
            
        english_data = self.english_answers[answer_id]
        
        try:
            translated_answer = self.translation_service.translate_text(english_data['answer'], target_lang)
            translated_key_points = [self.translation_service.translate_text(kp, target_lang) for kp in english_data['key_points']]
            translated_limitations = self.translation_service.translate_text(english_data['limitations'], target_lang)
            
            return {
                'answer': translated_answer,
                'key_points': translated_key_points,
                'limitations': translated_limitations
            }
        except Exception as e:
            st.error(f"Translation error: {e}")
            return english_data

    def generate_quiz_for_documents(self, num_questions: int = 5) -> List[Dict]:
        # use top-k chunks from all docs as context
        # simple: take first N chunks
        context = self.vector_store.chunks[:20] if len(self.vector_store.chunks) > 0 else []
        return self.llm.generate_quiz(context, num_questions=num_questions)

    def generate_short_questions(self, num_questions: int = 8) -> List[Dict]:
        context = self.vector_store.chunks[:20] if len(self.vector_store.chunks) > 0 else []
        return self.llm.generate_questions_short(context, num_questions=num_questions)

    def generate_study_schedule(self, days_available: int, hours_per_day: int) -> Dict:
        context = self.vector_store.chunks[:30] if len(self.vector_store.chunks) > 0 else []
        return self.scheduler.create_schedule(context, days_available, hours_per_day)

    def explain_image_content(self, image_text: str, question: str = None, translate_to: str = 'en') -> Dict:
        # Get explanation in English first
        explanation = self.llm.explain_image_text(image_text, question)
        
        # Translate if needed
        if translate_to and translate_to != 'en':
            try:
                explanation['explanation'] = safe_translate(explanation.get('explanation', ''), translate_to, self.translation_service)
                explanation['key_terms'] = [safe_translate(term, translate_to, self.translation_service) for term in explanation.get('key_terms', [])]
                explanation['important_concepts'] = [safe_translate(concept, translate_to, self.translation_service) for concept in explanation.get('important_concepts', [])]
                explanation['examples'] = [safe_translate(example, translate_to, self.translation_service) for example in explanation.get('examples', [])]
            except Exception as e:
                st.error(f"Translation error: {e}")
        
        return explanation

# ------------------------
# Streamlit UI
# ------------------------
def enhanced_study_planner_tab(tab6, qa):
    """Enhanced Study Planner tab with better display"""
    with tab6:
        st.header("📅 Study Planner")
        st.markdown("Create a personalized study schedule based on your uploaded materials.")
        
        if not qa.processed_documents:
            st.info("📤 Upload and process documents first to generate a study plan.")
        else:
            # Input parameters
            st.subheader("Study Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                days_available = st.number_input(
                    "Days available for study", 
                    min_value=1, 
                    max_value=90, 
                    value=7,
                    help="Total number of days you have for studying"
                )
            with col2:
                hours_per_day = st.number_input(
                    "Hours per day", 
                    min_value=1, 
                    max_value=12, 
                    value=3,
                    help="Average hours you can dedicate per day"
                )
            with col3:
                st.metric("Total Study Time", f"{days_available * hours_per_day} hours")
            
            # Generate button with better feedback
            if st.button("🎯 Generate Personalized Study Plan", type="primary"):
                with st.spinner("Creating your personalized study schedule... This may take a moment."):
                    try:
                        # Use the enhanced scheduler
                        enhanced_scheduler = EnhancedStudyScheduler(qa.llm)
                        context = qa.vector_store.chunks[:30] if len(qa.vector_store.chunks) > 0 else []
                        schedule = enhanced_scheduler.create_schedule(context, days_available, hours_per_day)
                        
                        if schedule:
                            # Store in session state
                            st.session_state['study_schedule'] = schedule
                            st.success("✅ Study plan generated successfully!")
                        else:
                            st.error("Failed to generate study schedule. Please try again.")
                    except Exception as e:
                        st.error(f"Error generating schedule: {e}")
            
            # Display the schedule if it exists
            if 'study_schedule' in st.session_state:
                schedule = st.session_state['study_schedule']
                
                # Use enhanced display
                enhanced_scheduler = EnhancedStudyScheduler(qa.llm)
                enhanced_scheduler.display_schedule(schedule)
                
                # Export options
                st.subheader("📥 Export Options")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # JSON export
                    st.download_button(
                        "📄 Download as JSON",
                        data=json.dumps(schedule, indent=2),
                        file_name=f"study_schedule_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json"
                    )
                
                with col2:
                    # Create readable text format
                    text_schedule = create_text_schedule(schedule)
                    st.download_button(
                        "📝 Download as Text",
                        data=text_schedule,
                        file_name=f"study_schedule_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain"
                    )
                
                with col3:
                    # Create CSV format
                    csv_schedule = create_csv_schedule(schedule)
                    st.download_button(
                        "📊 Download as CSV",
                        data=csv_schedule,
                        file_name=f"study_schedule_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )

def career_guidance_tab(tab7, qa):
    """Career Guidance tab with roadmap explanations"""
    with tab7:
        st.header("🧭 Career Path Guidance")
        st.markdown("Explore different tech career paths with detailed roadmaps, learning resources, and YouTube tutorials.")
        
        # Initialize career guidance
        career_guidance = CareerGuidance()
        
        # Display career selector
        career_guidance.display_career_selector()

def main():
    st.set_page_config(page_title="Academic PDF Q&A", page_icon="📚", layout="wide")
    st.title("📚 Enhanced Academic PDF Q&A System with Career Guidance")

    # Sidebar: IBM credentials
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("API Key", type="password")
        project_id = st.text_input("Project ID")
        url = st.text_input("URL", value="https://us-south.ml.cloud.ibm.com")

        if st.button("Initialize / Test"):
            if api_key and project_id and url:
                try:
                    credentials = {'api_key': api_key, 'project_id': project_id, 'url': url}
                    st.session_state['qa_system'] = QASystem(credentials)
                    st.success("✅ Watsonx client initialized (local check).")
                except Exception as e:
                    st.error(f"Init failed: {e}")
            else:
                st.warning("Provide API Key, Project ID and URL")

    if 'qa_system' not in st.session_state:
        if api_key and project_id and url:
            try:
                st.session_state['qa_system'] = QASystem({'api_key': api_key, 'project_id': project_id, 'url': url})
            except Exception:
                st.warning("Please click Initialize / Test in the sidebar after entering credentials.")
                return
        else:
            st.info("Enter Watsonx credentials in the sidebar to start.")
            return

    qa: QASystem = st.session_state['qa_system']

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📤 Upload", "❓ Ask", "🎤 Voice", "🧾 Image/OCR & Audio", "📝 Quiz & QGen", "📅 Study Planner", "🧭 Career Guidance"])

    # ---------- Upload tab ----------
    with tab1:
        st.header("Upload PDF documents")
        uploaded_files = st.file_uploader("Upload PDFs (multiple allowed)", type=['pdf'], accept_multiple_files=True)

        st.markdown("**Optional:** Upload images (jpg/png) whose text you want to include from OCR along with PDFs.")
        uploaded_images = st.file_uploader("Upload images (optional)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True, key="img_upload")

        if uploaded_files:
            if st.button("Process selected documents"):
                progress = st.progress(0)
                img_texts_all = []
                # process images first
                if uploaded_images:
                    for i, up_img in enumerate(uploaded_images):
                        st.text(f"OCR processing image: {up_img.name}")
                        try:
                            img = Image.open(up_img)
                            txt = pytesseract.image_to_string(img)
                            if txt.strip():
                                img_texts_all.append(f"[Image: {up_img.name}]\n{txt}")
                        except Exception as e:
                            st.error(f"OCR failed for {up_img.name}: {e}")
                for i, up in enumerate(uploaded_files):
                    st.text(f"Processing {up.name} ...")
                    success = qa.process_pdf(up, up.name, as_image_texts=img_texts_all)
                    if success:
                        st.success(f"Processed {up.name}")
                    else:
                        st.error(f"Failed {up.name}")
                    progress.progress((i + 1) / len(uploaded_files))
                st.session_state['qa_system'] = qa
                st.success("All uploads processed.")
                qa.vector_store.save_index(Config.CACHE_DIR)

    # ---------- Ask tab ----------
    with tab2:
        st.header("Ask questions")
        
        # Language selection
        col1, col2 = st.columns(2)
        with col1:
            use_general_knowledge = st.checkbox("Use general knowledge (AI answers)", value=False)
        
        question = st.text_area("Enter your question:", height=120)
        
        if st.button("Get Answer"):
            with st.spinner("Searching and generating answer..."):
                # Always get answer in English first
                result = qa.answer_question(
                    question, 
                    translate_to='en',  # Always get English answer first
                    use_general_knowledge=use_general_knowledge
                )
            
            # Store the result in session state for translation
            st.session_state['last_answer'] = result
            st.session_state['show_english_answer'] = True
            
        # Display English answer if available
        if st.session_state.get('show_english_answer') and 'last_answer' in st.session_state:
            result = st.session_state['last_answer']
            
            st.subheader("Answer (English)")
            st.write(result['answer'])
            
            if result['key_points']:
                st.subheader("Key points")
                for kp in result['key_points']:
                    st.write("• " + kp)
            
            if result['limitations']:
                st.subheader("Limitations")
                st.write(result['limitations'])
            
            if result['sources'] and not result['is_general_knowledge']:
                st.subheader("Sources")
                for s in result['sources']:
                    with st.expander(f"{s['document']} - Page {s['page']} (Relevance {s['relevance']:.2f})"):
                        st.write(s['text_snippet'])
            
            if result['is_general_knowledge']:
                st.info("ℹ️ This answer was generated using general knowledge (AI) rather than document content.")
        
        # Translation section - Fixed
        if 'last_answer' in st.session_state:
            st.markdown("---")
            st.subheader("Translate Answer")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                translate_lang = st.selectbox(
                    "Select language for translation",
                    options=list(Config.SUPPORTED_LANGUAGES.keys()),
                    format_func=lambda x: Config.SUPPORTED_LANGUAGES[x],
                    index=0,
                    key="translate_lang_select"
                )
            
            with col2:
                if st.button("Translate", key="translate_btn"):
                    if translate_lang != 'en':
                        with st.spinner(f"Translating to {Config.SUPPORTED_LANGUAGES[translate_lang]}..."):
                            try:
                                translated = qa.translate_answer(
                                    st.session_state['last_answer']['answer_id'],
                                    translate_lang
                                )
                                # Store translated result in session state
                                st.session_state['translated_result'] = translated
                                st.session_state['translated_language'] = translate_lang
                            except Exception as e:
                                st.error(f"Translation failed: {e}")
                    else:
                        st.info("Already in English")
            
            # Display translated answer if available
            if st.session_state.get('translated_result'):
                translated = st.session_state['translated_result']
                lang_name = Config.SUPPORTED_LANGUAGES[st.session_state.get('translated_language', 'en')]
                
                st.markdown("---")
                st.subheader(f"Answer ({lang_name})")
                st.write(translated['answer'])
                
                if translated['key_points']:
                    st.subheader(f"Key points ({lang_name})")
                    for kp in translated['key_points']:
                        st.write("• " + kp)
                
                if translated['limitations']:
                    st.subheader(f"Limitations ({lang_name})")
                    st.write(translated['limitations'])

    # ---------- Voice tab ----------
    with tab3:
        st.header("Voice Questions")
        st.markdown("Ask questions using your microphone")
        
        use_voice_general = st.checkbox("Use general knowledge for voice questions", value=False)
        
        if st.button("🎤 Start Recording", key="record_btn"):
            question = record_audio()
            if question and question not in ["No speech detected", "Could not understand audio"]:
                st.session_state['voice_question'] = question
                st.success(f"Recorded: {question}")
            else:
                st.warning(question)
        
        if 'voice_question' in st.session_state:
            st.text_area("Recorded question:", value=st.session_state['voice_question'], height=100)
            
            if st.button("Get Voice Answer"):
                with st.spinner("Generating answer..."):
                    # Always get answer in English first
                    result = qa.answer_question(
                        st.session_state['voice_question'], 
                        translate_to='en',
                        use_general_knowledge=use_voice_general
                    )
                
                # Store for translation
                st.session_state['last_voice_answer'] = result
                st.session_state['show_voice_answer'] = True
                
            # Display English voice answer if available
            if st.session_state.get('show_voice_answer') and 'last_voice_answer' in st.session_state:
                result = st.session_state['last_voice_answer']
                
                st.subheader("Answer (English)")
                st.write(result['answer'])
                
                if result['key_points']:
                    st.subheader("Key points")
                    for kp in result['key_points']:
                        st.write("• " + kp)
                
                # Translation for voice answers
                st.markdown("---")
                st.subheader("Translate Voice Answer")
                voice_translate_lang = st.selectbox(
                    "Select language for translation",
                    options=list(Config.SUPPORTED_LANGUAGES.keys()),
                    format_func=lambda x: Config.SUPPORTED_LANGUAGES[x],
                    index=0,
                    key="voice_translate_lang"
                )
                
                if st.button("Translate Voice Answer", key="voice_translate_btn"):
                    if 'last_voice_answer' in st.session_state:
                        with st.spinner("Translating..."):
                            translated = qa.translate_answer(
                                st.session_state['last_voice_answer']['answer_id'],
                                voice_translate_lang
                            )
                            
                            if translated:
                                # Store translated result
                                st.session_state['translated_voice_result'] = translated
                                st.session_state['translated_voice_language'] = voice_translate_lang
                
                # Display translated voice answer if available
                if st.session_state.get('translated_voice_result'):
                    translated = st.session_state['translated_voice_result']
                    lang_name = Config.SUPPORTED_LANGUAGES[st.session_state.get('translated_voice_language', 'en')]
                    
                    st.markdown("---")
                    st.subheader(f"Answer ({lang_name})")
                    st.write(translated['answer'])
                    
                    if translated['key_points']:
                        st.subheader(f"Key points ({lang_name})")
                        for kp in translated['key_points']:
                            st.write("• " + kp)

    # ---------- OCR & Audio tab ----------
    with tab4:
        st.header("Image OCR & Audio transcription")
        st.markdown("Use this to extract text from images and transcribe audio to use as a query.")
        
        # Image OCR section
        up_img = st.file_uploader("Upload one image for OCR", type=['png','jpg','jpeg'], key="ocr_single")
        if up_img:
            try:
                img = Image.open(up_img)
                st.image(img, caption="Uploaded image", use_column_width=True)
                ocr_text = pytesseract.image_to_string(img)
                st.subheader("Extracted Text:")
                st.write(ocr_text)
                
                # Add option to ask questions about the image text
                st.markdown("---")
                st.subheader("Ask about this image content")
                
                image_question = st.text_input("What would you like to know about this image content?", key="img_question")
                
                if st.button("Explain Image Content"):
                    with st.spinner("Analyzing image content..."):
                        explanation = qa.explain_image_content(ocr_text, image_question if image_question else None, translate_to='en')
                    
                    st.subheader("Explanation (English)")
                    st.write(explanation.get('explanation', ''))
                    
                    if explanation.get('key_terms'):
                        st.subheader("Key Terms")
                        for term in explanation.get('key_terms', []):
                            st.write(f"• {term}")
                    
                    if explanation.get('important_concepts'):
                        st.subheader("Important Concepts")
                        for concept in explanation.get('important_concepts', []):
                            st.write(f"• {concept}")
                    
                    if explanation.get('examples'):
                        st.subheader("Examples")
                        for example in explanation.get('examples', []):
                            st.write(f"• {example}")
                
                if st.button("Add this OCR text to index as 'image_ocr' doc"):
                    # create a pseudo document from this OCR text and index
                    pseudo_chunks = qa.pdf_processor.chunk_text(qa.pdf_processor.clean_text(f"[Image {up_img.name}]\n{ocr_text}"))
                    qa.vector_store.build_index(pseudo_chunks, f"image_{up_img.name}")
                    qa.processed_documents.add(f"image_{up_img.name}")
                    st.success("OCR text added to index.")
            except Exception as e:
                st.error(f"OCR error: {e}")

        st.markdown("---")
        
        # Audio transcription section
        st.subheader("Audio transcription (upload audio file)")
        st.markdown("Supported formats: WAV, MP3, M4A, FLAC, AAC, OGG, WMA")
        up_audio = st.file_uploader("Upload audio to transcribe", 
                                   type=['wav', 'mp3', 'm4a', 'flac', 'aac', 'ogg', 'wma'], 
                                   key="audio_up")
        if up_audio:
            st.audio(up_audio)
            if st.button("Transcribe Audio"):
                with st.spinner("Transcribing..."):
                    transcript = transcribe_audio_file(up_audio)
                    if transcript and transcript != "Could not understand audio":
                        st.success("Transcription complete")
                        st.write(transcript)
                        st.session_state['last_transcript'] = transcript
                    else:
                        st.warning("No text recognized from audio or audio format not supported")

    # ---------- Quiz & QGen tab ----------
    with tab5:
        st.header("Quiz & Question Generator")
        st.markdown("Generate study questions or quizzes from the indexed materials.")
        
        if not qa.processed_documents:
            st.info("Process documents first.")
        else:
            num_q = st.number_input("Number of questions", min_value=1, max_value=20, value=5)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Generate MCQ Quiz"):
                    with st.spinner("Generating quiz..."):
                        quiz = qa.generate_quiz_for_documents(num_questions=int(num_q))
                    
                    if quiz:
                        st.success("Quiz generated")
                        for i, q in enumerate(quiz, 1):
                            st.markdown(f"**Q{i}. {q.get('question','(no question)')}**")
                            opts = q.get('options', [])
                            for j, opt in enumerate(opts):
                                st.write(f"{chr(97+j)}) {opt}")
                            st.write(f"**Answer:** {q.get('answer','')}")
                            st.write(f"*Explanation:* {q.get('explanation','')}")
                            st.markdown("---")
                    else:
                        st.warning("No quiz returned by model. Try again or check your API credentials.")

            with col2:
                if st.button("Generate Short Answer Questions"):
                    with st.spinner("Generating short-answer questions..."):
                        qlist = qa.generate_short_questions(num_questions=int(num_q))
                    
                    if qlist:
                        st.success("Questions generated")
                        for i, q in enumerate(qlist, 1):
                            st.markdown(f"**Q{i}. {q.get('question','')}**")
                            st.write(f"*Expected Answer:* {q.get('expected_answer','')}")
                            st.markdown("---")
                    else:
                        st.warning("No questions returned by model. Try again or check your API credentials.")

    # ---------- Study Planner tab ----------
    enhanced_study_planner_tab(tab6, qa)
    
    # ---------- Career Guidance tab ----------
    career_guidance_tab(tab7, qa)

    # bottom: show system analytics
    st.sidebar.markdown("---")
    st.sidebar.subheader("System status")
    st.sidebar.write(f"Documents processed: {len(qa.processed_documents)}")
    if qa.vector_store.index:
        st.sidebar.write(f"Total chunks indexed: {int(qa.vector_store.index.ntotal)}")
    st.sidebar.write(f"Embedding model: {Config.EMBEDDING_MODEL}")

if __name__ == "__main__":
    main()