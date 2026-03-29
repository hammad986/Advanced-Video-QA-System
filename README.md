# Video-QA RAG: Evidence-Based Video Question Answering

## Short Description
A research-grade, strict evidence-based Video Question Answering (Video-QA) system that leverages Retrieval Augmented Generation (RAG) to answer questions about lecture videos using only transcript evidence. Built for transparency, reliability, and extensibility.

---

## Problem Statement
Modern LLMs often hallucinate or provide unverifiable answers, especially for video content. This project addresses the challenge of building a Video-QA system that answers user questions strictly from the video transcript, with no hallucination and full evidence traceability.

---

## Features
- Upload or link lecture videos (YouTube supported)
- Automatic audio extraction and speech-to-text transcription
- Semantic chunking and vector-based retrieval (FAISS)
- Evidence-based answer generation (LLM)
- NLI-based answer verification (anti-hallucination)
- Timestamped evidence display
- Streamlit web UI for interactive use
- Configurable pipeline and model selection

---

## System Architecture
![System Architecture](docs/architecture_diagram.png)

- **Video Input** → **Audio Extraction** → **Speech Recognition (WhisperX)** → **Transcript Chunking** → **Embedding Generation (BGE/E5)** → **FAISS Vector Index** → **Similarity Retrieval** → **LLM Answer Generation** → **NLI Verification** → **UI Display**

---

## Technology Stack
- Python 3.10+
- Streamlit (UI)
- FAISS (vector search)
- HuggingFace Transformers (LLM, embeddings, NLI)
- WhisperX (speech-to-text)
- ffmpeg, yt-dlp (video/audio processing)

---

## How It Works: RAG Pipeline
1. **Video Upload/Link**: User provides a video file or YouTube URL.
2. **Audio Extraction**: Audio is extracted using ffmpeg/yt-dlp.
3. **Transcription**: WhisperX transcribes audio to text with timestamps.
4. **Chunking**: Transcript is segmented into semantically meaningful chunks.
5. **Embedding**: Chunks are embedded using BGE/E5 models.
6. **Indexing**: Embeddings are indexed in FAISS for fast retrieval.
7. **Question Answering**: User question is embedded and used to retrieve top-k relevant chunks.
8. **LLM Answer Generation**: LLM generates an answer using only the retrieved evidence.
9. **NLI Verification**: NLI model checks if the answer is strictly supported by the evidence.
10. **UI Display**: Answer, evidence, and timestamps are shown in the web UI.

---

## Installation Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/video-qa-rag.git
   cd video-qa-rag
   ```
2. **Set up Python environment:**
   ```bash
   python -m venv my_project
   source my_project/Scripts/activate  # On Windows: my_project\Scripts\activate.bat
   pip install -r requirements.txt
   ```
3. **Configure API keys and settings:**
   - Edit `config.yaml` to add your HuggingFace API key and adjust model settings as needed.

---

## Usage Instructions
1. **Start the Streamlit app:**
   ```bash
   streamlit run video_qa/app.py
   ```
2. **Open your browser:**
   - Go to [http://localhost:8501](http://localhost:8501)
3. **Upload a video or enter a YouTube URL.**
4. **Ask questions about the video.**
5. **View answers with supporting evidence and timestamps.**

---

## Example Questions
- "What is the main topic discussed at 10 minutes?"
- "List the key points from the introduction."
- "Who is the speaker?"
- "What is the definition of retrieval augmented generation?"
- "Summarize the section between 15:00 and 20:00."

---

## Project Structure
```
video-qa-rag/
├── video_qa/
│   ├── app.py                # Streamlit UI
│   ├── video_processor.py    # Video/audio handling
│   ├── speech_understanding.py # WhisperX transcription
│   ├── knowledge_structuring.py # Chunking logic
│   ├── embeddings.py         # Embedding & FAISS
│   ├── retrieval.py          # Retrieval logic
│   ├── reranker.py           # (Optional) Cross-encoder reranking
│   ├── answer_generator.py   # LLM answer generation
│   ├── evidence_verifier.py  # NLI verification
│   ├── pipeline.py           # Pipeline orchestration
│   └── ...
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── ...
```

---

## Future Improvements
- Support for multi-lingual videos
- Advanced reranking and hybrid retrieval
- Integration with more LLM providers
- Enhanced UI/UX and analytics
- Docker deployment and cloud support
- Automated evaluation and benchmarking

---

> **This project implements a strict, evidence-based Video Question Answering system using Retrieval Augmented Generation. Answers are always grounded in transcript evidence—no hallucinations, just facts.**
