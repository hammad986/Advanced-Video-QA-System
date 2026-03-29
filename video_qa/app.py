"""
Video-QA RAG System - Streamlit Application
Lecture Video Question Answering with Evidence-Grounded Answers

Fixes applied:
- Auto tab-switch to Ask Questions when video processing finishes
- session_state.active_tab persists across reruns (fixes Ctrl+Enter redirect)
- Get Answer shown via st.form (always visible when KB ready)
- video_id scoping disabled when active_video_id is not set (avoids 0 chunks)
- Graceful Whisper/thread shutdown on Ctrl+C
"""

import os
from pathlib import Path
from config_loader import load_config

# BASE_DIR is the project root
BASE_DIR = Path(__file__).parent.parent

CONFIG = load_config()

def running_on_streamlit_cloud():
    try:
        import streamlit as st
        # System variable set by Streamlit Cloud
        return st.secrets.get("CLOUD", False) or os.getenv("STREAMLIT_CLOUD_DEPLOY") == "true"
    except Exception:
        return False

# Global Cloud Flag
IS_CLOUD = running_on_streamlit_cloud()

def is_cloud():
    try:
        return st.secrets.get("CLOUD", False)
    except Exception:
        return False

# Only load .env if NOT on cloud
if not IS_CLOUD:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

import streamlit as st
import sys

# Add project root to sys.path natively before any package imports
sys.path.insert(0, str(BASE_DIR))

import tempfile
import shutil
import asyncio
import atexit
import threading
import time

from video_qa.query_rewriter import rewrite_query

# ── Windows asyncio fix (prevents "Event loop is closed" on shutdown) ──────
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass


# ── Graceful shutdown ────────────────────────────────────────────────────────
def graceful_shutdown():
    """Join non-daemon threads and free Whisper CTranslate2 workers."""
    try:
        from video_qa.speech_understanding import _whisper_model as _wm  # noqa
        if _wm is not None:
            del _wm
    except Exception:
        pass
    main_thread = threading.main_thread()
    for t in threading.enumerate():
        if t is not main_thread and not t.daemon:
            try:
                t.join(timeout=2.0)
            except Exception:
                pass


atexit.register(graceful_shutdown)

# ─────────────────────────────────────────────
# Streamlit page config  (MUST be first st.* call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Video-QA: Lecture Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* 1. Global Dark Theme */
.stApp {
    background: linear-gradient(135deg, #020617, #0f172a, #1e293b);
    color: #f8fafc;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* 2. Glowing Header Typography */
.glow-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    color: white;
    text-shadow: 0 0 15px rgba(125, 211, 252, 0.6);
    margin-bottom: 0.5rem;
}
.hr-divider {
    border: 0;
    height: 1px;
    background: linear-gradient(to right, transparent, rgba(255,255,255,0.2), transparent);
    margin-bottom: 2rem;
}

/* 3. 3D Interactive Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    height: 3rem !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 12px rgba(0,0,0,0.4) !important;
}
.stButton > button:active {
    transform: translateY(1px) scale(0.98) !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
}

/* 4. Glassmorphic Containers */
.glass-container {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
    margin-bottom: 1.5rem;
}

/* Expander glass styling */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.02) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* 5. Confidence Bar Colors */
.conf-high  { color: #4ade80; font-weight: 700; font-size: 1.5rem; }
.conf-med   { color: #fbbf24; font-weight: 700; font-size: 1.5rem; }
.conf-low   { color: #f87171; font-weight: 700; font-size: 1.5rem; }

/* 6. Most Relevant Evidence Card */
.top-evidence-card {
    background: linear-gradient(135deg, rgba(251,191,36,0.08), rgba(234,179,8,0.04));
    border-left: 4px solid #fbbf24;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0 1.25rem 0;
    box-shadow: 0 2px 12px rgba(251,191,36,0.08);
}
.top-evidence-card h4 {
    color: #fde68a;
    margin: 0 0 0.5rem 0;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.top-evidence-card .ts-badge {
    display: inline-block;
    background: rgba(251,191,36,0.2);
    color: #fde68a;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

/* 7. Answer Card */
.answer-card {
    background: rgba(59,130,246,0.07);
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    font-size: 1.05rem;
    line-height: 1.7;
}

/* 8. Soft <mark> highlight for matched words */
mark {
    background-color: rgba(251,191,36,0.25);
    color: #fde68a;
    border-radius: 3px;
    padding: 0 2px;
    font-weight: 500;
}

/* 9. Chunk card in evidence expander */
.chunk-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}
.chunk-card:hover { border-color: rgba(125,211,252,0.3); }

/* 10. System thinking flow */
.thinking-flow {
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin-top: 0.75rem;
    font-size: 0.85rem;
    color: #a5b4fc;
    line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────
INDEX_PATH = str(BASE_DIR / "models/video_index.faiss")
META_PATH  = str(BASE_DIR / "models/metadata.pkl")

# ─────────────────────────────────────────────
# Session-state initialisation  (must be early, before any widget)
# ─────────────────────────────────────────────
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "process"       # "process" | "ask"

if "active_video_id" not in st.session_state:
    st.session_state.active_video_id = None

if "video_path" not in st.session_state:
    st.session_state.video_path = None

if "video_processed_ok" not in st.session_state:
    st.session_state.video_processed_ok = False

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_question" not in st.session_state:
    st.session_state.last_question = ""\

# only show if it runs on cloud
if IS_CLOUD:
    st.info("☁️ **Cloud Mode Active**: Q&A restricted to pre-processed videos. Video processing and YouTube features are disabled.")


# ─────────────────────────────────────────────
# Cached pipeline
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Initialising pipeline…", ttl=3600)
def get_pipeline():
    from video_qa.pipeline import VideoQAPipeline
    return VideoQAPipeline()


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def knowledge_base_ready() -> bool:
    return os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)


def sec_to_time(seconds: float) -> str:
    total = int(seconds or 0)
    return f"{total // 60:02d}:{total % 60:02d}"


def download_youtube_to_file(url: str, status_placeholder) -> "str | None":
    """
    Download YouTube audio using yt-dlp.
    Returns WAV file path or None if download fails.
    """

    try:
        import yt_dlp
    except ImportError:
        st.error("yt-dlp is not installed. Run: pip install yt-dlp")
        return None

    tmp_dir = Path(tempfile.mkdtemp())
    out_tmpl = str(tmp_dir / "%(id)s.%(ext)s")

    ydl_opts = {
        "outtmpl": out_tmpl,
        "format": "bestaudio/best",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,

        # Helps bypass some YouTube blocks
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        },

        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"]
            }
        },

        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192"
        }],
    }

    status_placeholder.text("⬇️ Downloading YouTube audio...")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # find wav output
        candidates = list(tmp_dir.glob("*.wav"))

        if not candidates:
            candidates = [
                f for f in tmp_dir.glob("*.*")
                if f.suffix.lower() not in [".webp", ".jpg", ".png"]
            ]

        if candidates:
            return str(candidates[0])

        st.error("Download completed but WAV file not found.")
        return None

    except Exception as exc:

        # ⭐ Graceful fallback
        st.error(
            "⚠️ YouTube download failed.\n\n"
            "This usually happens because YouTube blocks cloud servers.\n\n"
            "👉 Please download the video locally and upload the file instead."
        )

        st.info(
            "Recommended workflow:\n"
            "1️⃣ Download video from YouTube\n"
            "2️⃣ Go to **Upload Video File**\n"
            "3️⃣ Upload the MP4 file\n"
            "4️⃣ Process normally"
        )

        return None

# ─────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────
def main():
    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown('<div class="glow-title">🎬 Video-QA Lecture Assistant</div><hr class="hr-divider">', unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align: center; color: #cbd5e1; margin-bottom: 2rem;'>"
        "<b style='font-size:1.05rem;'>Explainable Video AI</b> "
        "— Answers with Evidence, Confidence, and Verification"
        "</div>", unsafe_allow_html=True
    )

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📊 System Status")

        if knowledge_base_ready():
            try:
                import joblib
                metadata = joblib.load(META_PATH)
                st.success("✅ Knowledge Base Ready")
                st.metric("Indexed Chunks", len(metadata))
            except Exception:
                st.success("✅ Knowledge Base Ready")
        else:
            st.warning("⚠️ No videos processed yet")

        st.markdown("---")

        # ── Tab switcher buttons in sidebar ─────────────────────────────────
        st.subheader("Navigation")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📤 Process", use_container_width=True):
                st.session_state.active_tab = "process"
                st.rerun()
        with col_b:
            if st.button("❓ Ask", use_container_width=True):
                st.session_state.active_tab = "ask"
                st.rerun()

        st.markdown("---")
        st.caption(
            "**Pipeline:**\n"
            "1. Video Upload / YouTube URL\n"
            "2. Audio Extraction (ffmpeg)\n"
            "3. Whisper Transcription\n"
            "4. Semantic Chunking\n"
            "5. Embedding + FAISS Index\n"
            "6. Retrieval + Re-ranking\n"
            "7. LLM Answer Generation\n"
            "8. NLI Evidence Verification"
        )

    # ── Tab navigation via session_state ────────────────────────────────────
    tab_labels = ["📤 Process Video", "❓ Ask Questions"]
    selected_tab_idx = 0 if st.session_state.active_tab == "process" else 1
    tab1, tab2 = st.tabs(tab_labels)

    # Programmatic tab selection trick: inject JS to click the correct tab
    # after rerun when active_tab == "ask"
    if st.session_state.active_tab == "ask":
        # Use st.markdown JS injection to automatically click the Ask tab
        st.markdown(
            """
            <script>
            (function() {
                function clickAskTab() {
                    var tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
                    if (tabs && tabs.length >= 2) {
                        tabs[1].click();
                    } else {
                        setTimeout(clickAskTab, 100);
                    }
                }
                setTimeout(clickAskTab, 200);
            })();
            </script>
            """,
            unsafe_allow_html=True,
        )

    # ── Process Video Tab ────────────────────────────────────────────────────
    with tab1:
        # Switch to this tab if active_tab == "process"
        if st.session_state.active_tab == "ask" and knowledge_base_ready():
            # Do not render process tab content when user is on ask tab
            st.info("Switch to the ❓ Ask Questions tab to query your video.")
        else:
            _render_process_tab()

    # ── Ask Questions Tab ────────────────────────────────────────────────────
    with tab2:
        _render_ask_tab()


# ─────────────────────────────────────────────
# Tab renderers
# ─────────────────────────────────────────────
def _render_process_tab():
    """Render the video processing UI."""
    st.subheader("📤 Upload & Process Video")

    # Hide unsupported options on Streamlit Cloud
    if IS_CLOUD:
        input_options = ["Upload Video File"]
    else:
        input_options = ["Upload Video File", "YouTube URL", "Use Existing Videos"]

    input_method = st.radio(
        "Choose input method:",
        input_options,
        horizontal=True,
    )

    video_path: "str | None"  = None
    youtube_url: "str | None" = None

    if input_method == "Upload Video File":
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "mkv", "avi", "mov", "webm", "mp3", "wav", "m4a"],
        )
        if uploaded_file is not None:
            # 🚨 ❗ Cloud Upload Size Limit (100MB)
            if IS_CLOUD and uploaded_file.size > 100 * 1024 * 1024:
                st.error("❌ **File too large.** Max 100MB allowed on community cloud. Please upload a shorter or compressed video.")
                return

            # Check if this file has already been saved to temp
            if st.session_state.get("uploaded_filename") != uploaded_file.name:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    shutil.copyfileobj(uploaded_file, tmp)
                    st.session_state.temp_video_path = tmp.name
                    st.session_state.uploaded_filename = uploaded_file.name
            
            video_path = st.session_state.temp_video_path
            st.success(f"✅ Ready to process: {uploaded_file.name}")

    elif input_method == "YouTube URL":
        youtube_url = st.text_input(
            "Enter YouTube URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste a YouTube link. The audio will be extracted and transcribed automatically.",
        )
        if youtube_url:
            st.info("ℹ️ Click **🚀 Process Video** below to download and process this video automatically.")

    elif input_method == "Use Existing Videos":
        video_dir = BASE_DIR / "data/videos"
        if video_dir.exists():
            videos = sorted(
                list(video_dir.glob("*.mp4"))
                + list(video_dir.glob("*.mkv"))
                + list(video_dir.glob("*.avi"))
            )
            if videos:
                selected = st.selectbox("Select video:", [str(v) for v in videos])
                video_path = selected
            else:
                st.warning("No videos found in data/videos/")
        else:
            st.warning("data/videos/ directory not found")

    ready_to_process = bool(video_path or youtube_url)
    if ready_to_process:
        if st.button("🚀 Process Video", type="primary", use_container_width=True):
            _run_process_video(video_path=video_path, youtube_url=youtube_url)


def _render_ask_tab():
    """Render the Q&A UI — always shows form when KB is ready."""
    st.subheader("❓ Ask Questions About the Video")

    if not knowledge_base_ready():
        st.warning("⚠️ Please process a video first in the '📤 Process Video' tab.")
        return

    # Embed Video Player for Contextual RAG questioning
    _vp = st.session_state.get("video_path")
    if _vp:
        # ── YouTube URL → embed conversion ───────────────────────────────────
        import re as _re
        _yt_match = _re.search(
            r"(?:youtube\.com/watch\?v=|youtu\.be/)([\w\-]{11})", _vp
        )
        if _yt_match:
            _embed_url = f"https://www.youtube.com/embed/{_yt_match.group(1)}"
            st.video(_embed_url)
        elif os.path.exists(_vp):
            st.video(_vp)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # FIX: Apply pending rewrite before widget instantiation
    if "pending_rewrite" in st.session_state:
        st.session_state.current_question_input = st.session_state.pending_rewrite
        del st.session_state.pending_rewrite

    # FIX: Persist question input and result across reruns
    if "current_question_input" not in st.session_state:
        st.session_state.current_question_input = st.session_state.get("last_question", "")

    # FIX: Always show the last result if available, even after rerun
    last_result = st.session_state.get("last_result")
    last_question = st.session_state.get("last_question", "")

    question = st.text_area(
        "Enter your question:",
        height=80,
        placeholder="What is the main topic of this lecture?",
        key="current_question_input",
    )
    
    
    col1, col2 = st.columns(2)
    with col1:
        rewrite = st.button("✨ Rewrite Question", use_container_width=True)
    with col2:
        submit = st.button("🔍 Get Answer", type="primary", use_container_width=True)

    with st.expander("💡 Example Questions"):
        st.markdown(
            """
            <style>
            .example-q {
                background: rgba(255,255,255,0.05);
                padding: 10px 15px;
                border-radius: 8px;
                margin-bottom: 8px;
                border-left: 3px solid #3b82f6;
                transition: all 0.2s ease;
                font-size: 0.95rem;
            }
            .example-q:hover {
                background: rgba(255,255,255,0.1);
                transform: translateX(4px);
            }
            </style>
            <div class="example-q">What is the main topic of the lecture?</div>
            <div class="example-q">Explain the key concept discussed</div>
            <div class="example-q">What example is given for the topic?</div>
            <div class="example-q">Summarize the beginning of the video</div>
            <div class="example-q">Where does the speaker define a term?</div>
            <div class="example-q">What did the lecturer say about [specific subject]?</div>
            """, unsafe_allow_html=True
        )

    # Handle button actions
    if rewrite:
        if not question.strip():
            st.warning("Please enter a question to rewrite.")
        else:
            with st.spinner("✨ Optimizing your question with AI..."):
                rewritten, provider = rewrite_query(question, force=True)
                
            if provider == "ollama":
                st.success("🟢 Using Ollama")
            elif provider == "gemini":
                st.success("🔵 Rewritten using Gemini")
            elif provider == "openai":
                st.success("✅ Rewrite done using OpenAI")
            elif provider == "hf":
                st.success("✅ Rewrite done using HuggingFace")
            elif provider.startswith("ERROR:"):
                st.error(f"❌ Detailed Python Crash: {provider}")
            elif provider == "fallback":
                st.warning("⚠️ Using original question")
            else:
                st.warning(f"⚠️ Using original question ({provider})")

            if rewritten and rewritten != question:
                st.session_state.pending_rewrite = rewritten
                # Allow users to briefly see the status before refreshing to apply the rewritten string
                import time
                time.sleep(1.2)
                st.rerun()
                
    elif submit:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            st.session_state.last_question = question
            _run_answer_question(question)
            
    elif last_result is not None and last_question:
        _display_results(last_question, last_result)


# ─────────────────────────────────────────────
# Pipeline runners
# ─────────────────────────────────────────────
def _run_process_video(video_path: "str | None", youtube_url: "str | None" = None):
    """
    Full processing pipeline.
    Shielded on Streamlit Cloud.
    """
    if is_cloud():
        st.error("⚠️ Video processing is disabled in cloud.\nPlease use preprocessed data.")
        return

    try:
        with st.status("🎬 Processing Video Pipeline...", expanded=True) as status:
            st.warning("⚠️ Please wait. Don't close or refresh this page. Your video is being processed.")
            # Step 0: YouTube download
            if youtube_url:
                status.update(label="⬇️ Downloading YouTube video...")
                video_path = download_youtube_to_file(youtube_url, st.empty())
                if video_path is None:
                    status.update(label="❌ Download failed", state="error")
                    return

            # Steps 1-4: full pipeline
            pipeline = get_pipeline()
            status.update(label="⚙️ Running backend transcription, chunking, and indexing...")
            video_id = pipeline.process_video(video_path)

            if not video_id:
                status.update(label="❌ Processing failed", state="error")
                st.error(
                    "Processing failed. "
                    "Ensure FFmpeg is installed and faster-whisper is available."
                )
                return

            status.update(label="✅ Video processing complete!", state="complete")

        # Clear cached pipeline so next query re-loads the fresh index
        get_pipeline.clear()

        # Store video_id for scoped retrieval
        st.session_state["active_video_id"] = video_id
        st.session_state["video_path"] = video_path

        # ── AUTO-SWITCH to Ask Questions tab ────────────────────────────────
        st.session_state["active_tab"] = "ask"
        st.session_state["video_processed_ok"] = True
        st.session_state["last_result"] = None      # clear previous results
        st.session_state["last_question"] = ""

        st.rerun()  # rerun lands user directly on Ask tab

    except Exception as exc:
        st.error(f"Error during processing: {exc}")


def _run_answer_question(question: str):
    """Retrieve, generate, and display a grounded answer. Never changes active_tab."""
    try:
        with st.status("🧠 Generating Answer from Lecture...", expanded=True) as status:
            pipeline = get_pipeline()

            # Pass active_video_id for scoped retrieval.
            # If None (e.g. knowledge base was pre-built), search all chunks.
            active_id = st.session_state.get("active_video_id")
            result = pipeline.ask(question, active_video_id=active_id)

            status.update(label="✅ Answer generated cleanly!", state="complete")

        # active_tab must remain "ask" after query  — DO NOT change it here
        st.session_state.last_result = result

        _display_results(question, result)

    except Exception as exc:
        st.error(f"Error: {exc}")


# ─────────────────────────────────────────────
# Results display
# ─────────────────────────────────────────────
def _highlight_words(text: str, answer: str) -> str:
    """Wrap answer keywords found in `text` with <mark> tags for soft highlighting."""
    import re as _re
    answer_words = set(w.lower() for w in _re.findall(r"\b[a-zA-Z]{4,}\b", answer))
    if not answer_words:
        return text

    def replacer(m):
        word = m.group(0)
        return f"<mark>{word}</mark>" if word.lower() in answer_words else word

    return _re.sub(r"\b[a-zA-Z]{4,}\b", replacer, text)


def _display_results(question: str, result: dict):
    """Full Explainability Dashboard — confidence, hallucination, evidence."""
    st.markdown("---")
    st.subheader("📊 Analysis Results")

    answer            = result.get("answer", "Question cannot be answered from this video.")
    timestamp         = result.get("timestamp")
    all_contexts      = result.get("all_contexts") or result.get("contexts", [])
    contexts          = result.get("contexts", [])
    verified          = result.get("verified", False)
    verification_info = result.get("verification", {})
    is_summary        = result.get("is_summary", False)

    # ── SUMMARY PATH ──────────────────────────────────────────────────────────
    if is_summary:
        st.info(
            "📋 **Lecture Overview**\n\nThis is a high-level summary generated from the full transcript. "
            "For specific questions, ask factual questions and the RAG pipeline "
            "will retrieve exact transcript evidence with timestamps."
        )
        
        # Display Answer
        clean_answer = answer.replace("**Lecture Overview**", "").replace("**Lecture Overview (Generated)**", "").strip()
        # Force bullet points and newlines to render correctly inside the HTML div
        formatted_summary = clean_answer.replace(" • ", "<br><br>• ").replace(" - ", "<br><br>- ").replace("\n", "<br>")
        st.markdown(f"<div class='answer-card'>{formatted_summary}</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence + Hallucination Panels for Summary
        col_conf, col_hall = st.columns(2)
        
        with col_conf:
            conf = result.get("confidence", 100)
            conf_label = result.get("confidence_label", "High")
            conf_exp = result.get("confidence_explanation", [])
            st.markdown("<h4 style='margin-bottom:0.5rem;'>📊 Confidence Score</h4>", unsafe_allow_html=True)
            color_cls = "conf-high" if conf >= 70 else ("conf-med" if conf >= 40 else "conf-low")
            st.markdown(f"<div class='{color_cls}'>{conf}%</div>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:#94a3b8; font-size:0.85rem;'>Label: **{conf_label}**</span>", unsafe_allow_html=True)
            st.progress(conf / 100)
            
            st.markdown("**Source Context:** Full Transcript")
            for bullet in conf_exp:
                st.markdown(f"- {bullet}")

        with col_hall:
            v_info = result.get("verification", {})
            v_status = v_info.get("status", "UNKNOWN")
            v_just = v_info.get("justification", "")
            
            st.markdown("<h4 style='margin-bottom:0.5rem;'>🛡️ Hallucination Check</h4>", unsafe_allow_html=True)
            if v_status == "VERIFIED":
                st.success("✅ VERIFIED")
            elif v_status == "PARTIALLY_SUPPORTED":
                st.warning("⚠️ PARTIALLY SUPPORTED")
            elif v_status == "HALLUCINATION":
                st.error("❌ HALLUCINATION DETECTED")
            else:
                st.info("❓ UNKNOWN")
                
            if v_just:
                st.markdown(f"**Reason:** {v_just}")
            st.caption(f"Verified via: `{v_info.get('method', 'full_transcript')}` | Trust: **{v_info.get('trust_score', 100)}%**")
            st.progress(min(v_info.get('trust_score', 100), 100) / 100)
            
        return

    # ── STANDARD RAG PATH ─────────────────────────────────────────────────────

    # Confidence data
    confidence       = result.get("confidence", 0)
    conf_label       = result.get("confidence_label", "Low")
    conf_explanation = result.get("confidence_explanation", [])
    conf_breakdown   = result.get("confidence_breakdown", {})
    useful_chunks    = result.get("useful_chunks", 0)
    total_chunks     = result.get("total_chunks", len(all_contexts))

    # Verification data
    v_status = verification_info.get("status", "UNKNOWN")      if verification_info else "UNKNOWN"
    v_trust  = verification_info.get("trust_score", 0)         if verification_info else 0
    v_just   = verification_info.get("justification", "")      if verification_info else ""
    v_method = verification_info.get("method", "fallback")     if verification_info else "fallback"

    # ① AI ANSWER ─────────────────────────────────────────────────────────────
    provider = result.get("provider", "unknown")
    provider_badge = {
        "ollama": "🟢 Local AI (Ollama)",
        "gemini": "🔵 Gemini",
        "openai": "🟣 OpenAI",
        "hf":     "🟡 HuggingFace",
    }.get(provider, f"⚙️ {provider}")

    st.markdown("<h3 style='margin-bottom:0.25rem;'>🤖 AI Answer</h3>", unsafe_allow_html=True)
    st.caption(f"Generated by: {provider_badge}")
    if timestamp:
        st.caption(f"📍 Primary Evidence Timestamp: **{timestamp}**")

    if not all_contexts:
        st.warning("❌ No relevant information found in the video.")
        return

    if not answer or len(answer.strip()) < 5 or "not found" in answer.lower():
        answer = all_contexts[0].get("text", "")[:200]

    # Force bullet points and newlines to render correctly inside the HTML div
    formatted_answer = answer.replace(" • ", "<br><br>• ").replace(" - ", "<br><br>- ").replace("\n", "<br>")
    st.markdown(f"<div class='answer-card'>{formatted_answer}</div>", unsafe_allow_html=True)

    # ② LOW-CONFIDENCE WARNING ────────────────────────────────────────────────
    if confidence < 40:
        st.warning(
            "⚠️ **Low Confidence Answer** — The system could not find strong evidence for this answer. "
            "Try rephrasing your question or check another part of the video."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ③ CONFIDENCE + HALLUCINATION PANELS ─────────────────────────────────────
    col_conf, col_hall = st.columns(2)

    with col_conf:
        st.markdown("<h4 style='margin-bottom:0.5rem;'>📊 Confidence Score</h4>", unsafe_allow_html=True)
        color_cls = "conf-high" if confidence >= 70 else ("conf-med" if confidence >= 40 else "conf-low")
        st.markdown(f"<div class='{color_cls}'>{confidence}%</div>", unsafe_allow_html=True)
        st.markdown(f"<span style='color:#94a3b8; font-size:0.85rem;'>Label: **{conf_label}**</span>", unsafe_allow_html=True)
        st.progress(confidence / 100)

        if conf_explanation:
            st.markdown("**Why this score?**")
            for bullet in conf_explanation:
                st.markdown(f"- {bullet}")

        with st.expander("📐 Score Breakdown", expanded=False):
            bd = conf_breakdown
            st.markdown(
                f"| Factor | Value | Weight |\n"
                f"|--------|-------|--------|\n"
                f"| Retrieval Similarity | `{bd.get('avg_similarity', 0):.3f}` | 50% |\n"
                f"| Keyword Overlap | `{bd.get('context_overlap', 0):.3f}` | 30% |\n"
                f"| Chunk Agreement | `{bd.get('chunk_agreement', 0):.3f}` | 20% |"
            )

    with col_hall:
        st.markdown("<h4 style='margin-bottom:0.5rem;'>🛡️ Hallucination Check</h4>", unsafe_allow_html=True)

        if v_status == "VERIFIED":
            st.success("✅ VERIFIED")
        elif v_status == "PARTIALLY_SUPPORTED":
            st.warning("⚠️ PARTIALLY SUPPORTED")
        elif v_status == "HALLUCINATION":
            st.error("❌ HALLUCINATION DETECTED")
        else:
            st.info("❓ UNKNOWN")

        if v_just:
            st.markdown(f"**Reason:** {v_just}")

        st.caption(f"Verified via: `{v_method}` | Trust: **{v_trust}%**")
        st.progress(min(v_trust, 100) / 100)

    st.markdown("<br>", unsafe_allow_html=True)

    # ④ SOURCES USED INDICATOR ────────────────────────────────────────────────
    THRESHOLD = 0.50
    if all_contexts:
        useful = [c for c in all_contexts if float(c.get("score", 0)) >= THRESHOLD]
        st.caption(
            f"🔎 **Sources Used:** {len(useful)} / {len(all_contexts)} chunks "
            f"above similarity threshold ({THRESHOLD})"
        )

    # ⑤ MOST RELEVANT EVIDENCE CARD ───────────────────────────────────────────
    if all_contexts:
        top_ctx   = max(all_contexts, key=lambda c: float(c.get("score", 0)))
        top_start = sec_to_time(top_ctx.get("start", 0))
        top_end   = sec_to_time(top_ctx.get("end",   0))
        top_score = float(top_ctx.get("score", 0))
        top_text  = top_ctx.get("text", "")
        top_hl    = _highlight_words(top_text[:400], answer)

        st.markdown(
            f"<div class='top-evidence-card'>"
            f"<h4>⭐ Most Relevant Evidence</h4>"
            f"<div class='ts-badge'>[{top_start} – {top_end}]</div>"
            f"&nbsp;&nbsp;<code style='font-size:0.78rem;'>similarity: {top_score:.3f}</code><br><br>"
            f"<span style='color:#e2e8f0; line-height:1.7;'>\"{top_hl}{'…' if len(top_text) > 400 else ''}\"</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ⑥ SYSTEM THINKING FLOW ──────────────────────────────────────────────────
    n_retrieved = len(all_contexts) or 5
    n_llm       = min(2, len(contexts) if contexts else 2)
    st.markdown(
        f"<div class='thinking-flow'>"
        f"<b style='color:#c7d2fe; font-size:0.88rem;'>🧠 How the System Answered</b><br>"
        f"1. Retrieved <b>{n_retrieved}</b> relevant video segments using FAISS vector search<br>"
        f"2. Selected top <b>{n_llm}</b> most relevant chunks for LLM context<br>"
        f"3. Generated answer using <b>{provider_badge}</b> grounded in transcript<br>"
        f"4. Verified answer against retrieved evidence via <b>{v_method}</b><br>"
        f"5. Computed confidence score: <b>{confidence}%</b> ({conf_label})"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ⑦ ALL SOURCE EVIDENCE (expandable) ─────────────────────────────────────
    with st.expander("📚 All Source Evidence", expanded=True):
        if all_contexts:
            for i, ctx in enumerate(all_contexts[:5], 1):
                start   = sec_to_time(ctx.get("start", 0))
                end     = sec_to_time(ctx.get("end",   0))
                text    = ctx.get("text", "")
                score   = ctx.get("rerank_score", ctx.get("score", 0))
                hl_text = _highlight_words(text[:500], answer)
                accent  = "#4ade80" if score >= 0.75 else ("#fbbf24" if score >= 0.5 else "#94a3b8")

                st.markdown(
                    f"<div class='chunk-card' style='border-left: 3px solid {accent};'>"
                    f"<b>Evidence {i}</b> &nbsp;"
                    f"<span style='color:#94a3b8;'>[{start} – {end}]</span> &nbsp;"
                    f"<code style='font-size:0.78rem;'>score: {score:.3f}</code><br><br>"
                    f"<div style='color:#cbd5e1; line-height:1.7;'>{hl_text}{'…' if len(text) > 500 else ''}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No evidence passages retrieved.")


if __name__ == "__main__":
    main()

