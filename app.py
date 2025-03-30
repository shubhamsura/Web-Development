import streamlit as st
import time
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(
    page_title="AI Video Assistant | Shubham Sura",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎬 AI Video & Meeting Assistant")
st.subheader("Transform long videos and meetings into structured insights & interactive Q&A")

st.sidebar.header("Configuration")
source_type = st.sidebar.radio("Source Type", ["YouTube URL", "Local Video/Audio File"])
language = st.sidebar.selectbox("Language Mode", ["english", "hinglish"], help="Select English or Hinglish (Sarvam AI API)")

source_input = None
if source_type == "YouTube URL":
    source_input = st.sidebar.text_input("Enter YouTube Video URL")
else:
    uploaded_file = st.sidebar.file_uploader("Upload Audio/Video", type=["mp4", "mp3", "wav", "m4a", "webm"])
    if uploaded_file:
        source_input = f"downloads/{uploaded_file.name}"
        with open(source_input, "wb") as f:
            f.write(uploaded_file.getbuffer())

if st.sidebar.button("Process Meeting", type="primary") and source_input:
    with st.spinner("Processing video and extracting transcript..."):
        chunks = process_input(source_input)
        transcript = transcribe_all(chunks, language)
        title = generate_title(transcript)
        summary = summarize(transcript)
        action_items = extract_action_items(transcript)
        decisions = extract_key_decisions(transcript)
        questions = extract_questions(transcript)
        rag_chain = build_rag_chain(transcript)

        st.session_state["pipeline_result"] = {
            "title": title,
            "transcript": transcript,
            "summary": summary,
            "action_items": action_items,
            "key_decisions": decisions,
            "open_questions": questions,
            "rag_chain": rag_chain,
        }

if "pipeline_result" in st.session_state:
    res = st.session_state["pipeline_result"]
    st.header(res["title"])

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Summary",
        "✅ Action Items",
        "🔑 Key Decisions",
        "❓ Open Questions",
        "📜 Transcript",
        "💬 Chat Assistant",
    ])

    with tab1:
        st.markdown(res["summary"])

    with tab2:
        st.markdown(res["action_items"])

    with tab3:
        st.markdown(res["key_decisions"])

    with tab4:
        st.markdown(res["open_questions"])

    with tab5:
        st.text_area("Full Transcript", res["transcript"], height=400)

    with tab6:
        st.subheader("Chat with Meeting RAG Agent")
        q = st.text_input("Ask any question about the meeting context:")
        if st.button("Submit Question") and q:
            ans = ask_question(res["rag_chain"], q)
            st.write(f"**Answer:** {ans}")
