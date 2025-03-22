from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()


def run_pipeline(source: str, language: str = "english") -> dict:
    """Execute end-to-end meeting analysis pipeline on input video/audio."""
    print("Starting AI Video Assistant pipeline...")

    chunks = process_input(source)
    transcript = transcribe_all(chunks, language)
    print(f"\nRaw transcription sample (first 300 chars):\n{transcript[:300]}...\n")

    title = generate_title(transcript)
    summary = summarize(transcript)
    action_items = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  AI Video & Audio Assistant — Shubham Sura")
    print("=" * 60)
    source = input("Enter YouTube URL or local video/audio file path: ").strip()
    language = input("Target language mode (english/hinglish) [default: english]: ").strip() or "english"

    if source:
        result = run_pipeline(source, language)

        print("\n" + "=" * 60)
        print(f"📌 Title: {result['title']}")
        print(f"\n📋 Summary:\n{result['summary']}")
        print(f"\n✅ Action Items:\n{result['action_items']}")
        print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
        print(f"\n❓ Open Questions:\n{result['open_questions']}")
        print("=" * 60)

        # Interactive RAG Session
        print("\n💬 Chat with your meeting transcript (type 'exit' or 'q' to quit)\n")
        rag_chain = result["rag_chain"]
        while True:
            question = input("You: ").strip()
            if question.lower() in ["exit", "quit", "q"]:
                print("👋 Session ended.")
                break
            if not question:
                continue
            answer = ask_question(rag_chain, question)
            print(f"\n🤖 Assistant: {answer}\n")
