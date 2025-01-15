# AI Video & Audio Meeting Assistant 🎬

An intelligent meeting analyzer and conversational AI assistant created by **Shubham Sura**.

## Overview
AI Video Assistant automates video and audio meeting processing. It extracts full transcripts, generates structured summaries, identifies action items, key decisions, and open questions, and allows you to chat with meeting recordings using RAG (Retrieval-Augmented Generation).

## Key Features
- **YouTube & Local File Processing**: Extract audio directly from YouTube URLs or local video/audio files.
- **Multi-Engine Speech-to-Text**: High-accuracy transcription using OpenAI Whisper (English) and Sarvam AI (Hinglish/Translation).
- **LLM-Powered Insights**: Automated Map-Reduce summarization and structured extraction powered by Mistral AI & LangChain.
- **RAG Chat Engine**: Context-grounded Q&A over meeting transcripts using Chroma vector store and HuggingFace embeddings.
- **Interactive UI**: Futuristic Streamlit interface with rich CSS themes and dashboard cards.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/shubhamsura/Web-Development.git
   cd Web-Development
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variables in `.env`:
   ```env
   MISTRAL_API_KEY=your_key_here
   SARVAM_API_KEY=your_key_here
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
