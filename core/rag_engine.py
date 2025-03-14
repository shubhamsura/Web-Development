import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever


def get_llm():
    """Instantiate ChatMistralAI model for RAG responses."""
    api_key = os.getenv("MISTRAL_API_KEY")
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=0.3,
    )


def format_docs(docs):
    """Concatenate page contents of retrieved documents."""
    return "\n\n".join([doc.page_content for doc in docs])


def build_rag_chain(transcript: str):
    """Construct LCEL RAG chain for transcript querying."""
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting assistant. Answer the user's question "
                "based ONLY on the meeting transcript context provided below.\n\n"
                "If the answer is not found in the context, respond with:\n"
                "\"I could not find this information in the meeting transcript.\"\n\n"
                "Be concise, precise, and objective. Quoted statements should be accurate.\n\n"
                "Context from meeting transcript:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def load_rag_chain():
    """Load existing RAG chain from stored Chroma vector DB."""
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting assistant. Answer the user's question "
                "based ONLY on the meeting transcript context provided below.\n\n"
                "If the answer is not found in the context, respond with:\n"
                "\"I could not find this information in the meeting transcript.\"\n\n"
                "Context from meeting transcript:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def ask_question(rag_chain, question: str) -> str:
    """Execute question against RAG chain and return answer string."""
    print(f"Question: {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")
    return answer
