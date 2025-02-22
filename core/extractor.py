import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


def get_llm():
    """Instantiate low-temperature ChatMistralAI model for precise extraction."""
    api_key = os.getenv("MISTRAL_API_KEY")
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=0.1,
    )


def build_extraction_chain(system_prompt: str):
    """Build LCEL extraction chain given a specific system prompt."""
    llm = get_llm()
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:
    """Extract action items, assigned owners, and deadlines from meeting transcript."""
    chain = build_extraction_chain(
        "You are an expert meeting analyst. Extract all actionable tasks from the transcript.\n"
        "For each task provide:\n"
        "- Task description\n"
        "- Owner / Responsible party (if mentioned, else 'Unassigned')\n"
        "- Deadline / Timeline (if mentioned, else 'Not specified')\n\n"
        "Format as a clean numbered list. If none are found, return 'No action items found.'"
    )
    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    """Extract key decisions made during the meeting."""
    chain = build_extraction_chain(
        "You are an expert meeting analyst. Extract all major decisions finalized in the transcript.\n"
        "Format as a clean numbered list. If none are found, return 'No key decisions found.'"
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    """Extract unresolved questions or topics needing follow-up."""
    chain = build_extraction_chain(
        "You are an expert meeting analyst. Extract all open questions or unresolved topics needing follow-up.\n"
        "Format as a clean numbered list. If none are found, return 'No open questions found.'"
    )
    return chain.invoke(transcript)
