import os
import logging
from typing import List
from groq import Groq
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# Initialize Groq client directly (like the working example)
client = Groq(api_key=os.getenv("LLM_API_KEY"))


from app.utils.llm_client import generate_story, client
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)


def extract_short_hint(story_text: str) -> str:
    """
    Extract a single 5-10 word context hint from a story segment.
    Used for database metadata.
    """
    system_prompt = "You extract ultra-short story context hints. Output ONLY 5-10 words that capture the key context."
    user_prompt = f"Extract a 5-10 word hint capturing the key context from this story segment:\n\n{story_text[-2000:]}"

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=50
        )
        hint = response.choices[0].message.content.strip()
        return ' '.join(hint.split()[:10])
    except Exception as e:
        logger.error(f"Error extracting hint: {e}")
        return ""


def retrieve_relevant_hints(history_hints: List[str], summary: str = None, top_k: int = 5) -> List[str]:
    """
    Retrieve the most relevant hints:
    1. Last K hints (recency).
    2. Hints containing keywords from the story summary (characters/locations).
    """
    if not history_hints:
        return []
        
    final_hints = []
    
    # 1. Start with the most recent (recency)
    recent_hints = history_hints[-top_k:]
    for h in recent_hints:
        if h not in final_hints:
            final_hints.append(h)
    
    # 2. Key-word prioritization from summary (basic RAG-style relevance)
    if summary:
        # Simple keyword extract: words > 4 chars (likely names/locations/nouns)
        summary_words = set([w.strip(".,!?;:()").lower() for w in summary.split() if len(w) > 4])
        
        # Look for these keywords in older hints
        older_hints = history_hints[:-top_k]
        for hint in older_hints:
            hint_lower = hint.lower()
            if any(word in hint_lower for word in summary_words):
                if hint not in final_hints:
                    final_hints.append(hint)
                    
    # Cap total hints to maintain token efficiency (max 10)
    return final_hints[:10]


def generate_story_with_context(
    user_prompt: str, 
    genre: str = "",
    history: List[dict] = None,
    summary: str = None,
    previous_hints: List[str] = None,
    previous_nsi: int = 100,
    world_rules: str = None
) -> tuple[str, str, dict, str]:
    """
    Generate a story segment using hybrid memory: summary + hints + history window.
    Returns (story_text, hint, violations, updated_rules).
    """
    retrieved_hints = retrieve_relevant_hints(previous_hints, summary=summary)
    
    try:
        story_text, violations, updated_rules = generate_story(
            context=user_prompt,
            genre=genre,
            history=history,
            summary=summary,
            retrieved_hints=retrieved_hints,
            previous_nsi=previous_nsi,
            world_rules=world_rules,
            temperature=0.8,
            max_tokens=1200
        )
        new_hint = extract_short_hint(story_text)
        return story_text, new_hint, violations, updated_rules
    except Exception as e:
        logger.error(f"Error generating story: {e}")
        raise Exception(f"Failed to generate story: {str(e)}")


def refine_single_segment(
    original_text: str,
    refine_prompt: str,
    history: List[dict] = None,
    summary: str = None,
    previous_hints: List[str] = None,
    previous_nsi: int = 100,
    world_rules: str = None
) -> tuple[str, str, dict, str]:
    """
    Refine a single story segment with hybrid memory context.
    Returns (refined_text, hint, violations, updated_rules).
    """
    refine_instruction = f"Refine this specific segment: '{original_text}'. Instructions: {refine_prompt}"
    retrieved_hints = retrieve_relevant_hints(previous_hints, summary=summary)
    
    try:
        refined_text, violations, updated_rules = generate_story(
            context=refine_instruction,
            history=history,
            summary=summary,
            retrieved_hints=retrieved_hints,
            previous_nsi=previous_nsi,
            world_rules=world_rules,
            temperature=0.6,
            max_tokens=1200
        )
        new_hint = extract_short_hint(refined_text)
        return refined_text, new_hint, violations, updated_rules
    except Exception as e:
        logger.error(f"Error refining segment: {e}")
        raise Exception(f"Failed to refine: {str(e)}")


def generate_continuation(
    user_prompt: str,
    genre: str = "",
    history: List[dict] = None,
    summary: str = None,
    all_previous_hints: List[str] = None,
    previous_nsi: int = 100,
    world_rules: str = None
) -> tuple[str, str, dict, str]:
    """
    Generate the next part of the story using hybrid memory: summary + memory hints + recent history.
    Returns (story_text, hint, violations, updated_rules).
    """
    retrieved_hints = retrieve_relevant_hints(all_previous_hints, summary=summary)
    
    try:
        story_text, violations, updated_rules = generate_story(
            context=user_prompt,
            genre=genre,
            history=history,
            summary=summary,
            retrieved_hints=retrieved_hints,
            previous_nsi=previous_nsi,
            world_rules=world_rules,
            temperature=0.85,
            max_tokens=1400
        )
        new_hint = extract_short_hint(story_text)
        return story_text, new_hint, violations, updated_rules
    except Exception as e:
        logger.error(f"Error generating continuation: {e}")
        raise Exception(f"Failed to generate continuation: {str(e)}")
