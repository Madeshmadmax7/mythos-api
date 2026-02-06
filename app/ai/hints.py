import os
import logging
from typing import List
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

client = Groq(api_key=os.getenv("LLM_API_KEY"))


def extract_short_hint(story_text: str) -> str:
    """
    Extract a single 5-10 word context hint from a story segment.
    Used for RAG-like context continuity.
    
    Args:
        story_text: The story text to extract hint from
    
    Returns:
        A short 5-10 word hint string
    """
    system_prompt = """You extract ultra-short story context hints. Output ONLY 5-10 words that capture the key context."""

    user_prompt = f"""Extract a 5-10 word hint capturing the key context from this story segment.
Output ONLY the hint, nothing else. No bullet points, no explanations.

Story:
{story_text[-2000:]}"""  # Use last 2000 chars for context

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
        # Clean and limit to ~10 words
        words = hint.split()[:10]
        return ' '.join(words)
    
    except Exception as e:
        logger.error(f"Error extracting hint: {e}")
        return ""


def generate_story_with_context(
    user_prompt: str, 
    genre: str = "",
    previous_hints: List[str] = None,
    previous_content_summary: str = None
) -> tuple[str, str]:
    """
    Generate a story segment using hints as RAG/few-shot context.
    Returns both the story and a new hint for this segment.
    
    Args:
        user_prompt: The user's story prompt
        genre: Optional genre
        previous_hints: List of previous context hints (5-10 words each)
        previous_content_summary: Brief summary of what happened before
    
    Returns:
        Tuple of (story_text, new_hint)
    """
    # Build context from hints (RAG-like)
    hints_context = ""
    if previous_hints:
        hints_context = "\n\nPrevious story context (DO NOT repeat, continue from here):\n"
        for i, hint in enumerate(previous_hints, 1):
            hints_context += f"- {hint}\n"
    
    summary_context = ""
    if previous_content_summary:
        summary_context = f"\n\nStory so far (brief): {previous_content_summary}\n"
    
    genre_str = f" in the {genre} genre" if genre else ""
    
    system_prompt = f"""You are a creative storyteller{genre_str}.
Write engaging, original story content.
CRITICAL RULES:
1. NEVER repeat content from previous segments
2. Continue naturally from where the story left off
3. Advance the plot with new events
4. Keep characters and settings consistent"""

    user_content = f"""Write the next part of the story based on this prompt: {user_prompt}
{hints_context}
{summary_context}
Write a fresh, engaging continuation. DO NOT repeat any previous content."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.8,
            max_tokens=1200
        )
        
        story_text = response.choices[0].message.content
        
        # Extract hint for this new segment
        new_hint = extract_short_hint(story_text)
        
        return story_text, new_hint
    
    except Exception as e:
        logger.error(f"Error generating story: {e}")
        raise Exception(f"Failed to generate story: {str(e)}")


def refine_single_segment(
    original_text: str,
    refine_prompt: str,
    previous_hints: List[str] = None
) -> tuple[str, str]:
    """
    Refine ONLY a single story segment without affecting others.
    
    Args:
        original_text: The original text to refine
        refine_prompt: Instructions for refinement
        previous_hints: Context hints to maintain continuity
    
    Returns:
        Tuple of (refined_text, new_hint)
    """
    hints_context = ""
    if previous_hints:
        hints_context = "\n\nMaintain consistency with this context:\n"
        for hint in previous_hints:
            hints_context += f"- {hint}\n"
    
    system_prompt = """You are a story editor. Refine the given text based on instructions.
Keep the same general story beats but improve based on the user's request.
Maintain consistency with any provided context."""

    user_content = f"""Original text:
{original_text}

Refinement instructions: {refine_prompt}
{hints_context}
Write the refined version:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.7,
            max_tokens=1200
        )
        
        refined_text = response.choices[0].message.content
        new_hint = extract_short_hint(refined_text)
        
        return refined_text, new_hint
    
    except Exception as e:
        logger.error(f"Error refining segment: {e}")
        raise Exception(f"Failed to refine: {str(e)}")


def generate_continuation(
    user_prompt: str,
    genre: str = "",
    all_previous_hints: List[str] = None
) -> tuple[str, str]:
    """
    Generate the next part of the story using accumulated hints as context.
    This is the main continuation function that prevents repetition.
    
    Args:
        user_prompt: What the user wants to happen next
        genre: Story genre
        all_previous_hints: All hints from previous segments (5-10 words each)
    
    Returns:
        Tuple of (story_text, new_hint)
    """
    # Build rich context from all previous hints
    context_block = ""
    if all_previous_hints and len(all_previous_hints) > 0:
        context_block = "\n\n=== STORY CONTEXT (What happened before - DO NOT repeat) ===\n"
        for i, hint in enumerate(all_previous_hints, 1):
            context_block += f"{i}. {hint}\n"
        context_block += "=== END CONTEXT ===\n"
    
    genre_str = f" ({genre})" if genre else ""
    
    system_prompt = f"""You are a creative storyteller writing an ongoing narrative{genre_str}.

CRITICAL RULES:
1. The context shows what ALREADY happened - NEVER repeat these events
2. Start your continuation AFTER where the context ends
3. Create NEW plot developments, scenes, and dialogue
4. Keep characters and world consistent but advance the story
5. Write engaging, vivid prose"""

    user_content = f"""{context_block}

USER REQUEST: {user_prompt}

Write the NEXT part of the story. This must be NEW content that continues from where the story left off.
Do NOT summarize or repeat previous events. Jump right into new action/scenes."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.85,
            max_tokens=1400
        )
        
        story_text = response.choices[0].message.content
        new_hint = extract_short_hint(story_text)
        
        return story_text, new_hint
    
    except Exception as e:
        logger.error(f"Error generating continuation: {e}")
        raise Exception(f"Failed to generate continuation: {str(e)}")
