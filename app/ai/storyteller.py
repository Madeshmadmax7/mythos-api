from app.utils.llm_client import generate_story


def create_story(context: str, genre: str = None) -> str:
    """
    Create a story based on context and optional genre.
    
    Args:
        context: The context/prompt for the story
        genre: Optional genre for the story
    
    Returns:
        The generated story text
    """
    genre_str = genre if genre else ""
    story = generate_story(context=context, genre=genre_str)
    return story
