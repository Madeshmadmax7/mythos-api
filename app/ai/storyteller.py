from app.utils.llm_client import generate_story


def create_story(context: str, genre: str = "", history: list = None, summary: str = None, retrieved_hints: list = None) -> str:
    """
    Create a story based on context and optional genre.
    """
    story = generate_story(
        context=context, 
        genre=genre or "", 
        history=history,
        summary=summary,
        retrieved_hints=retrieved_hints
    )
    return story
