import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv(override=True)

client = Groq(api_key=os.getenv("LLM_API_KEY"))


def generate_story(context: str, genre: str) -> str:
    """
    Generate a story using the Groq LLM API.
    
    Args:
        context: The context/prompt for the story
        genre: The genre of the story (optional)
    
    Returns:
        The generated story text
    """
    genre_prompt = f" in the {genre} genre" if genre else ""
    
    system_prompt = "You are a creative storyteller. Generate engaging and imaginative stories based on the given context."
    user_prompt = f"Write a short story{genre_prompt} based on the following context:\n\n{context}"
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=1024
    )
    
    story = response.choices[0].message.content
    
    return story
