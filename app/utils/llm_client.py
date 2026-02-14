import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv(override=True)

client = Groq(api_key=os.getenv("LLM_API_KEY"))


def generate_story(
    context: str, 
    genre: str = "", 
    history: list = None, 
    summary: str = None,
    retrieved_hints: list = None,
    previous_nsi: int = 100,
    world_rules: str = None,
    temperature: float = 0.85, 
    max_tokens: int = 1200
) -> str:
    """
    Generate a story continuation using genre-adaptive world consistency engine.
    Returns (clean_text, violations, updated_rules).
    """

    genre_str = f" in the {genre} genre" if genre else ""
    active_genre = (genre or "general").upper().replace(" ", "_")

    # Build world rules context (dedicated column > summary fallback)
    rules_context = world_rules or summary or "No established world rules yet."
    # Build hint RAG context
    hint_rag = "\n".join([f"- {h}" for h in (retrieved_hints or [])]) or "No previous hints."

    # 🔥 Genre-Adaptive World Consistency Engine
    system_prompt = (
        "[ANTIGRAVITY_EXECUTION_BLOCK]\n\n"
        "OBJECTIVE:\n"
        "Maintain persistent world consistency across turns according to ACTIVE_GENRE.\n"
        "Preserve narrative stability.\n"
        "Prevent contradictions.\n\n"

        "--------------------------------\n"
        f"ACTIVE_GENRE: {active_genre}\n"
        f"EXISTING_WORLD_RULES: {rules_context}\n"
        f"PREVIOUS_WORLD_HINTS:\n{hint_rag}\n"
        f"PREVIOUS_NSI_SCORE: {previous_nsi}\n"
        "--------------------------------\n\n"

        "INTERNAL EXECUTION (DO NOT OUTPUT):\n\n"
        "1. Extract constraints from:\n"
        "   - EXISTING_WORLD_RULES\n"
        "   - PREVIOUS_WORLD_HINTS\n"
        "   - Current user input\n\n"

        "2. Merge into WORLD_RULE_SET.\n"
        "   - Never remove prior constraints unless explicitly changed in-story.\n"
        "   - Preserve environmental, physical, magical, technological, biological rules.\n\n"

        "3. Apply GENRE_ADAPTIVE_ENFORCEMENT:\n\n"

        "   HARD_SCI_FI:\n"
        "     - Real-world physics strictly enforced.\n"
        "     - No survival in vacuum without protection.\n"
        "     - Combustion requires oxygen.\n"
        "     - Radiation, gravity, pressure obey science.\n\n"

        "   SOFT_SCI_FI:\n"
        "     - Plausible science.\n"
        "     - Speculative tech must remain internally consistent.\n\n"

        "   FANTASY:\n"
        "     - Real-world physics irrelevant unless previously established.\n"
        "     - Magical systems must remain consistent.\n"
        "     - Power limitations persist.\n\n"

        "   HORROR:\n"
        "     - Maintain environmental continuity.\n"
        "     - Supernatural elements must follow introduced logic.\n\n"

        "   REALISM:\n"
        "     - Real-world logic strictly enforced.\n\n"

        "4. If new input contradicts WORLD_RULE_SET:\n"
        "     - Adjust narrative logically.\n"
        "     - Prevent impossible survival or contradictions.\n"
        "     - Preserve immersion.\n\n"

        "5. Update WORLD_RULE_SET with validated new constraints.\n\n"

        "ADAPTIVE STABILIZATION:\n"
        f"If PREVIOUS_NSI_SCORE ({previous_nsi}) < 80:\n"
        "  - Prioritize continuity stabilization.\n"
        "  - Avoid introducing new plot branches.\n"
        "  - Reinforce established constraints.\n\n"

        "--------------------------------\n"
        "SCORING (INTERNAL CALCULATION):\n"
        "--------------------------------\n\n"

        "NSI VIOLATION DETECTION (STATIC — DO NOT MODIFY RULES):\n\n"
        "Detect violations using ONLY the following categories:\n"
        "Detection must be conservative and evidence-based.\n"
        "Do not assume violations unless clearly present.\n\n"
        "1. CHARACTER_INCONSISTENCY\n"
        "   - Personality shifts without cause\n"
        "   - Skill changes without explanation\n"
        "   - Motivation contradictions\n\n"
        "2. TIMELINE_CONTRADICTION\n"
        "   - Events occurring in impossible order\n"
        "   - Time skips without acknowledgment\n"
        "   - Logical sequence breaks\n\n"
        "3. WORLD_RULE_VIOLATION\n"
        "   - Breaking established environmental, physical, magical, or technological constraints\n\n"
        "4. IGNORED_FACT\n"
        "   - Previously established facts not respected\n\n"

        "--------------------------------\n"
        "OUTPUT RULES:\n"
        "--------------------------------\n"
        "- Output immersive plain text story only.\n"
        "- Append hidden metadata block at the very end:\n\n"
        "<WRLD>\n"
        "UPDATED_RULES: ...\n"
        "VIOLATION_COUNTS:\n"
        "  CHARACTER_INCONSISTENCY: <int>\n"
        "  TIMELINE_CONTRADICTION: <int>\n"
        "  WORLD_RULE_VIOLATION: <int>\n"
        "  IGNORED_FACT: <int>\n"
        "</WRLD>\n\n"

        "Rules for VIOLATION_COUNTS:\n"
        "- Output only integer counts.\n"
        "- If no violations, output 0.\n"
        "- Do NOT calculate NSI score.\n"
        "- Do NOT invent new categories.\n"
        "- Do NOT explain reasoning.\n"
    )

    messages = [{"role": "system", "content": system_prompt}]

    if summary:
        messages.append({
            "role": "system",
            "content": f"=== STORY CANON SUMMARY ===\n{summary}\n=== END SUMMARY ==="
        })

    if retrieved_hints and len(retrieved_hints) > 0:
        hint_block = "\n".join([f"- {h}" for h in retrieved_hints])
        messages.append({
            "role": "system",
            "content": f"=== KEY STORY MEMORY NOTES ===\n{hint_block}\n=== END NOTES ==="
        })

    if history:
        messages.extend(history)

    current_prompt = (
        f"Continue the story{genre_str}.\n\n"
        f"Scene instruction:\n{context}\n\n"
        "Write the next scene in immersive prose."
    )

    messages.append({"role": "user", "content": current_prompt})

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    raw_output = response.choices[0].message.content.strip()

    # Parse violations from <WRLD> block before stripping
    violations = parse_wrld_violations(raw_output)
    # Extract updated world rules for persistence
    updated_rules = extract_updated_rules(raw_output)

    # Strip <WRLD> metadata block so it doesn't appear in UI
    clean_output = re.sub(r"<WRLD>.*?</WRLD>", "", raw_output, flags=re.DOTALL).strip()

    return clean_output, violations, updated_rules


def generate_summary(history: list, current_summary: str = None) -> str:
    """
    Generate or update a rolling summary of the story context.
    """
    history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history])
    
    system_prompt = (
        "You are a narrative summarizer. Create a concise, high-density summary "
        "of the story so far. Focus on characters, locations, and key events.\n\n"
        "STRICT RULES:\n"
        "1. No Hallucinations: Do NOT introduce new facts, characters, or plot points. Only summarize provided content.\n"
        "2. Conciseness: Keep the total summary under 300 words.\n"
        "3. Focus: Prioritize plot-critical transitions and character state changes."
    )
    
    user_prompt = f"Update the following story summary with these new events:\n\n"
    if current_summary:
        user_prompt += f"CURRENT SUMMARY: {current_summary}\n\n"
    user_prompt += f"NEW EVENTS:\n{history_text}\n\nWrite a single cohesive, factual paragraph summary."

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3, # Low temperature for factual summarization
        max_tokens=600
    )
    
    return response.choices[0].message.content.strip()


def parse_wrld_violations(raw_output: str) -> dict:
    """
    Extract violation counts from the <WRLD> metadata block.
    Returns a dict with integer counts for each violation category.
    """
    violations = {
        "CHARACTER_INCONSISTENCY": 0,
        "TIMELINE_CONTRADICTION": 0,
        "WORLD_RULE_VIOLATION": 0,
        "IGNORED_FACT": 0
    }

    wrld_match = re.search(r"<WRLD>(.*?)</WRLD>", raw_output, flags=re.DOTALL)
    if not wrld_match:
        return violations

    wrld_block = wrld_match.group(1)

    for key in violations:
        match = re.search(rf"{key}\s*:\s*(\d+)", wrld_block)
        if match:
            violations[key] = int(match.group(1))

    return violations


def compute_nsi(violations: dict) -> int:
    """
    Deterministic Narrative Stability Index calculation.
    LLM detects violations, backend computes score.
    """
    score = 100
    score -= violations.get("CHARACTER_INCONSISTENCY", 0) * 10
    score -= violations.get("TIMELINE_CONTRADICTION", 0) * 10
    score -= violations.get("WORLD_RULE_VIOLATION", 0) * 15
    score -= violations.get("IGNORED_FACT", 0) * 5
    return max(score, 0)


def extract_updated_rules(raw_output: str) -> str:
    """
    Extract UPDATED_RULES from the <WRLD> metadata block.
    Returns the rules string, or empty string if not found.
    """
    wrld_match = re.search(r"<WRLD>(.*?)</WRLD>", raw_output, flags=re.DOTALL)
    if not wrld_match:
        return ""

    wrld_block = wrld_match.group(1)
    match = re.search(r"UPDATED_RULES\s*:\s*(.*?)(?=VIOLATION_COUNTS\s*:)", wrld_block, flags=re.DOTALL)
    return match.group(1).strip() if match else ""
