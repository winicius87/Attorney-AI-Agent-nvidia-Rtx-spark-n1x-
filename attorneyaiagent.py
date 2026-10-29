import os
import json
import ollama

# -------------------------------------------------------------------------
# HARDWARE CONFIGURATION FOR 128GB N1X LAPTOP
# -------------------------------------------------------------------------
# Qwen3.5-122B Q4 requires ~72GB of VRAM. 
# On a 128GB N1X, we can aggressively allocate context up to 64k-128k tokens.
# -------------------------------------------------------------------------

MODEL_NAME = "qwen3.5:122b"

def generate_court_paper(case_facts, document_type, jurisdiction):
    """
    Spawns the Legal Agent using Qwen3.5-122B in native Deep Thinking Mode
    to draft structured, high-fidelity legal pleadings.
    """
    
    # Strict professional guardrails for legal formatting and citation ethics
    system_prompt = (
        "You are an elite, highly precise Autonomous AI Attorney Agent specializing in drafting court pleadings.\n"
        "Your task is to write high-fidelity, professional legal documents based on the user's case facts.\n\n"
        
        "CRITICAL OPERATIONAL DIRECTIVES:\n"
        "1. STRICT JURISDICTIONAL ADHERENCE: Format the document caption, signature block, and text to precisely match the laws, civil procedure rules, and styling conventions of the requested jurisdiction.\n"
        "2. ZERO HALLUCINATION POLICY: Never invent, guess, or hallucinate case law, statutes, or rules. If you need a specific case citation or rule number but do not have it in your local context, insert a distinct placeholder like: **[INSERT RELEVANT CASE CITATION HERE]**.\n"
        "3. LOGICAL CHAIN-OF-THOUGHT: Use your thinking capacity to break down the legal elements of the claims or defenses before writing the final motion text.\n"
        "4. FORMATTING: Output the document cleanly with structured headers (e.g., I. PRELIMINARY STATEMENT, II. STATEMENT OF FACTS, III. ARGUMENT).\n"
    )

    user_content = f"""
    JURISDICTION: {jurisdiction}
    DOCUMENT TYPE REQUESTED: {document_type}
    
    RAW CASE FACTS & EVIDENCE:
    ---
    {case_facts}
    ---
    
    Please reason step-by-step through the legal theory, outline the elements of the required arguments, and generate the formal court document.
    """

    print("Initializing Qwen3.5-122B on N1X Superchip...")
    print("Activating Deep Thinking Mode (Chain-of-Thought processing enabled)...\n")

    try:
        # Stream the response so the attorney can watch the reasoning unfold in real time
        response_stream = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            options={
                "temperature": 0.2,       # Low temperature for highly predictable, conservative legal logic
                "num_ctx": 65536,         # Huge 64k token context window optimized for 128GB unified memory
                "top_p": 0.9,
                # Force the model to leverage its internal reasoning/thinking budget
                "thinking_mode": "enabled", 
                "num_predict": 8192        # Allow for long-form, multi-page legal document generation
            },
            stream=True
        )

        print("=== AGENT THINKING & DRAFTING IN PROGRESS ===")
        for chunk in response_stream:
            content = chunk.get('message', {}).get('content', '')
            print(content, end='', flush=True)
        print("\n\n=== GENERATION COMPLETE ===")

    except Exception as e:
        print(f"\nAn error occurred while connecting to Ollama: {e}")
        print(f"Please verify that Ollama is running locally and that '{MODEL_NAME}' is fully pulled.")

# -------------------------------------------------------------------------
# EXAMPLE TEST RUN
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Test Data: Simple Breach of Contract scenario
    test_facts = """
    Plaintiff: Gotham Logistics LLC. Defendant: Wayne Enterprises Inc.
    On June 12, 2025, parties entered into a written freight hauling contract. 
    Wayne Enterprises agreed to pay $15,000 per month for dedicated shipping lanes.
    Wayne Enterprises failed to pay invoices for August, September, and October 2025.
    Total outstanding balance is $45,000. 
    Gotham Logistics sent a formal demand letter on November 1, 2025, but received no response.
    Contract contains a standard attorney's fees provision for the prevailing party.
    """
    
    target_doc = "Civil Complaint for Breach of Contract"
    target_jurisdiction = "State of New York, Supreme Court, New York County"
    
    generate_court_paper(test_facts, target_doc, target_jurisdiction)
