# RAG helper — loaded by the R Shiny app via reticulate.
#
# Flow:
#   1. On startup, embed the Mailchimp knowledge base using fastembed
#      (same all-MiniLM-L6-v2 model, runs via ONNX — no PyTorch needed)
#   2. When called, embed the user's email and retrieve top chunks
#   3. Send retrieved tips + email + prediction score to OpenRouter
#   4. Return a suggested revision and a list of key changes

import os
import re
import numpy as np
from openai import OpenAI
from fastembed import TextEmbedding
from sklearn.metrics.pairwise import cosine_similarity

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    import json
    keys_path = "keys.json"
    if os.path.exists(keys_path):
        with open(keys_path, "r") as f:
            api_key = json.load(f)["api_key"]

client = OpenAI(
    api_key = api_key,
    base_url = "https://openrouter.ai/api/v1"
)

# ── Mailchimp best-practice knowledge base ────────────────────────────────────
BEST_PRACTICES = [
    # Subject lines
    "Keep subject lines under 50 characters so they display fully on mobile. Mailchimp data shows shorter subject lines consistently achieve higher open rates.",
    "Use numbers and specifics in subject lines ('3 ways to boost sales') rather than vague promises. Specific subject lines outperform generic ones by up to 20%.",
    "Avoid spam trigger words in subject lines: FREE, URGENT, GUARANTEED, ACT NOW, LIMITED TIME. These dramatically increase the chance of landing in spam.",
    "Personalize the subject line with the recipient's first name or company when possible. Personalized subject lines lift open rates by an average of 26%.",
    "Ask a question in the subject line to spark curiosity: 'Are you making this email mistake?' Questions drive higher open and click-through rates.",
    "Use preview text (the snippet below the subject line) to extend your message — don't leave it blank or let it default to 'View in browser'.",

    # Email length and structure
    "Keep professional emails between 75 and 125 words for the highest response rates. Emails under 50 words feel curt; emails over 200 words lose readers.",
    "Front-load your most important information. Many readers skim — put the key ask or value proposition in the first two sentences.",
    "Use short paragraphs of 2–3 sentences maximum. Dense text blocks reduce readability and response rates.",
    "Use a single, clear call to action (CTA) per email. Multiple competing CTAs confuse readers and reduce clicks.",
    "Make the CTA specific and action-oriented: 'Book your free call' beats 'Click here' or 'Learn more'.",

    # Tone and clarity
    "Write like you speak to one person, not a mass audience. Use 'you' more than 'our customers' or 'subscribers'. Conversational tone lifts engagement.",
    "Lead with value to the reader, not features of your product. 'You'll save 3 hours a week' is more compelling than 'Our tool has an automation feature'.",
    "Avoid jargon and acronyms unless you're certain the reader knows them. Clarity always beats cleverness.",
    "Use active voice: 'We updated your account' not 'Your account has been updated'. Active voice is clearer and more direct.",
    "End with a clear next step or question. Emails that leave the reader knowing exactly what to do next get replied to faster.",

    # Urgency and engagement
    "Create a genuine sense of urgency with real deadlines ('Offer ends Friday') rather than artificial scarcity. Readers can detect fake urgency and it damages trust.",
    "Include a question in the email body to invite a reply. Even a simple 'Does this Thursday work for you?' dramatically increases response rates.",
    "Mirror the formality level of the recipient. A casual 'Hey [name]' works for warm contacts; a formal greeting is better for cold outreach.",
    "Follow up. Mailchimp research shows 70% of email replies come after at least one follow-up. A single send is rarely enough.",

    # Mobile and formatting
    "Avoid large image-heavy emails — many email clients block images by default. Your email should make sense even with images turned off.",
    "Use bullet points to break up key information. Bulleted lists are easier to skim and increase time-on-email.",
    "Test your email on mobile before sending. Over 60% of emails are opened on a mobile device.",

    # Sending time
    "Tuesday, Wednesday, and Thursday mornings between 9–11 AM in the recipient's time zone are the highest-performing send times per Mailchimp benchmark data.",
    "Avoid sending on Monday mornings (inbox is full) and Friday afternoons (people are checking out for the weekend).",
]

# Load embedding model and pre-compute KB embeddings at startup
_embedder  = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
_kb_matrix = np.array(list(_embedder.embed(BEST_PRACTICES)))


def _retrieve(email_text, top_k=5):
    """Return the top_k most relevant best-practice chunks for this email."""
    query_emb = np.array(list(_embedder.embed([email_text])))
    scores    = cosine_similarity(query_emb, _kb_matrix).flatten()
    top_idx   = np.argsort(scores)[::-1][:top_k]
    return [BEST_PRACTICES[i] for i in top_idx]


def suggest_revision(email_text, response_prob):
    """
    Run the full RAG pipeline and return a dict with:
      - 'revised_email': the suggested improved email
      - 'changes':       a list of short bullet-point explanations
    """
    context_chunks = _retrieve(email_text)
    context_block  = "\n".join(f"- {tip}" for tip in context_chunks)

    prompt = f"""You are an expert email copywriter trained on Mailchimp's email marketing research.

The user wrote the email below. A machine learning model predicts it has a {response_prob:.0f}% chance of getting a reply within 3 hours.

Your job:
1. Rewrite the email so it is more likely to get a fast reply. Use the actual words and content from the original email — do NOT use placeholder variables like {{{{name}}}}, {{{{company}}}}, or any template syntax. Write a complete, ready-to-send email using the real content provided.
2. List the 3–5 most impactful changes you made, each as a short bullet point.

Use the following Mailchimp best practices as guidance:
{context_block}

--- Original Email ---
{email_text}
--- End Original Email ---

Respond in this exact format:

REVISED EMAIL:
[your rewritten email here]

KEY CHANGES:
- [change 1]
- [change 2]
- [change 3]
"""

    response = client.chat.completions.create(
        model       = "openrouter/owl-alpha",
        messages    = [{"role": "user", "content": prompt}],
        temperature = 0.7,
        max_tokens  = 1200
    )

    raw = response.choices[0].message.content.strip()

    split_match = re.search(r'KEY CHANGES[:\s]*\n', raw, re.IGNORECASE)

    if split_match:
        revised_block = raw[:split_match.start()].strip()
        changes_block = raw[split_match.end():].strip()
        revised_email = re.sub(r'^[A-Z][A-Z\s]+:\s*\n', '', revised_block).strip()
        if not revised_email:
            revised_email = revised_block
        changes = [
            line.lstrip("-•* ").strip()
            for line in changes_block.splitlines()
            if re.match(r'^\s*[-•*\d]', line) and line.strip()
        ]
        if not changes:
            changes = [l.strip() for l in changes_block.splitlines() if l.strip()]
    else:
        revised_email = re.sub(r'^[A-Z][A-Z\s]+:\s*\n', '', raw).strip() or raw
        changes = ["Review the revised email above for all improvements made."]

    return {"revised_email": revised_email, "changes": changes}
