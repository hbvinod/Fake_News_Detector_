# app.py (patched)
import os
import re
import json
import uvicorn
import requests
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fake-news-backend")

# ---------------------------------------------------------
# ENVIRONMENT VARIABLES
# ---------------------------------------------------------
load_dotenv()
CHATGPT_KEY = os.getenv("CHATGPT_KEY")
SERP_ID = os.getenv("SERP_ID")
GROQ_KEY = os.getenv("GROQ_ID")  # user requested env name

if not CHATGPT_KEY:
    logger.warning("CHATGPT_KEY not set. OpenAI calls will fail until set.")
if not SERP_ID:
    logger.warning("SERP_ID not set. SerpAPI calls will fail until set.")
if not GROQ_KEY:
    logger.info("GROQ_ID not provided — Groq step will be skipped (optional).")

# Initialize OpenAI client (your code used OpenAI(api_key=...))
client = OpenAI(api_key=CHATGPT_KEY)

# ---------------------------------------------------------
# FASTAPI SETUP
# ---------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\-']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_related_news(query: str, top_n: int = 8) -> List[Dict]:
    """SERP NEWS for comparison"""
    if not SERP_ID:
        logger.debug("No SERP_ID provided; returning empty related news.")
        return []
    url = "https://serpapi.com/search"
    params = {"engine": "google_news", "q": query, "api_key": SERP_ID}
    try:
        res = requests.get(url, params=params, timeout=10).json()
        return res.get("news_results", [])[:top_n]
    except Exception as e:
        logger.error(f"SERP news error: {e}")
        return []


def fetch_general_search(query: str, top_n: int = 8) -> List[Dict]:
    """SERP WEB search results"""
    if not SERP_ID:
        logger.debug("No SERP_ID provided; returning empty web results.")
        return []
    url = "https://serpapi.com/search"
    params = {"engine": "google", "q": query, "api_key": SERP_ID}
    try:
        res = requests.get(url, params=params, timeout=10).json()
        out = []
        for item in res.get("organic_results", [])[:top_n]:
            out.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", "")
            })
        return out
    except Exception as e:
        logger.error(f"SERP web error: {e}")
        return []


# ---------------------------------------------------------
# PERSON NAME DETECTION
# ---------------------------------------------------------
PERSON_NAME_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,3})\b")


def detect_person_name(text: str) -> Optional[str]:
    matches = PERSON_NAME_RE.findall(text)
    if matches:
        matches = sorted(matches, key=len, reverse=True)
        return matches[0]
    return None


def fetch_profile_sources(name: str, top_n: int = 6):
    queries = [
        f"{name} wikipedia",
        f"{name} biography",
        f"{name} interview",
        f"{name} profile",
        f"{name} news",
    ]
    out = []
    for q in queries:
        hits = fetch_general_search(q, 3)
        for h in hits:
            if h not in out and h.get("link"):
                out.append(h)
            if len(out) >= top_n:
                return out
    return out[:top_n]


# ---------------------------------------------------------
# GPT SYSTEM PROMPT (UPGRADED)
# ---------------------------------------------------------
SYSTEM_INSTRUCTIONS = """
You are an expert fact-checker. Your job is to break a news text into distinct factual claims
and judge each using EVIDENCE ONLY from the search results provided.

You MUST output JSON with EXACTLY this structure:

{
  "claims": [
    {
      "id": "c1",
      "text": "<claim sentence>",
      "status": "REAL | FAKE | CONTRADICTED | UNKNOWN",
      "confidence": 0.0-1.0,
      "supporting_evidence": ["title — link", ...],
      "counter_evidence": ["title — link", ...],
      "explanation": "<short reasoning>"
    }
  ],
  "summary": "<overall summary>"
}

Rules:
- Mark a claim REAL only if matching credible evidence exists.
- Mark FAKE or CONTRADICTED if strong evidence disproves the claim.
- Mark UNKNOWN if you cannot find direct matches in the evidence.
- ALWAYS include evidence URLs when available.
- NEVER output anything except the JSON object.
"""

# ---------------------------------------------------------
# OPTIONAL GROQ STEP (attempt to extract structured claims)
# ---------------------------------------------------------
def call_groq_extract(text: str) -> Optional[Dict[str, Any]]:
    """
    Optional: use GROQ (if GROQ_ID provided) to pre-extract claims.
    This is best-effort: if Groq API shape differs, it will fail and we fallback to GPT.
    """
    if not GROQ_KEY:
        return None

    try:
        # NOTE: Groq API endpoint & request shape may vary by account/product.
        # This implementation uses a generic POST to an inference endpoint — adjust if your Groq endpoint is different.
        url = "https://api.groq.ai/v1/infer"  # placeholder path — adjust if your Groq endpoint differs
        headers = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"}
        payload = {"input": text, "instruction": "Extract factual claims as JSON list of strings."}
        r = requests.post(url, json=payload, headers=headers, timeout=10)
        if r.status_code != 200:
            logger.warning("Groq call failed status=%s body=%s", r.status_code, r.text)
            return None
        data = r.json()
        # Expecting something like {"claims": ["claim1", ...]} — adapt if your Groq returns different structure.
        if isinstance(data, dict) and "claims" in data and isinstance(data["claims"], list):
            return data
        # otherwise attempt to parse text
        if isinstance(data, dict) and "output" in data:
            # try to parse JSON inside output
            try:
                parsed = json.loads(data["output"])
                return parsed
            except Exception:
                return {"claims": data.get("output", [])}
        return None
    except Exception as e:
        logger.warning("Groq extraction failed: %s", e)
        return None


# ---------------------------------------------------------
# GPT CALL
# ---------------------------------------------------------
def call_gpt(headline: str, profiles: List[str], related: List[str]) -> Dict[str, Any]:
    prompt = f"""
Headline:
{headline}

Authoritative Profiles:
{chr(10).join('- ' + p for p in profiles)}

Related Evidence:
{chr(10).join('- ' + r for r in related)}

Extract claims. Judge each claim STRICTLY using above evidence.
Return a JSON object with 'claims' array and 'summary'.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        txt = response.choices[0].message.content
        logger.info("GPT RAW JSON → %s", txt)
        try:
            return json.loads(txt)
        except Exception as e:
            logger.error("Failed to parse GPT JSON: %s", e)
            # Attempt to recover by returning raw string wrapped
            return {"claims": [], "summary": txt}
    except Exception as e:
        # Capture rate limit and other errors
        logger.exception("OpenAI call failed: %s", e)
        # If it's a rate-limit from OpenAI, craft a clear response for the frontend
        err_msg = str(e)
        if "rate limit" in err_msg.lower() or "429" in err_msg:
            raise RuntimeError("OpenAI rate limit or quota reached: " + err_msg)
        return {"claims": [], "summary": "OpenAI call failed."}


# ---------------------------------------------------------
# VERDICT AGGREGATION (UPDATED)
# ---------------------------------------------------------
def aggregate_claims(claims: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not claims:
        return {
            "verdict": "Mixed",
            "confidence": 0.0,
            "explanation": "No claims found.",
            "supporting_evidence": [],
            "counter_evidence": [],
            "details": []
        }

    counts = {"REAL": 0, "FAKE": 0, "CONTRADICTED": 0, "UNKNOWN": 0}
    support = []
    counter = []
    conf_sum = 0.0

    for c in claims:
        status = c.get("status", "UNKNOWN").upper()
        if status not in counts:
            status = "UNKNOWN"
        counts[status] += 1
        try:
            conf_sum += float(c.get("confidence", 0.0))
        except Exception:
            conf_sum += 0.0

        for s in c.get("supporting_evidence", []):
            if s and s not in support:
                support.append(s)
        for s in c.get("counter_evidence", []):
            if s and s not in counter:
                counter.append(s)

    n = len(claims)
    avg_conf = round((conf_sum / n) if n else 0.0, 3)

    real = counts["REAL"]
    fake = counts["FAKE"] + counts["CONTRADICTED"]
    unknown = counts["UNKNOWN"]

    # FINAL VERDICT LOGIC - map to allowed labels exactly:
    # Real, Fake, Partially Real, Partially Fake, Mixed
    if real == n:
        verdict = "Real"
    elif fake == n:
        verdict = "Fake"
    elif real > fake:
        verdict = "Partially Real"
    elif fake > real:
        verdict = "Partially Fake"
    else:
        # equal or unclear
        verdict = "Mixed"

    explanation = f"{real} real claim(s), {fake} fake/contradicted claim(s), {unknown} unknown claim(s)."

    return {
        "verdict": verdict,
        "confidence": max(0.0, min(1.0, float(avg_conf))),
        "explanation": explanation,
        "supporting_evidence": support,
        "counter_evidence": counter,
        "details": claims
    }


# ---------------------------------------------------------
# MAIN PREDICTION PIPELINE
# ---------------------------------------------------------
def generate_prediction(text: str) -> Dict[str, Any]:
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "verdict": "Mixed",
            "confidence": 0.0,
            "explanation": "Empty input after cleaning.",
            "supporting_evidence": [],
            "counter_evidence": [],
            "details": []
        }

    # person detection -> profile sources
    person = detect_person_name(cleaned)
    profiles = []
    if person:
        hits = fetch_profile_sources(person)
        profiles = [f"{h.get('title','').strip()} — {h.get('link','').strip()}" for h in hits if h.get("link")]

    # related evidence (news)
    related_hits = fetch_related_news(cleaned)
    related = []
    for r in related_hits:
        title = r.get("title", "").strip()
        link = r.get("link", "").strip()
        # prefer title — link format
        if title and link:
            related.append(f"{title} — {link}")
        elif title:
            related.append(title)
        elif link:
            related.append(link)

    # Try Groq-based claim extraction (best-effort)
    gre = None
    if GROQ_KEY:
        try:
            gre = call_groq_extract(cleaned)
            if gre and isinstance(gre, dict) and gre.get("claims"):
                logger.info("Groq provided claims; transforming into structured claims.")
                extracted_claims = []
                for i, claim_text in enumerate(gre.get("claims", [])[:30]):
                    # tentative claim object minimal — will be judged by GPT below preferably, but if Groq provides also evidence use it
                    extracted_claims.append({
                        "id": f"c{i+1}",
                        "text": claim_text if isinstance(claim_text, str) else str(claim_text),
                        "status": "UNKNOWN",
                        "confidence": 0.0,
                        "supporting_evidence": [],
                        "counter_evidence": [],
                        "explanation": ""
                    })
                # If we have extracted claims from Groq, ask GPT to judge them using the evidence
                try:
                    # Build a prompt to instruct GPT to judge the provided claims using the evidence lists
                    claim_text_block = "\n".join(f"- {c['text']}" for c in extracted_claims)
                    prompt = f"Given the following claims:\n{claim_text_block}\n\nUse ONLY the evidence listed under 'Authoritative Profiles' and 'Related Evidence' to judge each claim. Return JSON with 'claims' array objects as specified in system instructions."
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                            {"role": "user", "content": prompt + "\n\nAuthoritative Profiles:\n" + "\n".join(profiles) + "\n\nRelated Evidence:\n" + "\n".join(related)}
                        ],
                        max_tokens=1200
                    )
                    txt = response.choices[0].message.content
                    logger.info("GPT judged Groq-extracted claims.")
                    try:
                        judged = json.loads(txt)
                        claims = judged.get("claims", [])
                    except Exception:
                        logger.warning("Failed to parse GPT judgement JSON after Groq; falling back to single-pass GPT.")
                        claims = []
                except Exception as e:
                    logger.warning("GPT judgement of Groq claims failed: %s", e)
                    claims = []
            else:
                claims = []
        except Exception as e:
            logger.warning("Groq step failed: %s", e)
            claims = []
    else:
        claims = []

    # If no claims from Groq+judge, fallback to single GPT pass
    if not claims:
        try:
            gpt_result = call_gpt(cleaned, profiles, related)
            claims = gpt_result.get("claims", [])
            summary = gpt_result.get("summary", "")
        except RuntimeError as e:
            # e.g., OpenAI rate limit -> return helpful response
            logger.error("OpenAI rate limit or critical failure: %s", e)
            return {
                "verdict": "Mixed",
                "confidence": 0.0,
                "explanation": "OpenAI rate limit or quota reached — please try later.",
                "supporting_evidence": related[:6],
                "counter_evidence": [],
                "details": [],
                "summary": str(e)
            }
        except Exception as e:
            logger.exception("GPT pipeline unexpectedly failed: %s", e)
            return {
                "verdict": "Mixed",
                "confidence": 0.0,
                "explanation": "Internal error while generating prediction.",
                "supporting_evidence": related[:6],
                "counter_evidence": [],
                "details": [],
                "summary": ""
            }
    else:
        summary = ""

    # Ensure structure and safe defaults for each claim
    normalized_claims = []
    for i, c in enumerate(claims):
        # Some GPT responses may already be dicts; some may be strings.
        try:
            if isinstance(c, str):
                text = c
                status = "UNKNOWN"
                confidence = 0.0
                sup = []
                cnt = []
                explanation = ""
            elif isinstance(c, dict):
                text = c.get("text", "") or c.get("claim", "") or ""
                status = (c.get("status") or c.get("verdict") or "UNKNOWN").upper()
                confidence = float(c.get("confidence", 0.0)) if c.get("confidence") is not None else 0.0
                sup = c.get("supporting_evidence", []) or []
                cnt = c.get("counter_evidence", []) or []
                explanation = c.get("explanation", "") or c.get("reason", "") or ""
            else:
                text = str(c)
                status = "UNKNOWN"
                confidence = 0.0
                sup = []
                cnt = []
                explanation = ""
        except Exception:
            text = str(c)
            status = "UNKNOWN"
            confidence = 0.0
            sup = []
            cnt = []
            explanation = ""

        # Normalize confidence to 0..1 if >1 (percent)
        try:
            if isinstance(confidence, (int, float)) and confidence > 1.0:
                confidence = min(1.0, float(confidence) / 100.0)
        except Exception:
            confidence = 0.0

        # Convert evidence items into "title — url" when possible (if Serp results contain link/title)
        normalized_sup = []
        for s in sup:
            s_str = str(s).strip()
            normalized_sup.append(s_str)
        normalized_cnt = []
        for s in cnt:
            s_str = str(s).strip()
            normalized_cnt.append(s_str)

        normalized_claims.append({
            "id": f"c{i+1}",
            "text": text,
            "status": status,
            "confidence": confidence,
            "supporting_evidence": normalized_sup,
            "counter_evidence": normalized_cnt,
            "explanation": explanation
        })

    # Aggregate final verdict
    final = aggregate_claims(normalized_claims)
    # Include summary if available
    final["summary"] = summary

    # Ensure supporting/counter evidence lists include helpful items (add related/profile hits if none)
    if not final.get("supporting_evidence"):
        # provide up to 6 related/profile hits as supporting context if present
        extras = []
        for p in profiles:
            if p not in extras:
                extras.append(p)
            if len(extras) >= 6:
                break
        for r in related:
            if r not in extras and len(extras) < 6:
                extras.append(r)
        final["supporting_evidence"] = extras

    # ensure counter_evidence at least empty list
    final["counter_evidence"] = final.get("counter_evidence", [])

    return final


# ---------------------------------------------------------
# API MODEL
# ---------------------------------------------------------
class PredictRequest(BaseModel):
    text: str


# ---------------------------------------------------------
# API ROUTE
# ---------------------------------------------------------
@app.post("/api/predict")
def api_predict(req: PredictRequest):
    try:
        result = generate_prediction(req.text)
        return JSONResponse(content=result)
    except RuntimeError as e:
        # Known runtime errors (like OpenAI rate-limit) — return structured response
        logger.error("Prediction failure (runtime): %s", e)
        return JSONResponse(status_code=429, content={
            "verdict": "Mixed",
            "confidence": 0.0,
            "explanation": f"Temporary failure: {str(e)}",
            "supporting_evidence": [],
            "counter_evidence": [],
            "details": []
        })
    except Exception as e:
        logger.exception("Prediction failure: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# SERVE FRONTEND
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse("static/index.html")


# ---------------------------------------------------------
# RUN SERVER
# ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
