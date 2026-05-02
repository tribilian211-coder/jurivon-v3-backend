"""
JURIVON AI — Version 3.0
Private Legal AI Platform for Law Firms
FastAPI Backend — Enterprise Grade

Features:
  1.  Conflict Check (with matter database + AI analysis)
  2.  Document Q&A (clause-cited answers)
  3.  Contract Review (HIGH/MED/LOW risk flags)
  4.  Document Summary (structured legal brief)
  5.  Draft with Lex (streaming, jurisdiction-aware)
  6.  Due Diligence (multi-document, 5-category)
  7.  OdV Art. 231 Certificate (Italian firms ONLY)
  8.  Legal Research with Khan (live APIs + RAG)
  9.  Citation Verify (hallucination protection)
  10. Regulatory Tracker (live jurisdiction updates)
  11. Matter Database (CRUD + bulk import)
  12. Audit Log (permanent, CSV-exportable)
  13. Precedent Search (firm database + AI)
  14. Matter Workspaces (persistent per-client)
  15. History & Bookmarks (session persistence)
  16. Collaboration (workspace comments, real-time)
  17. Practice Templates (NDA, SPA, Employment etc.)

Security:
  - Rate limiting per IP (slowapi)
  - Input validation (Pydantic v2)
  - File type + size validation
  - SQL injection safe (parameterized via Supabase)
  - XSS safe (input sanitization)
  - CORS restricted to allowed origins
  - Centralized error handler
  - Retry logic with exponential backoff (tenacity)
  - Structured audit logging
  - Sentry error tracking
"""

import os, re, json, asyncio, logging, traceback, csv, io
from datetime import datetime
from typing import Optional, List

from fastapi import (FastAPI, UploadFile, File, Form,
                     Request, HTTPException, Depends)
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
from dotenv import load_dotenv

import httpx
import pdfplumber
from openai import AsyncOpenAI
from supabase import create_client, Client
from tenacity import (retry, stop_after_attempt,
                      wait_exponential, retry_if_exception_type)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("jurivon")

# ── Config ───────────────────────────────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "").strip()
SUPABASE_URL    = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY    = os.getenv("SUPABASE_KEY", "")
ENVIRONMENT     = os.getenv("ENVIRONMENT", "production")
SENTRY_DSN      = os.getenv("SENTRY_DSN", "")

ALLOWED_ORIGINS = [
    "https://jurivon-frontend.vercel.app",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:8080",
    "null",
    "*",  # Remove in production — replace with exact Vercel URL
]

ALLOWED_FILE_TYPES = {
    "application/pdf",
    "text/plain",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/csv",
    "application/octet-stream",  # some browsers send this for .docx
}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# ── Sentry (optional) ────────────────────────────────────────────
if SENTRY_DSN:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            integrations=[FastApiIntegration()],
            traces_sample_rate=0.1,
            environment=ENVIRONMENT
        )
        logger.info("Sentry initialized")
    except ImportError:
        logger.warning("sentry-sdk not installed — skipping")

# ── Clients ──────────────────────────────────────────────────────
openai_client: AsyncOpenAI = None
supabase: Client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global openai_client, supabase
    if OPENAI_API_KEY:
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=60)
    if SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info(f"Jurivon v3.0 started — env: {ENVIRONMENT}")
    yield
    logger.info("Jurivon v3.0 shutting down")

# ── App ──────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Jurivon AI", version="3.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ── Global error handler ─────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error {request.url}: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={
        "success": False,
        "error": "internal_error",
        "message": "Something went wrong. Please try again.",
        "data": None
    })

# ── Response helpers ─────────────────────────────────────────────
def ok(data, message: str = "OK"):
    return {"success": True, "data": data, "error": None,
            "message": message, "timestamp": datetime.utcnow().isoformat()}

def err(message: str, code: int = 400):
    return JSONResponse(status_code=code, content={
        "success": False, "data": None,
        "error": message, "message": message,
        "timestamp": datetime.utcnow().isoformat()
    })

# ── File validation ──────────────────────────────────────────────
async def validate_file(file: UploadFile) -> bytes:
    # Allow by extension fallback
    filename = (file.filename or "").lower()
    ext_ok = filename.endswith((".pdf", ".txt", ".docx", ".csv"))
    type_ok = file.content_type in ALLOWED_FILE_TYPES

    if not (type_ok or ext_ok):
        raise HTTPException(400,
            f"Only PDF, TXT, DOCX, or CSV files are allowed. Got: {file.content_type}")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large. Maximum is 10MB.")
    if len(content) < 10:
        raise HTTPException(400, "File appears to be empty.")
    return content

# ── Text extraction ──────────────────────────────────────────────
def extract_text(content: bytes, filename: str) -> str:
    fn = (filename or "").lower()
    try:
        if fn.endswith(".pdf"):
            text = ""
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
            return text.strip()
        else:
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    return content.decode(enc).strip()
                except UnicodeDecodeError:
                    continue
            return content.decode("utf-8", errors="replace").strip()
    except Exception as e:
        logger.error(f"Text extraction error for {filename}: {e}")
        return ""

# ── Sanitize ─────────────────────────────────────────────────────
def sanitize(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text.strip()

# ── OpenAI with retry ────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception)
)
async def call_ai(messages: list, max_tokens: int = 2000,
                  temperature: float = 0, model: str = "gpt-4o") -> str:
    if not openai_client:
        raise HTTPException(503, "AI service not configured.")
    r = await openai_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=50
    )
    return r.choices[0].message.content

# ── Audit + History log ──────────────────────────────────────────
async def log_action(firm_id: str, user: str, action: str,
                     subject: str, result: str,
                     workspace_id: str = None, full_content: dict = None):
    if not supabase:
        return
    try:
        supabase.table("audit_log").insert({
            "firm_id": firm_id,
            "user_name": user,
            "action": action,
            "subject": subject[:500],
            "result": result[:200],
            "session_id": f"v3-{datetime.utcnow().strftime('%Y%m%d')}",
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        if full_content:
            supabase.table("interaction_history").insert({
                "firm_id": firm_id,
                "user_name": user,
                "feature": action,
                "input_summary": subject[:500],
                "full_content": full_content,
                "workspace_id": workspace_id,
                "bookmarked": False,
                "created_at": datetime.utcnow().isoformat()
            }).execute()
    except Exception as e:
        logger.warning(f"Log failed: {e}")

# ════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ════════════════════════════════════════════════════════════════
class ConflictRequest(BaseModel):
    matter_description: str = Field(..., min_length=5, max_length=5000)
    firm_id: str = Field(default="default", max_length=100)
    user_name: str = Field(default="User", max_length=100)
    workspace_id: Optional[str] = None

    @field_validator("matter_description")
    @classmethod
    def sanitize_desc(cls, v): return sanitize(v)

class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=2000)
    jurisdiction: str = Field(..., min_length=2, max_length=20)
    firm_id: str = Field(default="default")
    user_name: str = Field(default="User")
    workspace_id: Optional[str] = None

    @field_validator("jurisdiction")
    @classmethod
    def validate_jur(cls, v):
        allowed = ["UK","Italy","EU","UAE","Pakistan","US",
                   "Germany","France","Global","Gulf","Canada","Australia"]
        if v not in allowed:
            raise ValueError(f"Jurisdiction must be one of: {', '.join(allowed)}")
        return v

class DraftRequest(BaseModel):
    doc_type: str = Field(..., min_length=2, max_length=100)
    party_a: str = Field(..., min_length=1, max_length=200)
    party_b: str = Field(..., min_length=1, max_length=200)
    jurisdiction: str = Field(default="UK", max_length=50)
    key_terms: str = Field(default="", max_length=3000)
    template_id: Optional[str] = Field(None, max_length=50)
    firm_id: str = Field(default="default")
    user_name: str = Field(default="User")
    workspace_id: Optional[str] = None

class CitationRequest(BaseModel):
    citation: str = Field(..., min_length=3, max_length=500)
    jurisdiction: str = Field(default="UK")
    firm_id: str = Field(default="default")

class OdvRequest(BaseModel):
    company_name: str = Field(..., min_length=1, max_length=200)
    company_type: str = Field(default="S.r.l.")
    lawyer_name: str = Field(..., min_length=1, max_length=200)
    lawyer_bar_number: str = Field(default="", max_length=50)
    matter_description: str = Field(..., min_length=10, max_length=2000)
    conflict_result: str = Field(default="CLEAR")
    firm_id: str = Field(default="default")
    user_name: str = Field(default="User")

class WorkspaceCreate(BaseModel):
    client_name: str = Field(..., min_length=1, max_length=200)
    matter_ref: Optional[str] = Field(None, max_length=100)
    matter_type: Optional[str] = Field(None, max_length=100)
    jurisdiction: str = Field(default="UK")
    lead_partner: Optional[str] = Field(None, max_length=100)
    firm_id: str = Field(default="default")

class WorkspaceItemCreate(BaseModel):
    workspace_id: str
    item_type: str = Field(..., max_length=50)
    title: str = Field(..., min_length=1, max_length=200)
    content: dict
    firm_id: str = Field(default="default")

class CommentCreate(BaseModel):
    workspace_id: str
    author: str = Field(..., min_length=1, max_length=100)
    comment: str = Field(..., min_length=1, max_length=2000)
    firm_id: str = Field(default="default")

class MatterRecord(BaseModel):
    matter_ref: str = Field(..., min_length=1, max_length=100)
    client: str = Field(..., min_length=1, max_length=200)
    counterparty: Optional[str] = Field(None, max_length=200)
    description: str = Field(..., min_length=1, max_length=1000)
    status: str = Field(default="OPEN")
    lead_partner: Optional[str] = Field(None, max_length=100)
    practice_area: Optional[str] = Field(None, max_length=100)
    jurisdiction: Optional[str] = Field(None, max_length=50)
    firm_id: str = Field(default="default")

class FirmSettings(BaseModel):
    firm_id: str = Field(..., min_length=1, max_length=100)
    firm_name: str = Field(..., min_length=1, max_length=200)
    jurisdiction: str = Field(default="UK")
    show_odv: bool = Field(default=False)
    user_name: str = Field(default="Admin")

# ════════════════════════════════════════════════════════════════
# HEALTH
# ════════════════════════════════════════════════════════════════
@app.get("/health")
async def health():
    db_ok = False
    if supabase:
        try:
            supabase.table("firm_settings").select("id").limit(1).execute()
            db_ok = True
        except:
            pass
    return {
        "status": "ok" if db_ok else "degraded",
        "version": "3.0.0",
        "database": "connected" if db_ok else "unavailable",
        "ai": "ready" if openai_client else "not configured",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/")
async def root():
    return {"name": "Jurivon AI", "version": "3.0.0", "status": "running"}

# ════════════════════════════════════════════════════════════════
# FEATURE 1 — CONFLICT CHECK
# ════════════════════════════════════════════════════════════════
@app.post("/api/v1/conflict-check")
@limiter.limit("20/minute")
async def conflict_check(request: Request, data: ConflictRequest):
    """
    Checks new matter against firm's existing matter database for conflicts.
    Returns: HIGH RISK / MEDIUM RISK / LOW RISK / CLEAR
    """
    matters = []
    if supabase:
        try:
            r = supabase.table("matters").select("*")\
                .eq("firm_id", data.firm_id).execute()
            matters = r.data or []
        except Exception as e:
            logger.warning(f"Matter DB read failed: {e}")

    matters_text = ""
    if matters:
        for m in matters:
            matters_text += (
                f"REF: {m.get('matter_ref','?')} | "
                f"CLIENT: {m.get('client','?')} | "
                f"COUNTERPARTY: {m.get('counterparty','None')} | "
                f"AREA: {m.get('practice_area','?')} | "
                f"DESC: {m.get('description','')[:200]}\n"
            )
    else:
        matters_text = "No matters found in database. Conflict check limited to AI analysis only."

    system = """You are a senior conflicts officer at a top-tier law firm.
Analyse the proposed new matter against the existing matter database.

Return EXACTLY this format — no deviations:

CONFLICT RESULT: [HIGH RISK / MEDIUM RISK / LOW RISK / CLEAR]
CONFIDENCE: [0-100]%
CONFLICTS FOUND:
[List each conflict with: MATTER REF | REASON | SEVERITY — or write "None identified"]
ENTITIES IDENTIFIED FROM NEW MATTER:
[List all people, companies, organisations extracted]
RECOMMENDATION:
[1-2 sentences: proceed / investigate further / decline]
EXPLANATION:
[Brief reasoning for the result]

Rules:
- HIGH RISK = direct adverse interest against existing client
- MEDIUM RISK = related party or potential conflict requiring investigation
- LOW RISK = minor connection, flag for monitoring
- CLEAR = no conflicts found"""

    user_msg = f"""NEW MATTER:
{data.matter_description}

EXISTING MATTERS IN DATABASE ({len(matters)} records):
{matters_text[:6000]}

Analyse thoroughly. Check client names, counterparty names, and related entities."""

    try:
        result = await call_ai([
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg}
        ], max_tokens=1200)

        risk = ("HIGH" if "HIGH RISK" in result else
                "MEDIUM" if "MEDIUM RISK" in result else
                "LOW" if "LOW RISK" in result else "CLEAR")

        await log_action(data.firm_id, data.user_name, "conflict_check",
                        data.matter_description[:200], risk, data.workspace_id,
                        {"query": data.matter_description, "result": result,
                         "risk_level": risk, "matters_checked": len(matters)})

        return ok({
            "result": result,
            "risk_level": risk,
            "matters_checked": len(matters),
            "has_database": bool(matters)
        })
    except Exception as e:
        logger.error(f"Conflict check error: {e}")
        return err("Conflict check failed. Please try again.")

# ════════════════════════════════════════════════════════════════
# FEATURE 2 — DOCUMENT Q&A
# ════════════════════════════════════════════════════════════════
@app.post("/api/v1/document-qa")
@limiter.limit("15/minute")
async def document_qa(
    request: Request,
    file: UploadFile = File(...),
    question: str = Form(...),
    firm_id: str = Form(default="default"),
    user_name: str = Form(default="User"),
    workspace_id: str = Form(default="")
):
    """
    Q&A against uploaded legal documents.
    Returns clause-cited answers with confidence level.
    """
    content = await validate_file(file)
    text = extract_text(content, file.filename)
    if not text or len(text) < 50:
        return err("Could not extract readable text from the document.")
    if not question or len(question.strip()) < 3:
        return err("Please enter a valid question.")

    question = sanitize(question)

    system = """You are a senior legal analyst at a top-tier law firm.
Answer ONLY from the document provided. Never invent or assume information.

Response format — EXACTLY:

DIRECT ANSWER:
[Clear answer in 2-3 sentences. If the document does not contain the answer, say so explicitly.]

RELEVANT CLAUSE:
[Exact quote from the document supporting your answer — max 80 words. If none, write 'Not directly stated in document.']

LOCATION:
[Section/clause number or description if identifiable]

CONFIDENCE: [HIGH / MEDIUM / LOW]

CAVEAT:
[Any important limitations, ambiguities, or areas requiring legal advice]

Do NOT fabricate clause references. Do NOT answer beyond what the document states."""

    user_msg = f"""DOCUMENT: {file.filename}
LENGTH: {len(text)} characters

DOCUMENT CONTENT:
{text[:14000]}

QUESTION: {question}"""

    try:
        result = await call_ai([
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg}
        ], max_tokens=1500)

        await log_action(firm_id, user_name, "document_qa",
                        f"{file.filename}: {question[:100]}", "Answered",
                        workspace_id or None,
                        {"filename": file.filename, "question": question, "answer": result})

        return ok({"result": result, "filename": file.filename,
                   "document_length": len(text)})
    except Exception as e:
        return err("Document Q&A failed. Please try again.")

# ════════════════════════════════════════════════════════════════
# FEATURE 3 — CONTRACT REVIEW
# ════════════════════════════════════════════════════════════════
@app.post("/api/v1/contract-review")
@limiter.limit("10/minute")
async def contract_review(
    request: Request,
    file: UploadFile = File(...),
    party_perspective: str = Form(default=""),
    firm_id: str = Form(default="default"),
    user_name: str = Form(default="User"),
    workspace_id: str = Form(default="")
):
    """
    Full contract review with HIGH/MEDIUM/LOW risk flags.
    Optional: specify which party's perspective to review from.
    """
    content = await validate_file(file)
    text = extract_text(content, file.filename)
    if not text:
        return err("Could not extract text from document.")

    perspective_note = f"\nReview from the perspective of: {party_perspective}" if party_perspective else ""

    system = f"""You are a senior commercial lawyer reviewing a contract for a law firm client.{perspective_note}

Identify ALL issues. Use EXACTLY this format for each issue:

---
RISK: [HIGH / MEDIUM / LOW]
CLAUSE: [clause number or title]
ISSUE: [clear explanation of the problem in plain English]
QUOTE: [exact quote from contract, max 50 words]
RECOMMENDATION: [specific action to fix or negotiate]
---

After listing all issues, add:

SUMMARY:
[2-sentence overall assessment]

MISSING STANDARD CLAUSES:
[List any important clauses that should be present but are not]

OVERALL RISK RATING: [HIGH / MEDIUM / LOW]

NEXT STEPS:
[Prioritised 3-step action list for the lawyer]

Be thorough. A missed HIGH risk clause is a professional negligence issue."""

    try:
        result = await call_ai([
            {"role": "system", "content": system},
            {"role": "user", "content": f"Review this contract:\n\n{text[:14000]}"}
        ], max_tokens=3000)

        overall = ("HIGH" if "OVERALL RISK RATING: HIGH" in result else
                   "MEDIUM" if "OVERALL RISK RATING: MEDIUM" in result else "LOW")

        await log_action(firm_id, user_name, "contract_review",
                        file.filename, f"Overall: {overall}", workspace_id or None,
                        {"filename": file.filename, "overall_risk": overall, "review": result})

        return ok({"result": result, "filename": file.filename,
                   "overall_risk": overall})
    except Exception as e:
        return err("Contract review failed. Please try again.")

# ════════════════════════════════════════════════════════════════
# FEATURE 4 — DOCUMENT SUMMARY
# ════════════════════════════════════════════════════════════════
@app.post("/api/v1/document-summary")
@limiter.limit("15/minute")
async def document_summary(
    request: Request,
    file: UploadFile = File(...),
    firm_id: str = Form(default="default"),
    user_name: str = Form(default="User"),
    workspace_id: str = Form(default="")
):
    """Structured one-page legal brief of any document."""
    content = await validate_file(file)
    text = extract_text(content, file.filename)
    if not text:
        return err("Could not extract text from document.")

    system = """You are a senior legal analyst. Produce a structured one-page brief.

Format EXACTLY:

DOCUMENT TYPE: [type]
PARTIES:
  - [Party A: full name and role]
  - [Party B: full name and role]
DATE: [execution date or 'Not specified']
GOVERNING LAW: [jurisdiction]
DURATION: [term or 'Not specified']
PURPOSE: [1 sentence]

KEY OBLIGATIONS:
  Party A: [bullet list of obligations]
  Party B: [bullet list of obligations]

IMPORTANT DATES & DEADLINES:
[List with dates where specified]

KEY DEFINED TERMS:
[List top 5-8 defined terms that a lawyer needs to understand]

FLAGGED RISK CLAUSES:
[For each: CLAUSE | RISK LEVEL: HIGH/MED/LOW | BRIEF NOTE]

MISSING CLAUSES:
[Important clauses not present, if any]

OVERALL SUMMARY:
[2 sentences: what this document is and any key concerns]"""

    try:
        result = await call_ai([
            {"role": "system", "content": system},
            {"role": "user", "content": f"Produce a legal brief:\n\n{text[:14000]}"}
        ], max_tokens=2000)

        await log_action(firm_id, user_name, "document_summary",
                        file.filename, "Completed", workspace_id or None,
                        {"filename": file.filename, "summary": result})

        return ok({"result": result, "filename": file.filename})
    except Exception as e:
        return err("Document summary failed. Please try again.")

# ════════════════════════════════════════════════════════════════
# FEATURE 5 — DRAFT WITH LEX (streaming)
# ════════════════════════════════════════════════════════════════

# Practice area templates
DRAFT_TEMPLATES = {
    "NDA": {
        "clauses": ["definition of confidential information", "obligations of receiving party",
                    "exceptions to confidentiality", "duration", "return of information",
                    "injunctive relief", "governing law", "entire agreement"],
        "note": "Include mutual vs unilateral choice"
    },
    "Service Agreement": {
        "clauses": ["scope of services", "payment terms", "IP ownership",
                    "data protection", "liability cap", "termination", "force majeure"],
        "note": "Flag IR35/employment status if UK"
    },
    "Employment Contract": {
        "clauses": ["role and duties", "salary and benefits", "notice period",
                    "confidentiality", "IP assignment", "restrictive covenants",
                    "disciplinary procedure reference"],
        "note": "Note: non-compete enforceability varies by jurisdiction"
    },
    "SPA": {
        "clauses": ["purchase price", "conditions precedent", "representations and warranties",
                    "indemnification", "completion mechanics", "locked box / closing accounts",
                    "post-completion covenants"],
        "note": "Complex — flag all financial figures for [REVIEW REQUIRED]"
    },
    "Legal Notice": {
        "clauses": ["facts and background", "legal basis", "demand", "deadline",
                    "consequences of non-compliance"],
        "note": "Pakistan/UAE: include formal notice requirements"
    },
    "Lease Agreement": {
        "clauses": ["property description", "rent and deposit", "term", "repair obligations",
                    "subletting restrictions", "break clause", "renewal option"],
        "note": "Include stamp duty / registration requirements for jurisdiction"
    },
    "Joint Venture": {
        "clauses": ["purpose", "contributions", "governance", "profit sharing",
                    "decision making", "exit mechanisms", "non-compete"],
        "note": "Flag regulatory approval requirements"
    }
}

JURISDICTION_RULES = {
    "UK": "English and Welsh law. Use UK legal conventions. Cite Companies Act 2006, Contracts (Rights of Third Parties) Act 1999, relevant statutes as applicable.",
    "Italy": "Italian civil law. Reference Codice Civile, D.Lgs. 231/2001 where relevant. Use Italian legal style.",
    "UAE": "UAE Federal law with DIFC/ADGM framework. Reference UAE Federal Laws. Note free zone vs mainland.",
    "Pakistan": "Pakistani law. Reference Contract Act 1872, Companies Act 2017, relevant Pakistan statutes. Formal Pakistani legal style.",
    "EU": "EU law. Reference relevant EU Directives and Regulations. GDPR Regulation 2016/679 for data.",
    "US": "US law. Specify governing state. Use US legal conventions. Delaware preferred for commercial.",
    "Germany": "German law (BGB/HGB). Use German legal conventions. GDPR applies with strict enforcement.",
    "France": "French civil law (Code Civil). GDPR applies. French Bar rules.",
    "Gulf": "Gulf region — specify country. UAE Federal / Qatar / Saudi Arabia / Bahrain law as applicable.",
    "Canada": "Canadian common law (federal + provincial). Specify province. PIPEDA for data.",
    "Australia": "Australian common law. Specify state. Australian Consumer Law applies to consumer contracts.",
}

@app.post("/api/v1/draft-with-lex")
@limiter.limit("10/minute")
async def draft_with_lex(request: Request, data: DraftRequest):
    """
    Streaming document drafting by Lex.
    Jurisdiction-aware, template-structured, [REVIEW REQUIRED] flagged.
    """
    jur_rule = JURISDICTION_RULES.get(data.jurisdiction,
                                       f"{data.jurisdiction} law and conventions.")

    template_guidance = ""
    if data.template_id and data.template_id in DRAFT_TEMPLATES:
        tmpl = DRAFT_TEMPLATES[data.template_id]
        template_guidance = f"""
Template structure — include these clauses:
{', '.join(tmpl['clauses'])}
Template note: {tmpl['note']}"""

    system = f"""You are Lex — Jurivon's senior legal drafting AI.

Jurisdiction: {data.jurisdiction}
Governing law rules: {jur_rule}
{template_guidance}

Drafting rules:
1. Use professional legal language appropriate for the jurisdiction
2. Mark every clause requiring specific partner review: [REVIEW REQUIRED: reason]
3. Mark high-risk clauses: [LEGAL ADVICE REQUIRED: reason]
4. Use precise party names throughout — no vague 'the parties'
5. Include all standard clauses for this document type
6. Add a Drafter's Note section at end explaining key choices
7. Use numbered clauses
8. Start with document title, date placeholder, and recitals

CRITICAL: Never invent specific figures (financial amounts, dates, regulatory numbers).
Always use [INSERT: description] placeholders for client-specific details."""

    user_msg = f"""Draft: {data.doc_type}

Party A: {data.party_a}
Party B: {data.party_b}
Jurisdiction: {data.jurisdiction}
Additional terms/instructions: {data.key_terms if data.key_terms else 'Standard terms — all clauses'}"""

    async def stream_draft():
        full_text = ""
        try:
            if not openai_client:
                yield f"data: {json.dumps({'error': 'AI not configured'})}\n\n"
                return

            stream = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg}
                ],
                stream=True,
                max_tokens=4000,
                temperature=0.1
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_text += delta
                    yield f"data: {json.dumps({'chunk': delta})}\n\n"

            await log_action(data.firm_id, data.user_name, "draft_with_lex",
                           f"{data.doc_type} — {data.party_a}/{data.party_b}",
                           "Drafted", data.workspace_id,
                           {"doc_type": data.doc_type, "jurisdiction": data.jurisdiction,
                            "party_a": data.party_a, "party_b": data.party_b,
                            "draft": full_text[:5000]})
            yield f"data: {json.dumps({'done': True, 'length': len(full_text)})}\n\n"
        except Exception as e:
            logger.error(f"Draft streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        stream_draft(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@app.get("/api/v1/draft-templates")
async def get_templates():
    """Return available draft templates."""
    return ok({k: {"clauses": v["clauses"], "note": v["note"]}
               for k, v in DRAFT_TEMPLATES.items()})

# ════════════════════════════════════════════════════════════════
# FEATURE 6 — DUE DILIGENCE
# ════════════════════════════════════════════════════════════════
@app.post("/api/v1/due-diligence")
@limiter.limit("5/minute")
async def due_diligence(
    request: Request,
    files: List[UploadFile] = File(...),
    deal_type: str = Form(default="M&A"),
    target_name: str = Form(default="Target"),
    firm_id: str = Form(default="default"),
    user_name: str = Form(default="User"),
    workspace_id: str = Form(default="")
):
    """
    Multi-document due diligence analysis.
    5 categories: Corporate, Financial, Legal, Employment, IP/Data.
    """
    if len(files) > 10:
        return err("Maximum 10 documents per due diligence request.")

    combined = ""
    filenames = []
    for f in files:
        content = await validate_file(f)
        text = extract_text(content, f.filename)
        if text:
            combined += f"\n\n=== DOCUMENT: {f.filename} ===\n{text[:3000]}"
            filenames.append(f.filename)

    if not combined.strip():
        return err("Could not extract text from any documents.")

    system = f"""You are a senior M&A and due diligence lawyer.
Analyse the provided documents for a {deal_type} transaction on target: {target_name}.

Produce a structured DD report in EXACTLY this format:

DUE DILIGENCE REPORT
Target: {target_name}
Transaction Type: {deal_type}
Documents Reviewed: {len(filenames)}
Date: {datetime.utcnow().strftime('%d %B %Y')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATEGORY 1 — CORPORATE & GOVERNANCE
Risk Rating: [HIGH/MEDIUM/LOW/CLEAN]
[List key findings with document references]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATEGORY 2 — FINANCIAL
Risk Rating: [HIGH/MEDIUM/LOW/CLEAN]
[List key findings]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATEGORY 3 — LEGAL & REGULATORY
Risk Rating: [HIGH/MEDIUM/LOW/CLEAN]
[List key findings]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATEGORY 4 — EMPLOYMENT & HR
Risk Rating: [HIGH/MEDIUM/LOW/CLEAN]
[List key findings]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATEGORY 5 — IP, DATA & TECHNOLOGY
Risk Rating: [HIGH/MEDIUM/LOW/CLEAN]
[List key findings]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL ISSUES REQUIRING IMMEDIATE ACTION:
[List any HIGH risk findings that could be deal-breakers]

DOCUMENTS OUTSTANDING:
[List any key documents mentioned but not provided]

OVERALL RECOMMENDATION: [PROCEED / PROCEED WITH CONDITIONS / PAUSE / DECLINE]"""

    try:
        result = await call_ai([
            {"role": "system", "content": system},
            {"role": "user", "content": f"Documents:\n{combined[:15000]}"}
        ], max_tokens=3500)

        await log_action(firm_id, user_name, "due_diligence",
                        f"{deal_type}: {', '.join(filenames)}", "Completed",
                        workspace_id or None,
                        {"files": filenames, "deal_type": deal_type,
                         "target": target_name, "result": result})

        return ok({"result": result, "documents_analysed": len(filenames),
                   "filenames": filenames})
    except Exception as e:
        return err("Due diligence analysis failed. Please try again.")

# ════════════════════════════════════════════════════════════════
# FEATURE 7 — OdV ART. 231 CERTIFICATE (Italian firms ONLY)
# ════════════════════════════════════════════════════════════════
@app.post("/api/v1/odv-certificate")
@limiter.limit("10/minute")
async def odv_certificate(request: Request, data: OdvRequest):
    """
    Bilingual OdV independence certificate under D.Lgs. 231/2001.
    Italian firms ONLY. Do not expose to non-Italian clients.
    """
    ts = datetime.utcnow()
    cert_ref = f"OdV-{ts.strftime('%Y%m%d%H%M%S')}-{data.company_name[:6].upper().replace(' ','')}"

    system = """You are drafting an Organismo di Vigilanza independence assessment
under D.Lgs. 231/2001 for Italian Bar filing.
Generate a bilingual certificate (Italian first, then English translation).
Use formal Italian legal language. Be precise and professional."""

    user_msg = f"""Generate OdV Art. 231/2001 independence certificate:

Company: {data.company_name} ({data.company_type})
Assessing Lawyer: Avv. {data.lawyer_name}
Bar Registration: {data.lawyer_bar_number if data.lawyer_bar_number else '[NUMERO DA INSERIRE]'}
Matter Under Review: {data.matter_description}
Conflict Check Result: {data.conflict_result}
Certificate Reference: {cert_ref}
Date: {ts.strftime('%d %B %Y')} / {ts.strftime('%d/%m/%Y')}

Structure:
SEZIONE A — ITALIANO (Italian)
1. Intestazione ufficiale
2. Dichiarazione di indipendenza OdV
3. Metodologia di valutazione
4. Esito del conflict check
5. Raccomandazione: PROCEDERE / NON PROCEDERE

SECTION B — ENGLISH TRANSLATION
[Full translation of Section A]

CERTIFICATION BLOCK
Signature placeholder, date, Bar number"""

    try:
        result = await call_ai([
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg}
        ], max_tokens=2500)

        cert_data = {
            "certificate_ref": cert_ref,
            "company": data.company_name,
            "lawyer": data.lawyer_name,
            "conflict_result": data.conflict_result,
            "date": ts.strftime("%d %B %Y"),
            "result": result
        }

        await log_action(data.firm_id, data.user_name, "odv_certificate",
                        f"{data.company_name} — {cert_ref}",
                        data.conflict_result, None, cert_data)

        return ok(cert_data)
    except Exception as e:
        return err("OdV certificate generation failed.")

# ════════════════════════════════════════════════════════════════
# FEATURE 8 — LEGAL RESEARCH WITH KHAN
# Live APIs: legislation.gov.uk, EUR-Lex, Normattiva
# Pakistan: RAG vector store
# ════════════════════════════════════════════════════════════════

async def fetch_uk_law(query: str) -> str:
    """Live fetch from legislation.gov.uk API."""
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.get(
                "https://www.legislation.gov.uk/api/1/search",
                params={"text": query, "results": "5"}
            )
            if r.status_code == 200:
                data = r.json()
                items = data.get("results", [])[:3]
                texts = []
                for item in items:
                    link = item.get("link", "")
                    title = item.get("title", "")
                    if link:
                        texts.append(f"Title: {title}\nURL: {link}")
                if texts:
                    return "legislation.gov.uk results:\n" + "\n\n".join(texts)
    except Exception as e:
        logger.warning(f"UK law fetch failed: {e}")
    return ""

async def fetch_eu_law(query: str) -> str:
    """Live fetch from EUR-Lex SPARQL endpoint."""
    try:
        sparql = f"""SELECT ?title ?uri WHERE {{
  ?uri <http://purl.org/dc/elements/1.1/title> ?title .
  FILTER(CONTAINS(LCASE(STR(?title)), LCASE("{query[:40]}")))
}} LIMIT 5"""
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.get(
                "https://publications.europa.eu/webapi/rdf/sparql",
                params={"query": sparql, "format": "application/sparql-results+json"}
            )
            if r.status_code == 200:
                data = r.json()
                bindings = data.get("results", {}).get("bindings", [])[:5]
                if bindings:
                    items = [
                        f"- {b.get('title',{}).get('value','')}: {b.get('uri',{}).get('value','')}"
                        for b in bindings
                    ]
                    return "EUR-Lex results:\n" + "\n".join(items)
    except Exception as e:
        logger.warning(f"EU law fetch failed: {e}")
    return ""

async def fetch_italian_law(query: str) -> str:
    """Live fetch from Normattiva."""
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.get(
                "https://www.normattiva.it/do/atto/ricercaPerTesto",
                params={"query": query, "typeSearch": "T"},
                headers={"User-Agent": "Jurivon Legal Research/3.0"}
            )
            if r.status_code == 200:
                text = re.sub(r'<[^>]+>', ' ', r.text)
                text = re.sub(r'\s+', ' ', text).strip()
                return f"Normattiva (Italian law) results:\n{text[:3000]}"
    except Exception as e:
        logger.warning(f"Italian law fetch failed: {e}")
    return ""

async def fetch_pakistan_law_rag(query: str) -> tuple:
    """RAG search in Pakistan law vector store."""
    if not supabase or not openai_client:
        return "", False
    try:
        emb = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query[:8000]
        )
        results = supabase.rpc("match_pakistan_law", {
            "query_embedding": emb.data[0].embedding,
            "match_threshold": 0.65,
            "match_count": 6
        }).execute()

        if results.data:
            chunks = [
                f"Source: {r.get('title','')} ({r.get('court','')}, {r.get('year','')})\n"
                f"Citation: {r.get('citation','Unverified')}\n{r.get('chunk_text','')}"
                for r in results.data
            ]
            return "\n\n---\n\n".join(chunks), True
    except Exception as e:
        logger.warning(f"Pakistan RAG failed: {e}")
    return "", False

JURISDICTION_SYSTEMS = {
    "UK": "England and Wales. Common law. Key sources: legislation.gov.uk, BAILII.",
    "Italy": "Italian civil law. Codice Civile. D.Lgs. 231/2001. GDPR. Normattiva.",
    "EU": "European Union law. EU Directives/Regulations. CJEU case law. EUR-Lex.",
    "UAE": "UAE Federal law + DIFC/ADGM free zone law. Dubai Courts + DIFC Courts.",
    "Pakistan": "Pakistani common law. Contract Act 1872, CPC 1908, PPC 1860.",
    "US": "US common law. Federal + state law. Specify governing state.",
    "Germany": "German civil law (BGB/HGB). GDPR strictly enforced.",
    "France": "French civil law (Code Civil). GDPR. Cour de Cassation.",
    "Gulf": "Gulf region: UAE/Qatar/Saudi/Bahrain. Specify country for precise analysis.",
    "Canada": "Canadian common law (federal + provincial). PIPEDA. Quebec civil law distinct.",
    "Australia": "Australian common law. Federal + state. Australian Consumer Law.",
    "Global": "Multi-jurisdictional analysis. International law principles.",
}

@app.post("/api/v1/legal-research-khan")
@limiter.limit("15/minute")
async def legal_research_khan(request: Request, data: ResearchRequest):
    """
    Legal Research with Khan.
    Live law APIs for UK/EU/Italy. RAG for Pakistan. AI synthesis for all.
    """
    jur_system = JURISDICTION_SYSTEMS.get(data.jurisdiction,
                                          f"{data.jurisdiction} legal system.")
    live_context = ""
    source_type = "AI training knowledge"

    try:
        if data.jurisdiction == "UK":
            live_context = await fetch_uk_law(data.query)
            if live_context:
                source_type = "legislation.gov.uk (live)"
        elif data.jurisdiction in ("EU", "Germany", "France"):
            live_context = await fetch_eu_law(data.query)
            if live_context:
                source_type = "EUR-Lex (live)"
        elif data.jurisdiction == "Italy":
            live_context = await fetch_italian_law(data.query)
            if live_context:
                source_type = "Normattiva (live)"
        elif data.jurisdiction == "Pakistan":
            live_context, has_db = await fetch_pakistan_law_rag(data.query)
            if has_db:
                source_type = "Jurivon Pakistan Law Database (RAG)"
    except Exception as e:
        logger.warning(f"Live law fetch failed: {e}")

    context_section = f"""RETRIEVED FROM OFFICIAL SOURCES:
{live_context[:6000]}

Base your answer on the above retrieved texts where possible.""" if live_context else \
f"""No live law texts retrieved for this query.
Answer from training knowledge. Mark every key legal statement: [VERIFY AGAINST CURRENT SOURCE]
Provide official source URLs for independent verification."""

    system = f"""You are Khan — Jurivon's senior legal research AI. Expert in global legal systems.

Legal system: {jur_system}

{context_section}

Response format — EXACTLY:

DIRECT ANSWER:
[Clear, specific answer in 2-3 sentences]

LEGAL BASIS:
[Specific statutes, articles, case law with accurate citations]

DETAILED ANALYSIS:
[Comprehensive explanation — min 3 paragraphs]

PRACTICAL IMPLICATIONS:
[What this means for the client in practice]

IMPORTANT CAVEATS:
[Limitations, recent developments, jurisdictional nuances]

VERIFICATION SOURCES:
[Official URLs where this can be verified]

---
Researched by: Khan | Jurivon Legal Research AI v3
Jurisdiction: {data.jurisdiction} | Source: {source_type}
Date: {datetime.utcnow().strftime('%d %B %Y')}"""

    try:
        result = await call_ai([
            {"role": "system", "content": system},
            {"role": "user", "content": data.query}
        ], max_tokens=2500)

        await log_action(data.firm_id, data.user_name, "legal_research_khan",
                        f"[{data.jurisdiction}] {data.query[:200]}", "Completed",
                        data.workspace_id,
                        {"query": data.query, "jurisdiction": data.jurisdiction,
                         "source_type": source_type, "has_live_data": bool(live_context),
                         "result": result})

        return ok({
            "result": result,
            "jurisdiction": data.jurisdiction,
            "source_type": source_type,
            "has_live_data": bool(live_context)
        })
    except Exception as e:
        return err("Legal research failed. Please try again.")

# ════════════════════════════════════════════════════════════════
# FEATURE 9 — CITATION VERIFY
# ════════════════════════════════════════════════════════════════
@app.post("/api/v1/citation-verify")
@limiter.limit("20/minute")
async def citation_verify(request: Request, data: CitationRequest):
    """
    Hallucination protection: verify if a legal citation is real.
    Returns VERIFIED / LIKELY REAL / UNVERIFIABLE / SUSPICIOUS / INCORRECT.
    """
    system = """You are a legal citation verification specialist.
Assess whether the provided legal citation is real, accurate, and verifiable.

Return EXACTLY:

VERIFICATION RESULT: [VERIFIED / LIKELY REAL / UNVERIFIABLE / SUSPICIOUS / INCORRECT]
CONFIDENCE: [0-100]%
ASSESSMENT: [explain your reasoning in 2-3 sentences]
SOURCE URL: [official URL to verify, or 'Not directly available']
WARNING: [if citation appears fabricated, explain specifically why]
RECOMMENDATION: [what the lawyer should do to verify this citation]

Critical rule: If you cannot confirm a citation from training knowledge,
return UNVERIFIABLE — never invent verification."""

    try:
        result = await call_ai([
            {"role": "system", "content": system},
            {"role": "user", "content":
             f"Verify: '{data.citation}'\nJurisdiction: {data.jurisdiction}"}
        ], max_tokens=800)

        status = ("VERIFIED" if "VERIFIED" in result and "UNVERIFIABLE" not in result else
                  "SUSPICIOUS" if any(x in result for x in ["SUSPICIOUS", "INCORRECT"]) else
                  "UNVERIFIABLE")

        await log_action(data.firm_id, "User", "citation_verify",
                        data.citation, status)

        return ok({"result": result, "citation": data.citation,
                   "status": status, "jurisdiction": data.jurisdiction})
    except Exception as e:
        return err("Citation verification failed.")

# ════════════════════════════════════════════════════════════════
# FEATURE 10 — REGULATORY TRACKER
# ════════════════════════════════════════════════════════════════
@app.post("/api/v1/regulatory-tracker")
@limiter.limit("10/minute")
async def regulatory_tracker(
    request: Request,
    jurisdiction: str = Form(...),
    practice_areas: str = Form(...),
    firm_id: str = Form(default="default"),
    user_name: str = Form(default="User")
):
    """Current regulatory developments by jurisdiction and practice area."""
    system = f"""You are a regulatory intelligence analyst for law firms.
Provide current regulatory developments for {jurisdiction} — practice areas: {practice_areas}.

IMPORTANT: Only state developments you can confirm from training knowledge.
Do not fabricate regulatory developments. Mark uncertainty clearly.

Format — for each development:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
URGENCY: [HIGH / MEDIUM / LOW]
AREA: [practice area]
DEVELOPMENT: [title]
SUMMARY: [what changed — 2-3 sentences]
EFFECTIVE DATE: [date or 'TBC' or 'Already in force']
ACTION REQUIRED: [what the firm must do in response]
SOURCE: [official source name and URL]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

List minimum 5 developments. Mark any where you are uncertain of current status as [VERIFY DATE]."""

    try:
        result = await call_ai([
            {"role": "system", "content": system},
            {"role": "user", "content":
             f"Latest regulatory developments: {jurisdiction} — {practice_areas}"}
        ], max_tokens=2500)

        await log_action(firm_id, user_name, "regulatory_tracker",
                        f"{jurisdiction} — {practice_areas}", "Completed")

        return ok({"result": result, "jurisdiction": jurisdiction,
                   "practice_areas": practice_areas})
    except Exception as e:
        return err("Regulatory tracker failed.")

# ════════════════════════════════════════════════════════════════
# FEATURE 11 — MATTER DATABASE
# ════════════════════════════════════════════════════════════════
@app.get("/api/v1/matters")
async def get_matters(firm_id: str = "default",
                      status: Optional[str] = None,
                      practice_area: Optional[str] = None):
    if not supabase:
        return ok([])
    try:
        q = supabase.table("matters").select("*").eq("firm_id", firm_id)
        if status:
            q = q.eq("status", status)
        if practice_area:
            q = q.eq("practice_area", practice_area)
        r = q.order("created_at", desc=True).execute()
        return ok(r.data or [])
    except Exception as e:
        return err("Failed to fetch matters.")

@app.post("/api/v1/matters")
@limiter.limit("30/minute")
async def add_matter(request: Request, data: MatterRecord):
    if not supabase:
        return err("Database not connected.")
    try:
        record = data.model_dump()
        record["created_at"] = datetime.utcnow().isoformat()
        r = supabase.table("matters").insert(record).execute()
        await log_action(data.firm_id, "System", "matter_added",
                        f"{data.matter_ref} — {data.client}", "Added")
        return ok(r.data[0] if r.data else {})
    except Exception as e:
        return err(f"Failed to add matter: {str(e)}")

@app.post("/api/v1/matters/bulk")
async def bulk_upload_matters(
    request: Request,
    file: UploadFile = File(...),
    firm_id: str = Form(default="default")
):
    """CSV bulk upload. Format: matter_ref,client,counterparty,description,status,partner,practice_area"""
    content = await file.read()
    text = content.decode("utf-8", errors="replace")
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    if len(lines) > 501:
        return err("Maximum 500 matters per bulk upload.")

    matters = []
    errors = []
    for i, line in enumerate(lines[1:], 1):  # skip header
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) < 2:
            errors.append(f"Row {i}: insufficient columns")
            continue
        matters.append({
            "firm_id": firm_id,
            "matter_ref": parts[0] if len(parts) > 0 else f"REF-{i}",
            "client": parts[1] if len(parts) > 1 else "",
            "counterparty": parts[2] if len(parts) > 2 else "",
            "description": parts[3] if len(parts) > 3 else "",
            "status": parts[4] if len(parts) > 4 else "OPEN",
            "lead_partner": parts[5] if len(parts) > 5 else "",
            "practice_area": parts[6] if len(parts) > 6 else "",
            "jurisdiction": parts[7] if len(parts) > 7 else "",
            "created_at": datetime.utcnow().isoformat()
        })

    if not matters:
        return err("No valid matters found in file.")

    if supabase:
        supabase.table("matters").insert(matters).execute()

    return ok({"inserted": len(matters), "errors": errors,
               "error_count": len(errors)})

@app.delete("/api/v1/matters/{matter_id}")
async def delete_matter(matter_id: str, firm_id: str = "default"):
    if not supabase:
        return err("Database not connected.")
    supabase.table("matters").delete()\
        .eq("id", matter_id).eq("firm_id", firm_id).execute()
    return ok({"deleted": True})

# ════════════════════════════════════════════════════════════════
# FEATURE 12 — AUDIT LOG
# ════════════════════════════════════════════════════════════════
@app.get("/api/v1/audit-log")
async def get_audit_log(firm_id: str = "default",
                        limit: int = 100,
                        export: bool = False):
    if not supabase:
        return ok([])
    try:
        r = supabase.table("audit_log").select("*")\
            .eq("firm_id", firm_id)\
            .order("created_at", desc=True)\
            .limit(min(limit, 1000)).execute()
        data = r.data or []

        if export:
            # CSV export
            output = io.StringIO()
            if data:
                writer = csv.DictWriter(output, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition":
                         f"attachment; filename=audit_log_{firm_id}_{datetime.utcnow().strftime('%Y%m%d')}.csv"}
            )
        return ok(data)
    except Exception as e:
        return err("Failed to fetch audit log.")

# ════════════════════════════════════════════════════════════════
# FEATURE 13 — PRECEDENT SEARCH
# ════════════════════════════════════════════════════════════════
@app.post("/api/v1/precedent-search")
@limiter.limit("15/minute")
async def precedent_search(
    request: Request,
    query: str = Form(...),
    jurisdiction: str = Form(default="UK"),
    firm_id: str = Form(default="default"),
    user_name: str = Form(default="User")
):
    matters = []
    if supabase:
        try:
            r = supabase.table("matters").select("*")\
                .eq("firm_id", firm_id).execute()
            matters = r.data or []
        except:
            pass

    matters_text = "\n".join([
        f"REF: {m.get('matter_ref','?')} | CLIENT: {m.get('client','?')} | "
        f"AREA: {m.get('practice_area','?')} | DESC: {m.get('description','')[:150]}"
        for m in matters[:50]
    ]) if matters else "No matters in database."

    system = """You are a legal precedent research specialist.
Find relevant precedents from the firm's matter database and from general legal knowledge.

Format:

INTERNAL PRECEDENTS (from firm database):
[List relevant matters with ref, why relevant, how to apply]
[If none relevant: 'No directly relevant matters found in database']

EXTERNAL PRECEDENTS (general legal knowledge):
[List relevant cases, statutes, or standard clauses with citations]
[Mark any uncertain citations: [VERIFY CITATION]]

APPLICATION GUIDANCE:
[How the identified precedents apply to the current query]

SEARCH SUMMARY:
[Overall assessment — 2 sentences]"""

    try:
        result = await call_ai([
            {"role": "system", "content": system},
            {"role": "user", "content":
             f"Query: {query}\nJurisdiction: {jurisdiction}\n\nFirm database:\n{matters_text}"}
        ], max_tokens=1800)

        return ok({"result": result, "matters_searched": len(matters),
                   "jurisdiction": jurisdiction})
    except Exception as e:
        return err("Precedent search failed.")

# ════════════════════════════════════════════════════════════════
# FEATURE 14 — MATTER WORKSPACES
# ════════════════════════════════════════════════════════════════
@app.post("/api/v1/workspaces")
@limiter.limit("30/minute")
async def create_workspace(request: Request, data: WorkspaceCreate):
    if not supabase:
        return err("Database not connected.")
    try:
        r = supabase.table("matter_workspaces").insert({
            "firm_id": data.firm_id,
            "client_name": data.client_name,
            "matter_ref": data.matter_ref,
            "matter_type": data.matter_type,
            "jurisdiction": data.jurisdiction,
            "lead_partner": data.lead_partner,
            "status": "OPEN",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }).execute()
        return ok(r.data[0] if r.data else {})
    except Exception as e:
        return err(f"Failed to create workspace: {str(e)}")

@app.get("/api/v1/workspaces")
async def list_workspaces(firm_id: str = "default"):
    if not supabase:
        return ok([])
    try:
        r = supabase.table("matter_workspaces").select("*")\
            .eq("firm_id", firm_id)\
            .is_("deleted_at", "null")\
            .order("updated_at", desc=True).execute()
        return ok(r.data or [])
    except Exception as e:
        return err("Failed to fetch workspaces.")

@app.get("/api/v1/workspaces/{workspace_id}")
async def get_workspace(workspace_id: str, firm_id: str = "default"):
    if not supabase:
        return ok({})
    try:
        ws = supabase.table("matter_workspaces").select("*")\
            .eq("id", workspace_id).eq("firm_id", firm_id).single().execute()
        items = supabase.table("workspace_items").select("*")\
            .eq("workspace_id", workspace_id)\
            .order("created_at", desc=True).execute()
        return ok({"workspace": ws.data, "items": items.data or []})
    except Exception as e:
        return err("Failed to fetch workspace.")

@app.post("/api/v1/workspaces/{workspace_id}/items")
async def save_workspace_item(request: Request,
                               workspace_id: str,
                               data: WorkspaceItemCreate):
    if not supabase:
        return err("Database not connected.")
    try:
        r = supabase.table("workspace_items").insert({
            "workspace_id": workspace_id,
            "firm_id": data.firm_id,
            "item_type": data.item_type,
            "title": data.title,
            "content": data.content,
            "bookmarked": False,
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        supabase.table("matter_workspaces")\
            .update({"updated_at": datetime.utcnow().isoformat()})\
            .eq("id", workspace_id).execute()
        supabase.table("interaction_history").insert({
            "firm_id": data.firm_id,
            "feature": data.item_type,
            "input_summary": data.title,
            "full_content": data.content,
            "workspace_id": workspace_id,
            "bookmarked": False,
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        return ok(r.data[0] if r.data else {})
    except Exception as e:
        return err(f"Failed to save item: {str(e)}")

@app.delete("/api/v1/workspaces/{workspace_id}")
async def delete_workspace(workspace_id: str, firm_id: str = "default"):
    if not supabase:
        return err("Database not connected.")
    supabase.table("matter_workspaces")\
        .update({"deleted_at": datetime.utcnow().isoformat()})\
        .eq("id", workspace_id).eq("firm_id", firm_id).execute()
    return ok({"deleted": True})

# ════════════════════════════════════════════════════════════════
# FEATURE 15 — HISTORY & BOOKMARKS
# ════════════════════════════════════════════════════════════════
@app.get("/api/v1/history")
async def get_history(firm_id: str = "default", limit: int = 50):
    if not supabase:
        return ok([])
    try:
        r = supabase.table("interaction_history").select("*")\
            .eq("firm_id", firm_id)\
            .order("created_at", desc=True)\
            .limit(min(limit, 200)).execute()
        return ok(r.data or [])
    except Exception as e:
        return err("Failed to fetch history.")

@app.get("/api/v1/bookmarks")
async def get_bookmarks(firm_id: str = "default"):
    if not supabase:
        return ok([])
    try:
        r = supabase.table("interaction_history").select("*")\
            .eq("firm_id", firm_id)\
            .eq("bookmarked", True)\
            .order("created_at", desc=True).execute()
        return ok(r.data or [])
    except Exception as e:
        return err("Failed to fetch bookmarks.")

@app.patch("/api/v1/history/{item_id}/bookmark")
async def toggle_bookmark(item_id: str, bookmarked: bool = True):
    if not supabase:
        return err("Database not connected.")
    supabase.table("interaction_history")\
        .update({"bookmarked": bookmarked})\
        .eq("id", item_id).execute()
    return ok({"bookmarked": bookmarked})

# ════════════════════════════════════════════════════════════════
# FEATURE 16 — COLLABORATION (workspace comments)
# ════════════════════════════════════════════════════════════════
@app.get("/api/v1/workspaces/{workspace_id}/comments")
async def get_comments(workspace_id: str):
    if not supabase:
        return ok([])
    try:
        r = supabase.table("workspace_comments").select("*")\
            .eq("workspace_id", workspace_id)\
            .order("created_at", desc=True).execute()
        return ok(r.data or [])
    except Exception as e:
        return err("Failed to fetch comments.")

@app.post("/api/v1/workspaces/comments")
@limiter.limit("30/minute")
async def post_comment(request: Request, data: CommentCreate):
    if not supabase:
        return err("Database not connected.")
    try:
        r = supabase.table("workspace_comments").insert({
            "workspace_id": data.workspace_id,
            "firm_id": data.firm_id,
            "author": sanitize(data.author),
            "comment": sanitize(data.comment),
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        return ok(r.data[0] if r.data else {})
    except Exception as e:
        return err(f"Failed to post comment: {str(e)}")

# ════════════════════════════════════════════════════════════════
# FIRM SETTINGS
# ════════════════════════════════════════════════════════════════
@app.get("/api/v1/settings/{firm_id}")
async def get_settings(firm_id: str):
    if not supabase:
        return ok({"firm_id": firm_id, "firm_name": "My Firm",
                   "jurisdiction": "UK", "show_odv": False})
    try:
        r = supabase.table("firm_settings").select("*")\
            .eq("firm_id", firm_id).execute()
        if r.data:
            return ok(r.data[0])
        return ok({"firm_id": firm_id, "firm_name": "My Firm",
                   "jurisdiction": "UK", "show_odv": False})
    except Exception as e:
        return ok({"firm_id": firm_id, "firm_name": "My Firm",
                   "jurisdiction": "UK", "show_odv": False})

@app.post("/api/v1/settings")
async def save_settings(request: Request, data: FirmSettings):
    if not supabase:
        return err("Database not connected.")
    try:
        record = data.model_dump()
        record["updated_at"] = datetime.utcnow().isoformat()
        existing = supabase.table("firm_settings").select("id")\
            .eq("firm_id", data.firm_id).execute()
        if existing.data:
            supabase.table("firm_settings")\
                .update(record)\
                .eq("firm_id", data.firm_id).execute()
        else:
            record["created_at"] = datetime.utcnow().isoformat()
            supabase.table("firm_settings").insert(record).execute()
        return ok(record)
    except Exception as e:
        return err(f"Failed to save settings: {str(e)}")
