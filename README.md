# Enso Group HR Policy Assistant (Grounded RAG)

An enterprise Retrieval-Augmented Generation (RAG) assistant that answers employee questions strictly using the **Enso Group HR Policy Manual PDF** (79 pages). Every policy answer is backed by verifiable page and section citations with expandable verbatim excerpts. When the manual does not contain sufficient supporting evidence, the assistant deterministically returns **`Not found`**.

---

## LMS Answers (Verified Against the 79-Page Enso Group HR Policy Manual)

| # | Evaluation Question | Verified Answer | Source Document & Citation | Verbatim Supporting Excerpt |
| :- | :--- | :--- | :--- | :--- |
| **1** | **How many days of Privilege Leave (PL) does a permanent employee get per year?** | **12 days** per calendar year (1st January – 31st December) | `hr_manual.pdf` · **Page 60** · Section: `[G] Leave Policy` | *"Privilege Leave (PL) – 12 days... All permanent employees are eligible to avail PL. PL is calculated for a period of one calendar year (1st Jan-31st Dec). 12 days PL is allowed to each employee."* |
| **2** | **What is the probation period for employees up to Senior Manager?** | **3 months** from the date of joining | `hr_manual.pdf` · **Page 42** · Section: `[B] Probation Policy` | *"All employees up to Sr. Manager would be placed in Probation for a period of 3 months from the date of their joining. All employees above Sr. Manager will be treated as confirmed employees from the date of joining."* |
| **3** | **What is the company's health-insurance premium amount?** | **`Not found`** | None (Not stated in `hr_manual.pdf`) | *The manual mentions `Medical Reimbursement` and `ESIC` as salary components of CTC on Page 57 (`[D] Compensation Policy`) and `Travel insurance coverage` on Page 71, but does not state a health-insurance premium amount.* |
| **4** | **What is the current GST rate in India?** | **`Not found`** | None (External fact absent from `hr_manual.pdf`) | *The HR Policy Manual does not mention GST rates. Per the manual-only RAG grounding contract, external knowledge is never used.* |
| **5** | **How many days of maternity leave are given?** | **90 days (12 weeks)** with pay (of which not more than 45 days / 6 weeks should precede the expected date of delivery; and 4 weeks with pay in case of miscarriage or medical termination of pregnancy) | `hr_manual.pdf` · **Page 62** · Section: `[G] Leave Policy — Maternity Leave (Leeway)` | *"The company will provide a total of 12 weeks (90 days) as ML with pay, of which not more than 6 weeks (45 days) should precede the expected date of delivery."* |

---

## 1. Problem Statement

Employees need fast, trustworthy answers to HR policy questions (leave entitlements, probation rules, relocation eligibility, staff advances, travel reimbursement) without reading a 79-page manual cover to cover. Generic LLM assistants risk hallucinating numbers, inventing policy rules, or answering from general internet knowledge.

This project solves that problem with a **strictly grounded RAG pipeline** over the Enso Group HR Policy Manual (`data/hr_manual.pdf`) that:
1. Extracts and cleans text page by page while preserving human-readable 1-indexed page numbers and section headings.
2. Persists chunk embeddings and metadata in ChromaDB (`./data/chroma`).
3. Passes only retrieved policy excerpts to the language model (`gemini-3.8-flash`).
4. Validates every citation against actual retrieved chunk IDs and returns **`Not found`** whenever evidence is missing, insufficient, or contradictory.

---

## 2. Features

- **Page-Accurate PDF Ingestion**: Loads the 79-page `hr_manual.pdf`, normalizes hyphenated line wraps from PDF column justification (e.g., `calculat - ed` → `calculated`, `12 weeks( - 90days)` → `12 weeks (90 days)`), and preserves 1-indexed page numbers and policy section titles (`[B] Probation Policy`, `[G] Leave Policy`, etc.).
- **Persistent ChromaDB Vector Store**: Stores 272 chunks and embeddings in `./data/chroma` with idempotent re-indexing that prevents duplicate records across restarts.
- **Hybrid Semantic + Lexical Retrieval**: Combines embedding similarity with policy keyword/phrase verification and a configurable `RELEVANCE_THRESHOLD` (`0.32`).
- **Strict Answer & Citation Contract**: Returns structured JSON (`{ answer, found, sources, message }`) where every citation includes `source`, `page`, `section`, and verbatim `excerpt`.
- **Deterministic `Not found` Abstention**: Never guesses unstated policy figures (e.g., health-insurance premium amount) or external facts (e.g., India's GST rate).
- **Enterprise Web Interface**:
  - **Policy Search**: Interactive question input (`Enter` to submit, `Esc` to clear), clickable LMS evaluation questions, verified answer display, and expandable verbatim source excerpts.
  - **LMS Evaluation Suite**: One-click batch runner for all 5 evaluation questions with copyable LMS submission summary.
  - **Manual Index Explorer**: Filter and inspect all 272 indexed chunks across pages 1–79.
  - **Safe Ingestion & Upload Controls**: Re-index `hr_manual.pdf` (with confirmation step) or upload a replacement `.pdf` (`<= 25 MB`).

---

## 3. Architecture & RAG Flow

```text
INGESTION PIPELINE
HR Policy PDF (data/hr_manual.pdf, 79 pages)
    │
    ├──▶ 1. Extract text page by page (1-indexed page numbers preserved)
    ├──▶ 2. Clean extraction noise & repair PDF column hyphenation
    ├──▶ 3. Detect policy section headings ([A]..[I], SECTION I..IV)
    ├──▶ 4. Recursive chunk splitting (chunk_size=900, chunk_overlap=150)
    ├──▶ 5. Generate embeddings (gemini-embedding-2-preview / deterministic fallback)
    └──▶ 6. Persist chunks + vectors + metadata in ChromaDB (./data/chroma)

QUERY & GROUNDING PIPELINE
Employee Question
    │
    ├──▶ 1. Embed query using the same embedding model
    ├──▶ 2. Hybrid retrieval (Vector similarity + Lexical keyword verification)
    ├──▶ 3. Check relevance threshold & critical concept presence
    │        └── If insufficient evidence ──▶ Return exact "Not found"
    ├──▶ 4. Grounded LLM generation (gemini-3.8-flash) using ONLY retrieved excerpts
    ├──▶ 5. Citation Validator checks cited_chunk_ids against retrieved chunks
    │        └── If malformed JSON or invalid citation ──▶ Return exact "Not found"
    └──▶ 6. Render Answer + Source Filename + Page Number + Section + Excerpt
```

---

## 4. Technology Stack

- **Python RAG Package**: Python 3.10+, LangChain (`langchain`, `langchain-community`, `langchain-text-splitters`, `langchain-google-genai`), ChromaDB (`chromadb`), `pypdf`, `pydantic`, `pytest` / `unittest`.
- **Models**: Google Gemini (`gemini-3.8-flash` for grounded answer generation, `gemini-embedding-2-preview` for embeddings).
- **Web Interface & Full-Stack Runtime**: React 19, TypeScript, Tailwind CSS, Express, `pdfjs-dist`, Vitest.

---

## 5. Project Structure

```text
.
├── app.py                          # Python CLI / app entrypoint
├── server.ts                       # Full-stack Express + Vite web server (Port 3000)
├── src/
│   ├── hr_assistant/               # Modular Python RAG package
│   │   ├── __init__.py
│   │   ├── config.py               # Environment & model configuration
│   │   ├── schemas.py              # Strict AnswerResponse & CitationSource schemas
│   │   ├── ingestion.py            # PDF loading, cleaning, section detection, chunking
│   │   ├── embeddings.py           # Embedding model interface
│   │   ├── vector_store.py         # Persistent ChromaDB vector store
│   │   ├── retrieval.py            # Hybrid retrieval & relevance filtering
│   │   └── answer_service.py       # Grounded generation & citation validation
│   ├── server/
│   │   ├── ragEngine.ts            # Express/Node RAG engine synchronized with ./data/chroma
│   │   └── __tests__/
│   │       └── ragEngine.test.ts   # Vitest automated test suite
│   ├── App.tsx                     # Enterprise web UI
│   ├── index.css                   # Typography & design system tokens
│   └── main.tsx                    # React DOM root
├── scripts/
│   ├── ingest.py                   # CLI script to ingest data/hr_manual.pdf
│   └── evaluate_questions.py       # CLI script to evaluate the 5 LMS questions
├── tests/                          # Python automated test suite (unittest / pytest)
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_answer_service.py
│   ├── test_citations.py
│   └── test_ui_smoke.py
├── data/
│   ├── .gitkeep
│   ├── hr_manual.pdf               # 79-page Enso Group HR Policy Manual
│   └── evaluation_report.json      # Generated 5-question evaluation report
├── screenshots/
│   └── .gitkeep                    # Directory for UI screenshots
├── .env.example
├── .gitignore
├── requirements.txt
├── package.json
└── README.md
```

---

## 6. Setup & Installation

### Prerequisites

- **Python**: Python `3.10+`
- **Node.js**: Node.js `20+` and `npm` (for the interactive web interface)
- **API Key**: A Google Gemini API key (`GEMINI_API_KEY` or `GOOGLE_API_KEY`)

### macOS / Linux Setup

```bash
# 1. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install web application dependencies
npm install
```

### Windows (PowerShell) Setup

```powershell
# 1. Create and activate a Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install web application dependencies
npm install
```

---

## 7. Environment Variable Configuration

Copy `.env.example` to `.env` and set your credentials:

```bash
cp .env.example .env
```

```env
GEMINI_API_KEY="your_actual_gemini_api_key"
GOOGLE_API_KEY="your_actual_gemini_api_key"
LLM_PROVIDER=google
CHAT_MODEL=gemini-3.8-flash
EMBEDDING_MODEL=gemini-embedding-2-preview
CHROMA_PERSIST_DIR=./data/chroma
PDF_PATH=./data/hr_manual.pdf
RETRIEVAL_K=4
RELEVANCE_THRESHOLD=0.32
CHUNK_SIZE=900
CHUNK_OVERLAP=150
```

---

## 8. Placing & Ingesting the HR Policy Manual PDF

1. Ensure the Enso Group HR Policy Manual PDF is placed at `./data/hr_manual.pdf` (or upload it via the **Manage PDF Index** modal in the web UI).
2. Run the ingestion script to extract all 79 pages, clean text, split into 272 chunks, and persist into `./data/chroma`:

```bash
python3 scripts/ingest.py
```

---

## 9. Running the Application, Tests, and 5-Question Evaluation

### Start the Web Interface (Port 3000)

```bash
npm run dev
```

Open `http://localhost:3000` in your browser.

### Query via Python CLI

```bash
python3 app.py --status
python3 app.py --question "How many days of Privilege Leave (PL) does a permanent employee get per year?"
```

### Run the 5-Question LMS Evaluation Script

```bash
python3 scripts/evaluate_questions.py
```

This outputs the verified evaluation report to the terminal and saves `./data/evaluation_report.json`.

### Run Automated Tests

```bash
# Run Python unit and smoke tests (18 tests across 5 test suites)
python3 -m unittest discover -s tests -v

# Or with pytest (when installed in virtual environment)
pytest -v

# Run TypeScript / Vitest RAG test suite
npm test
```

---

## 10. Why an Answer May Be `Not found`

By design, this assistant is a **closed-book policy verifier**, not an open-ended chatbot. It returns `Not found` when:
1. **Unstated Policy Values (e.g., Question 3 — Health-insurance premium amount)**: While the manual mentions `Medical Reimbursement` and `ESIC` as salary components on Page 57 and `Travel insurance coverage` on Page 71, it never specifies a health-insurance premium amount.
2. **External Facts (e.g., Question 4 — Current GST rate in India)**: Tax rates outside the scope of the HR Policy Manual are not present in `hr_manual.pdf`.
3. **Insufficient or Contradictory Evidence**: If retrieved chunks score below `RELEVANCE_THRESHOLD` (`0.32`) or contain conflicting numbers without resolution, the system abstains rather than guessing.

---

## 11. Screenshots

Save screenshots of the running application under `screenshots/` for your LMS submission:
- `screenshots/01_policy_search_grounded_answer.png` — Main Policy Search view showing a grounded answer, Page/Section citation, and expanded verbatim excerpt.
- `screenshots/02_not_found_abstention.png` — Query for GST rate or health-insurance premium returning `Not found`.
- `screenshots/03_lms_evaluation_suite.png` — LMS 5-Question Evaluation tab showing all 5 verified results and citations.

*(To capture screenshots locally: start the app with `npm run dev`, open `http://localhost:3000`, run the sample questions and the LMS Evaluation tab, and save the browser captures into `screenshots/`.)*

---

## 12. Known Limitations, Privacy & Troubleshooting

- **Scanned PDFs without OCR**: If a user uploads a scanned image-only PDF with zero extractable text layer, ingestion rejects the file with an explicit error explaining that OCR text is required.
- **Privacy & External API Use**: During live LLM generation, only the user's question and the top `k=4` retrieved text excerpts from the HR manual are sent to the configured Google Gemini API (`gemini-3.8-flash`). The full PDF is processed locally and is never uploaded to third-party storage services.

---

## 13. Publishing to GitHub

To publish this repository to your personal GitHub account:

```bash
git init
git add .
git commit -m "Implement Enso Group HR Policy Assistant (Grounded RAG)"
git branch -M main
git remote add origin https://github.com/<your-username>/enso-hr-policy-assistant.git
git push -u origin main
```

*(Verify with `git status` before pushing that `.env` and `data/chroma/` remain ignored via `.gitignore`.)*
