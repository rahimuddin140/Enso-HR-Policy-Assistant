import React, { useState, useRef, useEffect } from 'react';
import {
  Search,
  BookOpen,
  CheckCircle2,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  RotateCcw,
  Upload,
  Copy,
  Check,
  ArrowUpRight,
  FileText,
  X,
  Send,
  MessageSquare,
} from 'lucide-react';

interface CitationSource {
  source: string;
  page: number;
  section: string | null;
  excerpt: string;
}

interface AnswerContract {
  answer: string;
  found: boolean;
  sources: CitationSource[];
  message: string;
  error?: string;
}

interface ChatTurn {
  id: string;
  question: string;
  response: AnswerContract | null;
  loading: boolean;
  error?: string | null;
  timestamp: string;
}

const QUICK_QUESTIONS = [
  'How many days of Privilege Leave (PL) does a permanent employee get per year?',
  'What is the probation period for employees up to Senior Manager?',
  "What is the company's health-insurance premium amount?",
  'What is the current GST rate in India?',
  'How many days of maternity leave are given?',
];

const POLICY_HIGHLIGHTS = [
  {
    number: '01.',
    title: 'Leave & Time Off Entitlements',
    summary:
      'Calendar-year allocations for Privilege Leave (12 days), Casual Leave (6 days), Sick Leave (6 days), Stress Leave (6 days), and Maternity Leave (12 weeks / 90 days).',
    reference: 'Annexure III · Page 60–62',
    sampleQuestion: 'How many days of Privilege Leave (PL) does a permanent employee get per year?',
    span: 'lg:col-span-7',
  },
  {
    number: '02.',
    title: 'Probation & Confirmation',
    summary:
      'Three-month probationary evaluation period for employees up to Senior Manager, including Probation Review Report (PRR) timelines and confirmation rules.',
    reference: 'Annexure I · Page 42–43',
    sampleQuestion: 'What is the probation period for employees up to Senior Manager?',
    span: 'lg:col-span-5',
  },
  {
    number: '03.',
    title: 'Domestic & International Travel',
    summary:
      'Per diem allowances, hotel tier limits across Asia, US, Europe, and Japan, booking class stipulations, and 7-day travel claim settlement procedures.',
    reference: 'Annexure IV · Page 67–78',
    sampleQuestion: 'What are the per diem reimbursement limits for international business travel?',
    span: 'lg:col-span-5',
  },
  {
    number: '04.',
    title: 'Relocation, Advances & Compensation',
    summary:
      'Relocation eligibility for transfers beyond 50 miles, interest-free emergency salary advances for tenured staff, and monthly CTC disbursement schedules.',
    reference: 'Annexure I & III · Page 45, 57–58',
    sampleQuestion: 'What is the minimum distance required to qualify for the Relocation Policy?',
    span: 'lg:col-span-7',
  },
];

const VERIFIED_FAQ_ITEMS = [
  {
    question: 'How many days of Privilege Leave (PL) does a permanent employee get per year?',
    answer:
      'A permanent employee gets 12 days of Privilege Leave (PL) per calendar year (1st January – 31st December). Unavailed PL cannot be carried forward to the next calendar year, and accumulated balance can be encashed only as part of full and final settlement at basic pay.',
    found: true,
    citation: 'Enso Group HR Manual · Page 60 · [G] Leave Policy',
  },
  {
    question: 'What is the probation period for employees up to Senior Manager?',
    answer:
      'All employees up to Senior Manager (Sr. Manager) are placed on probation for a period of 3 months from their date of joining. Employees above Senior Manager are treated as confirmed employees from the date of joining.',
    found: true,
    citation: 'Enso Group HR Manual · Page 42 · [B] Probation Policy',
  },
  {
    question: "What is the company's health-insurance premium amount?",
    answer: 'Not found',
    note: 'The HR Policy Manual mentions Medical Reimbursement and ESIC as salary components of CTC (Page 57) and Travel Insurance coverage (Page 71), but does not specify a health-insurance premium amount.',
    found: false,
    citation: 'Not stated in HR Policy Manual',
  },
  {
    question: 'What is the current GST rate in India?',
    answer: 'Not found',
    note: 'External tax rates are not covered in the Enso Group HR Policy Manual. The assistant answers strictly from official handbook content.',
    found: false,
    citation: 'Not stated in HR Policy Manual',
  },
  {
    question: 'How many days of maternity leave are given?',
    answer:
      'The company provides a total of 12 weeks (90 days) of Maternity Leave (ML) with pay for confirmed female staff who have worked at least 80 days in the preceding 12 months (with no more than 6 weeks / 45 days preceding the expected delivery date, and 4 weeks with pay in case of miscarriage or medical termination of pregnancy).',
    found: true,
    citation: 'Enso Group HR Manual · Page 62 · [G] Leave Policy — Maternity Leave (Leeway)',
  },
];

function formatCleanExcerpt(rawExcerpt: string): string {
  return rawExcerpt
    .replace(/^HR Manual\s*-\s*\d+\s*/i, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

export default function App() {
  const [question, setQuestion] = useState<string>('');
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [asking, setAsking] = useState<boolean>(false);
  const [inputError, setInputError] = useState<string | null>(null);
  const [expandedExcerpts, setExpandedExcerpts] = useState<Record<string, boolean>>({});
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Handbook upload modal state
  const [showHandbookModal, setShowHandbookModal] = useState<boolean>(false);
  const [confirmRefresh, setConfirmRefresh] = useState<boolean>(false);
  const [updatingHandbook, setUpdatingHandbook] = useState<boolean>(false);
  const [handbookMessage, setHandbookMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [manualReady, setManualReady] = useState<boolean>(true);

  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    fetch('/api/status')
      .then((r) => r.json())
      .then((data) => {
        if (data && typeof data.ready === 'boolean') {
          setManualReady(data.ready);
        }
      })
      .catch(() => {
        // ignore status check error
      });
  }, []);

  const focusAssistantInput = () => {
    const el = document.getElementById('assistant-workspace');
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    setTimeout(() => {
      inputRef.current?.focus();
    }, 150);
  };

  const handleAsk = async (presetQuestion?: string) => {
    const queryText = (presetQuestion !== undefined ? presetQuestion : question).trim();
    setInputError(null);

    if (!queryText) {
      setInputError('Please enter a policy question.');
      inputRef.current?.focus();
      return;
    }

    if (presetQuestion === undefined) {
      setQuestion('');
    }

    const turnId = `turn-${Date.now()}`;
    const newTurn: ChatTurn = {
      id: turnId,
      question: queryText,
      response: null,
      loading: true,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setTurns((prev) => [...prev, newTurn]);
    setAsking(true);

    setTimeout(() => {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 80);

    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: queryText }),
      });
      const data: AnswerContract = await res.json();

      setTurns((prev) =>
        prev.map((t) =>
          t.id === turnId
            ? {
                ...t,
                loading: false,
                response: data,
                error: !res.ok && data.error ? data.error : null,
              }
            : t
        )
      );

      if (data.sources && data.sources.length > 0) {
        setExpandedExcerpts((prev) => ({
          ...prev,
          [`${turnId}-0`]: true,
        }));
      }
    } catch (err: any) {
      setTurns((prev) =>
        prev.map((t) =>
          t.id === turnId
            ? {
                ...t,
                loading: false,
                error: 'Unable to reach the policy assistant right now. Please try again.',
              }
            : t
        )
      );
    } finally {
      setAsking(false);
    }
  };

  const handleClearConversation = () => {
    setTurns([]);
    setQuestion('');
    setInputError(null);
    setExpandedExcerpts({});
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!asking) {
        handleAsk();
      }
    } else if (e.key === 'Escape') {
      setQuestion('');
      setInputError(null);
    }
  };

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1800);
  };

  const toggleExcerpt = (key: string) => {
    setExpandedExcerpts((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const handleRefreshHandbook = async () => {
    if (!confirmRefresh) {
      setConfirmRefresh(true);
      return;
    }
    setUpdatingHandbook(true);
    setHandbookMessage(null);
    try {
      const res = await fetch('/api/ingest', { method: 'POST' });
      const data = await res.json();
      if (res.ok && data.success) {
        setManualReady(true);
        setConfirmRefresh(false);
        setHandbookMessage({
          type: 'success',
          text: 'The Enso Group HR Policy Manual has been refreshed and is ready for questions.',
        });
      } else {
        setHandbookMessage({
          type: 'error',
          text: data.error || 'Could not refresh the HR Policy Manual.',
        });
      }
    } catch {
      setHandbookMessage({
        type: 'error',
        text: 'A network error occurred while refreshing the handbook.',
      });
    } finally {
      setUpdatingHandbook(false);
    }
  };

  const handleUploadHandbook = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.pdf') && file.type !== 'application/pdf') {
      setHandbookMessage({
        type: 'error',
        text: 'Please select a valid PDF document (.pdf).',
      });
      return;
    }

    setUpdatingHandbook(true);
    setHandbookMessage(null);
    try {
      const formData = new FormData();
      formData.append('pdf', file);
      const res = await fetch('/api/upload-pdf', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok && data.success) {
        setManualReady(true);
        setHandbookMessage({
          type: 'success',
          text: `Successfully updated the handbook with "${file.name}".`,
        });
      } else {
        setHandbookMessage({
          type: 'error',
          text: data.error || 'Unable to process the uploaded PDF.',
        });
      }
    } catch {
      setHandbookMessage({
        type: 'error',
        text: 'An error occurred while uploading the PDF document.',
      });
    } finally {
      setUpdatingHandbook(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#F8FAFC] text-[#0F172A]">
      {/* Top Bar Contract: 3 zones separated by gap-8 */}
      <header className="sticky top-0 z-30 bg-white/95 backdrop-blur-sm border-b border-slate-200 px-6 py-4">
        <div className="max-w-[1280px] mx-auto flex items-center justify-between gap-8">
          {/* Zone 1: Brand Wordmark */}
          <a
            href="#assistant-workspace"
            className="font-display text-lg font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-blue-700"
          >
            Enso HR Assistant
          </a>

          {/* Zone 2: Single-line navigation links */}
          <nav
            aria-label="Primary Navigation"
            className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-600"
          >
            <a
              href="#assistant-workspace"
              className="hover:text-slate-900 transition-colors whitespace-nowrap shrink-0"
            >
              Ask Assistant
            </a>
            <a
              href="#policy-highlights"
              className="hover:text-slate-900 transition-colors whitespace-nowrap shrink-0"
            >
              Policy Guide
            </a>
            <a
              href="#verified-answers"
              className="hover:text-slate-900 transition-colors whitespace-nowrap shrink-0"
            >
              Common Questions
            </a>
            <button
              type="button"
              onClick={() => {
                setHandbookMessage(null);
                setConfirmRefresh(false);
                setShowHandbookModal(true);
              }}
              className="hover:text-slate-900 transition-colors whitespace-nowrap shrink-0"
            >
              Update Manual
            </button>
          </nav>

          {/* Zone 3: 1 Primary CTA */}
          <div className="flex items-center shrink-0">
            <button
              type="button"
              onClick={focusAssistantInput}
              className="px-4 py-2 text-xs font-semibold text-white bg-blue-700 rounded-lg hover:bg-blue-800 transition-colors whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-700"
            >
              Ask a Question
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1">
        {/* Hero & Interactive Policy Chat Workspace */}
        <section
          id="assistant-workspace"
          className="max-w-[1280px] mx-auto px-6 pt-10 pb-14"
        >
          {/* Editorial Product Header */}
          <div className="max-w-3xl mb-8">
            <p className="text-xs font-semibold text-blue-700 mb-2">
              Enso Group Employee Handbook · Official Policy Reference
            </p>
            <h1
              className="font-display text-3xl sm:text-4xl lg:text-[42px] font-bold text-slate-900 tracking-tight leading-[1.15]"
              style={{ textWrap: 'balance' }}
            >
              Instant, verifiable answers to Enso Group HR policies.
            </h1>
            <p className="text-base text-slate-600 mt-3 leading-relaxed max-w-[65ch]">
              Ask about leave entitlements, probation timelines, business travel allowances, or workplace guidelines. Every answer is backed by the exact page and section of the Enso Group HR Policy Manual—or returns <span className="font-mono font-semibold text-slate-800">Not found</span> if a topic is not covered in the handbook.
            </p>
          </div>

          {/* Main Interactive Assistant Container */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* Chat & Answer Column */}
            <div className="lg:col-span-8 bg-white border border-slate-200 rounded-2xl shadow-xs overflow-hidden flex flex-col">
              {/* Assistant Header Bar */}
              <div className="px-6 py-4 border-b border-slate-200 flex items-center justify-between gap-4 bg-slate-50/60">
                <div className="flex items-center gap-2.5 text-xs text-slate-600">
                  <BookOpen className="w-4 h-4 text-blue-700 shrink-0" />
                  <span className="font-semibold text-slate-900">
                    Enso Group HR Policy Manual
                  </span>
                  <span aria-hidden="true">·</span>
                  <span>
                    {manualReady ? '79 Pages Active' : 'Manual Required'}
                  </span>
                </div>

                {turns.length > 0 && (
                  <button
                    type="button"
                    onClick={handleClearConversation}
                    className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-600 hover:text-slate-900 transition-colors whitespace-nowrap shrink-0"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    Clear Conversation
                  </button>
                )}
              </div>

              {/* Conversation Thread / Welcome State */}
              <div className="p-6 space-y-6 min-h-[380px] max-h-[680px] overflow-y-auto">
                {turns.length === 0 ? (
                  <div className="py-8 space-y-6">
                    <div className="max-w-xl">
                      <h2 className="font-display text-lg font-bold text-slate-900">
                        What would you like to check in the HR Manual?
                      </h2>
                      <p className="text-sm text-slate-600 mt-1">
                        Choose a common employee question below or type your own question in the box underneath.
                      </p>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      {QUICK_QUESTIONS.map((q, idx) => (
                        <button
                          key={q}
                          type="button"
                          onClick={() => handleAsk(q)}
                          disabled={asking}
                          className="group text-left p-4 rounded-xl border border-slate-200 hover:border-blue-700/60 hover:bg-blue-50/20 transition-all flex items-start justify-between gap-3"
                        >
                          <div className="space-y-1">
                            <span className="font-mono text-xs font-semibold text-blue-700 tabular-nums">
                              0{idx + 1}.
                            </span>
                            <p className="text-xs font-medium text-slate-800 group-hover:text-slate-900 leading-relaxed">
                              {q}
                            </p>
                          </div>
                          <ArrowUpRight className="w-4 h-4 text-slate-400 group-hover:text-blue-700 shrink-0 mt-0.5 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  turns.map((turn) => (
                    <div key={turn.id} className="space-y-4 pb-6 border-b border-slate-100 last:border-b-0 last:pb-0">
                      {/* Employee Question */}
                      <div className="flex items-start justify-end">
                        <div className="bg-slate-900 text-white rounded-2xl rounded-tr-xs px-5 py-3.5 max-w-[85%]">
                          <p className="text-sm font-medium leading-relaxed">
                            {turn.question}
                          </p>
                        </div>
                      </div>

                      {/* Assistant Answer */}
                      {turn.loading ? (
                        <div
                          role="status"
                          aria-live="polite"
                          className="bg-slate-50 border border-slate-200 rounded-2xl rounded-tl-xs p-5 space-y-3"
                        >
                          <div className="flex items-center gap-2.5 text-xs font-semibold text-blue-700">
                            <div className="w-3.5 h-3.5 border-2 border-blue-700 border-t-transparent rounded-full animate-spin" />
                            <span>Checking the Enso Group HR Policy Manual...</span>
                          </div>
                          <div className="space-y-2">
                            <div className="h-3.5 bg-slate-200/70 rounded w-3/4 animate-pulse" />
                            <div className="h-3.5 bg-slate-200/70 rounded w-1/2 animate-pulse" />
                          </div>
                        </div>
                      ) : turn.error ? (
                        <div
                          role="alert"
                          className="bg-red-50/70 border border-red-200 rounded-2xl rounded-tl-xs p-5 flex items-start gap-3 text-xs text-red-900"
                        >
                          <AlertCircle className="w-4 h-4 text-red-700 shrink-0 mt-0.5" />
                          <div>
                            <p className="font-semibold">Unable to Retrieve Answer</p>
                            <p className="mt-0.5 text-red-800">{turn.error}</p>
                          </div>
                        </div>
                      ) : turn.response ? (
                        <div className="bg-slate-50/70 border border-slate-200 rounded-2xl rounded-tl-xs overflow-hidden">
                          <div className="p-5 bg-white">
                            <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5 text-xs text-slate-500">
                              <div className="flex items-center gap-2">
                                {turn.response.found ? (
                                  <span className="inline-flex items-center gap-1.5 font-semibold text-emerald-800">
                                    <CheckCircle2 className="w-4 h-4 text-emerald-700" />
                                    Verified Handbook Policy
                                  </span>
                                ) : (
                                  <span className="inline-flex items-center gap-1.5 font-semibold text-amber-800">
                                    <AlertCircle className="w-4 h-4 text-amber-700" />
                                    Not Covered in HR Manual
                                  </span>
                                )}
                                <span aria-hidden="true">·</span>
                                <span className="tabular-nums">{turn.timestamp}</span>
                              </div>

                              <button
                                type="button"
                                onClick={() => handleCopy(turn.id, turn.response!.answer)}
                                className="inline-flex items-center gap-1 text-xs font-medium text-slate-500 hover:text-slate-900 transition-colors whitespace-nowrap shrink-0"
                              >
                                {copiedId === turn.id ? (
                                  <>
                                    <Check className="w-3.5 h-3.5 text-emerald-700" />
                                    Copied
                                  </>
                                ) : (
                                  <>
                                    <Copy className="w-3.5 h-3.5" />
                                    Copy Answer
                                  </>
                                )}
                              </button>
                            </div>

                            {turn.response.found ? (
                              <p className="text-[15px] leading-relaxed text-slate-900 font-medium max-w-[68ch]">
                                {turn.response.answer}
                              </p>
                            ) : (
                              <div className="space-y-1.5">
                                <div className="font-mono text-base font-bold text-slate-900">
                                  Not found
                                </div>
                                <p className="text-sm text-slate-600 leading-relaxed max-w-[65ch]">
                                  {turn.response.message ||
                                    'This topic is not mentioned in the Enso Group HR Policy Manual. For questions outside the handbook, please contact your HR Business Partner.'}
                                </p>
                              </div>
                            )}
                          </div>

                          {/* Handbook Citations & Expandable Policy Excerpt */}
                          {turn.response.found && turn.response.sources.length > 0 && (
                            <div className="px-5 py-3.5 border-t border-slate-200 bg-slate-50 space-y-2.5">
                              <div className="text-xs font-semibold text-slate-600">
                                Handbook Reference
                              </div>

                              {turn.response.sources.map((src, sIdx) => {
                                const excerptKey = `${turn.id}-${sIdx}`;
                                const isExpanded = Boolean(expandedExcerpts[excerptKey]);
                                return (
                                  <div
                                    key={excerptKey}
                                    className="bg-white border border-slate-200 rounded-xl overflow-hidden"
                                  >
                                    <button
                                      type="button"
                                      onClick={() => toggleExcerpt(excerptKey)}
                                      className="w-full px-4 py-3 text-left flex items-center justify-between gap-4 hover:bg-slate-50 transition-colors"
                                    >
                                      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-700">
                                        <FileText className="w-3.5 h-3.5 text-blue-700 shrink-0" />
                                        <span className="font-semibold text-slate-900">
                                          Enso Group HR Manual
                                        </span>
                                        <span aria-hidden="true">·</span>
                                        <span className="font-mono font-semibold text-blue-700 tabular-nums">
                                          Page {src.page}
                                        </span>
                                        {src.section && (
                                          <>
                                            <span aria-hidden="true">·</span>
                                            <span className="font-medium text-slate-700">
                                              {src.section}
                                            </span>
                                          </>
                                        )}
                                      </div>

                                      <span className="inline-flex items-center gap-1 text-xs font-medium text-blue-700 whitespace-nowrap shrink-0">
                                        {isExpanded ? 'Hide Excerpt' : 'Read Excerpt'}
                                        {isExpanded ? (
                                          <ChevronUp className="w-3.5 h-3.5" />
                                        ) : (
                                          <ChevronDown className="w-3.5 h-3.5" />
                                        )}
                                      </span>
                                    </button>

                                    {isExpanded && (
                                      <div className="px-4 py-3.5 border-t border-slate-200 bg-slate-50/70">
                                        <p className="text-xs font-medium text-slate-500 mb-2">
                                          Verbatim Passage from Page {src.page}
                                        </p>
                                        <div className="text-xs leading-relaxed text-slate-800 whitespace-pre-line bg-white p-3.5 rounded-lg border border-slate-200">
                                          {formatCleanExcerpt(src.excerpt)}
                                        </div>
                                      </div>
                                    )}
                                  </div>
                                );
                              })}
                            </div>
                          )}
                        </div>
                      ) : null}
                    </div>
                  ))
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Composer Input Area */}
              <div className="p-4 sm:p-5 border-t border-slate-200 bg-white">
                {inputError && (
                  <div
                    role="alert"
                    className="mb-3 px-3.5 py-2.5 rounded-lg border border-red-200 bg-red-50 flex items-center justify-between gap-2 text-xs text-red-900"
                  >
                    <span>{inputError}</span>
                    <button
                      type="button"
                      onClick={() => setInputError(null)}
                      aria-label="Dismiss error"
                    >
                      <X className="w-3.5 h-3.5 text-red-700" />
                    </button>
                  </div>
                )}

                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    handleAsk();
                  }}
                  className="space-y-3"
                >
                  <label htmlFor="hr-assistant-input" className="sr-only">
                    Ask a question about Enso Group HR policies
                  </label>
                  <div className="relative">
                    <textarea
                      id="hr-assistant-input"
                      ref={inputRef}
                      rows={2}
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      onKeyDown={handleKeyDown}
                      disabled={asking}
                      placeholder="Ask a question about leave, probation, travel, or workplace policies..."
                      className="w-full rounded-xl border border-slate-300 bg-slate-50/50 focus:bg-white px-4 py-3 pr-28 text-sm text-slate-900 placeholder:text-slate-400 focus:border-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-700/15 disabled:opacity-60 transition-colors resize-none"
                    />
                    <div className="absolute right-2.5 bottom-2.5 flex items-center gap-1.5">
                      {question.trim() && (
                        <button
                          type="button"
                          onClick={() => setQuestion('')}
                          className="px-2.5 py-1.5 text-xs font-medium text-slate-500 hover:text-slate-800 rounded-lg hover:bg-slate-100 transition-colors"
                        >
                          Clear
                        </button>
                      )}
                      <button
                        type="submit"
                        disabled={asking}
                        className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold text-white bg-blue-700 rounded-lg hover:bg-blue-800 disabled:opacity-60 transition-colors whitespace-nowrap shrink-0"
                      >
                        <Send className="w-3.5 h-3.5" />
                        {asking ? 'Searching...' : 'Ask'}
                      </button>
                    </div>
                  </div>

                  <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500">
                    <span>Press Enter to ask · Shift+Enter for a new line</span>
                    <span>Responses cite exact handbook pages</span>
                  </div>
                </form>
              </div>
            </div>

            {/* Right Column: Suggested Questions & Handbook Overview */}
            <aside className="lg:col-span-4 space-y-6">
              <div className="bg-white border border-slate-200 rounded-2xl p-6">
                <div className="flex items-center gap-2 mb-3">
                  <MessageSquare className="w-4 h-4 text-blue-700" />
                  <h2 className="font-display text-sm font-bold text-slate-900">
                    Popular Employee Questions
                  </h2>
                </div>
                <p className="text-xs text-slate-600 mb-4 leading-relaxed">
                  Click any question below to get an immediate answer with its handbook page citation:
                </p>

                <div className="space-y-2">
                  {QUICK_QUESTIONS.map((q, i) => (
                    <button
                      key={q}
                      type="button"
                      onClick={() => handleAsk(q)}
                      disabled={asking}
                      className="w-full text-left p-3 rounded-xl border border-slate-200 hover:border-blue-700/50 hover:bg-slate-50 transition-colors flex items-start gap-2.5"
                    >
                      <span className="font-mono text-xs font-semibold text-blue-700 tabular-nums mt-0.5">
                        0{i + 1}.
                      </span>
                      <span className="text-xs font-medium text-slate-800 leading-snug">
                        {q}
                      </span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="bg-white border border-slate-200 rounded-2xl p-6 space-y-3">
                <h2 className="font-display text-sm font-bold text-slate-900">
                  More Handbook Topics
                </h2>
                <div className="space-y-1.5">
                  {[
                    'How many days of Casual Leave (CL) and Sick Leave (SL) are allowed per year?',
                    'What are the standard office working hours and lunch timings?',
                    'What are the eligibility rules for an interest-free Staff Advance?',
                    'What is the notice period required for employee resignation?',
                  ].map((topicQ) => (
                    <button
                      key={topicQ}
                      type="button"
                      onClick={() => handleAsk(topicQ)}
                      disabled={asking}
                      className="w-full text-left px-3 py-2 rounded-lg text-xs text-slate-700 hover:text-slate-900 hover:bg-slate-50 transition-colors flex items-center justify-between gap-2"
                    >
                      <span className="leading-snug">{topicQ}</span>
                      <ArrowUpRight className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                    </button>
                  ))}
                </div>
              </div>
            </aside>
          </div>
        </section>

        {/* Section 2: Curated Policy Guide (Asymmetric Bento Grid) */}
        <section
          id="policy-highlights"
          className="bg-white border-y border-slate-200 py-16 px-6"
        >
          <div className="max-w-[1280px] mx-auto">
            <div className="max-w-2xl mb-10">
              <p className="text-xs font-semibold text-blue-700 mb-2">
                Handbook Coverage
              </p>
              <h2 className="font-display text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight">
                Key Policy Areas in the Enso Group HR Manual
              </h2>
              <p className="text-sm text-slate-600 mt-2 leading-relaxed">
                Explore core chapters from the 79-page employee manual. Select any policy card to query the assistant directly.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {POLICY_HIGHLIGHTS.map((item) => (
                <div
                  key={item.title}
                  className={`${item.span} bg-[#F8FAFC] border border-slate-200 rounded-2xl p-6 flex flex-col justify-between hover:border-slate-300 transition-colors`}
                >
                  <div>
                    <div className="flex items-center justify-between gap-4 text-xs text-slate-500 mb-3">
                      <span className="font-mono font-semibold text-blue-700">
                        {item.number}
                      </span>
                      <span className="font-mono tabular-nums">{item.reference}</span>
                    </div>
                    <h3 className="font-display text-lg font-bold text-slate-900">
                      {item.title}
                    </h3>
                    <p className="text-sm text-slate-600 mt-2 leading-relaxed">
                      {item.summary}
                    </p>
                  </div>

                  <div className="mt-6 pt-4 border-t border-slate-200/80 flex items-center justify-between gap-4">
                    <span className="text-xs text-slate-500 truncate">
                      &ldquo;{item.sampleQuestion}&rdquo;
                    </span>
                    <button
                      type="button"
                      onClick={() => {
                        focusAssistantInput();
                        handleAsk(item.sampleQuestion);
                      }}
                      className="inline-flex items-center gap-1 text-xs font-semibold text-blue-700 hover:text-blue-800 whitespace-nowrap shrink-0"
                    >
                      Ask Assistant
                      <ArrowUpRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Section 3: Verified Handbook Reference Answers */}
        <section
          id="verified-answers"
          className="max-w-[1280px] mx-auto px-6 py-16"
        >
          <div className="flex flex-wrap items-end justify-between gap-4 mb-8">
            <div className="max-w-2xl">
              <p className="text-xs font-semibold text-blue-700 mb-2">
                Verified Reference Directory
              </p>
              <h2 className="font-display text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight">
                Frequently Checked Policy Answers
              </h2>
              <p className="text-sm text-slate-600 mt-2">
                Direct answers verified word-for-word against the official Enso Group HR Policy Manual.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4">
            {VERIFIED_FAQ_ITEMS.map((item, idx) => (
              <div
                key={item.question}
                className="bg-white border border-slate-200 rounded-2xl p-6 flex flex-col md:flex-row md:items-start justify-between gap-6"
              >
                <div className="space-y-2 max-w-3xl">
                  <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
                    <span className="font-mono font-semibold text-blue-700 tabular-nums">
                      0{idx + 1}.
                    </span>
                    <span>{item.citation}</span>
                  </div>

                  <h3 className="font-display text-base font-bold text-slate-900">
                    {item.question}
                  </h3>

                  {item.found ? (
                    <p className="text-sm text-slate-700 leading-relaxed">
                      {item.answer}
                    </p>
                  ) : (
                    <div className="space-y-1">
                      <span className="font-mono text-sm font-bold text-amber-800">
                        Not found
                      </span>
                      <p className="text-xs text-slate-500 leading-relaxed">
                        {item.note}
                      </p>
                    </div>
                  )}
                </div>

                <div className="flex items-center gap-3 shrink-0">
                  <button
                    type="button"
                    onClick={() => handleCopy(`faq-${idx}`, item.answer)}
                    className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-slate-700 bg-slate-100 rounded-lg hover:bg-slate-200 transition-colors whitespace-nowrap shrink-0"
                  >
                    {copiedId === `faq-${idx}` ? (
                      <>
                        <Check className="w-3.5 h-3.5 text-emerald-700" />
                        Copied
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5" />
                        Copy Answer
                      </>
                    )}
                  </button>

                  <button
                    type="button"
                    onClick={() => {
                      focusAssistantInput();
                      handleAsk(item.question);
                    }}
                    className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-blue-700 bg-blue-50 rounded-lg hover:bg-blue-100 transition-colors whitespace-nowrap shrink-0"
                  >
                    <Search className="w-3.5 h-3.5" />
                    Verify in Chat
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>
      </main>

      {/* Handbook Upload / Refresh Modal */}
      {showHandbookModal && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="handbook-modal-title"
          className="fixed inset-0 z-50 bg-slate-900/50 flex items-center justify-center p-4"
        >
          <div className="bg-white border border-slate-200 rounded-2xl max-w-md w-full p-6 space-y-5 shadow-lg">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2
                  id="handbook-modal-title"
                  className="font-display text-lg font-bold text-slate-900"
                >
                  Update HR Policy Manual
                </h2>
                <p className="text-xs text-slate-600 mt-1">
                  Upload a revised PDF handbook or refresh the active Enso Group HR Policy Manual.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setShowHandbookModal(false)}
                className="text-slate-400 hover:text-slate-700"
                aria-label="Close modal"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {handbookMessage && (
              <div
                className={`p-3.5 rounded-xl border text-xs leading-relaxed ${
                  handbookMessage.type === 'success'
                    ? 'border-emerald-200 bg-emerald-50 text-emerald-900'
                    : 'border-red-200 bg-red-50 text-red-900'
                }`}
              >
                {handbookMessage.text}
              </div>
            )}

            <div className="space-y-4">
              <div className="p-4 rounded-xl border border-slate-200 space-y-3">
                <div className="flex items-center gap-2 text-xs font-semibold text-slate-900">
                  <Upload className="w-4 h-4 text-blue-700" />
                  <span>Upload New Handbook PDF</span>
                </div>
                <p className="text-xs text-slate-600">
                  Select an official HR Policy Manual PDF file (maximum 25 MB).
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,application/pdf"
                  onChange={handleUploadHandbook}
                  disabled={updatingHandbook}
                  aria-label="Upload HR Policy Manual PDF"
                  className="block w-full text-xs text-slate-600 file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                />
              </div>

              <div className="p-4 rounded-xl border border-slate-200 space-y-3">
                <div className="text-xs font-semibold text-slate-900">
                  Refresh Current Manual
                </div>
                <p className="text-xs text-slate-600">
                  Reloads the active 79-page Enso Group HR Policy Manual.
                </p>
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={handleRefreshHandbook}
                    disabled={updatingHandbook}
                    className={`px-4 py-2 text-xs font-semibold rounded-lg transition-colors whitespace-nowrap shrink-0 ${
                      confirmRefresh
                        ? 'bg-amber-700 text-white hover:bg-amber-800'
                        : 'bg-slate-900 text-white hover:bg-slate-800'
                    }`}
                  >
                    {updatingHandbook
                      ? 'Updating Manual...'
                      : confirmRefresh
                      ? 'Click Again to Confirm Refresh'
                      : 'Refresh Active Manual'}
                  </button>
                  {confirmRefresh && !updatingHandbook && (
                    <button
                      type="button"
                      onClick={() => setConfirmRefresh(false)}
                      className="text-xs text-slate-500 hover:text-slate-800"
                    >
                      Cancel
                    </button>
                  )}
                </div>
              </div>
            </div>

            <div className="flex justify-end pt-1">
              <button
                type="button"
                onClick={() => setShowHandbookModal(false)}
                className="px-4 py-2 text-xs font-medium text-slate-700 bg-slate-100 rounded-lg hover:bg-slate-200"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Quiet Product Footer */}
      <footer className="bg-white border-t border-slate-200 px-6 py-6">
        <div className="max-w-[1280px] mx-auto flex flex-wrap items-center justify-between gap-4 text-xs text-slate-500">
          <div className="flex items-center gap-3">
            <span className="font-display font-bold text-slate-900">Enso HR Assistant</span>
            <span aria-hidden="true">·</span>
            <span>Official Employee Policy Reference</span>
          </div>
          <div className="flex items-center gap-6">
            <a href="#assistant-workspace" className="hover:text-slate-900 transition-colors">
              Ask Assistant
            </a>
            <a href="#policy-highlights" className="hover:text-slate-900 transition-colors">
              Policy Guide
            </a>
            <a href="#verified-answers" className="hover:text-slate-900 transition-colors">
              Common Questions
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
