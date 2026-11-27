import { useEffect, useRef, useState } from 'react';
import {
  ArrowUp,
  BookOpen,
  Bot,
  Cpu,
  FileUp,
  Loader2,
  PanelRightOpen,
  Settings2,
  Sparkles,
  User,
} from 'lucide-react';
import { motion } from 'framer-motion';
import type { ChatMessage, CitationItem, LoadedPdfDoc } from '../types/rag';
import type { RagMode } from '../App';
import { PRESET_PROMPTS, PROJECT_SUBTITLE, PROJECT_TITLE } from '../data/demoScenario';
import { PipelineStepper } from './PipelineStepper';
import { MarkdownMessage } from './MarkdownMessage';
import { cn } from '../lib/utils';

interface LlmConfig {
  apiKey: string;
  baseUrl: string;
  model: string;
}

interface ChatPaneProps {
  messages: ChatMessage[];
  docs: LoadedPdfDoc[];
  embedderLabel: string;
  ragMode: RagMode;
  onChangeRagMode: (mode: RagMode) => void;
  llmConfig: LlmConfig;
  onChangeLlmConfig: (cfg: LlmConfig) => void;
  isGenerating: boolean;
  isPdfOpen: boolean;
  activeCitationId: string | null;
  onSendQuery: (query: string) => void;
  onSelectCitation: (citation: CitationItem) => void;
  onTogglePdfPane: () => void;
  onUploadPdf: (file: File) => void;
}

export function ChatPane({
  messages,
  docs,
  embedderLabel,
  ragMode,
  onChangeRagMode,
  llmConfig,
  onChangeLlmConfig,
  isGenerating,
  isPdfOpen,
  activeCitationId,
  onSendQuery,
  onSelectCitation,
  onTogglePdfPane,
  onUploadPdf,
}: ChatPaneProps) {
  const [input, setInput] = useState('');
  const [showLlmSettings, setShowLlmSettings] = useState(false);
  const scrollEndRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    scrollEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isGenerating]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isGenerating) return;
    setInput('');
    onSendQuery(trimmed);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type === 'application/pdf') {
      onUploadPdf(file);
    }
    e.target.value = '';
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="flex flex-col h-full bg-white relative overflow-hidden">
      {/* Top Header */}
      <header className="flex items-center justify-between px-4 sm:px-6 py-3 border-b border-slate-100 bg-white/90 backdrop-blur-md z-10 shrink-0">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="flex items-center justify-center w-8 h-8 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 text-white shadow-sm shrink-0">
            <BookOpen className="w-4 h-4" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h1 className="text-sm font-semibold text-slate-900 truncate font-display">
                {PROJECT_TITLE}
              </h1>
              <span className="hidden md:inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200/70">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                {embedderLabel}
              </span>
            </div>
            <p className="text-[11px] text-slate-500 truncate">{PROJECT_SUBTITLE}</p>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {/* Mode Toggle: Real AI vs RAG Test Only */}
          <div
            className="hidden sm:inline-flex items-center p-0.5 rounded-full bg-slate-100 border border-slate-200/80"
            role="group"
            aria-label="RAG Execution Mode"
          >
            <button
              type="button"
              onClick={() => onChangeRagMode('ai_chat')}
              className={cn(
                'inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-medium transition-all cursor-pointer',
                ragMode === 'ai_chat'
                  ? 'bg-white text-blue-700 shadow-2xs'
                  : 'text-slate-600 hover:text-slate-900'
              )}
              title="ใช้ AI สังเคราะห์คำตอบจากหลักฐานที่ RAG ค้นพบ"
            >
              <Sparkles className="w-3 h-3" />
              <span>ใช้ AI จริง</span>
            </button>
            <button
              type="button"
              onClick={() => onChangeRagMode('rag_only')}
              className={cn(
                'inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-medium transition-all cursor-pointer',
                ragMode === 'rag_only'
                  ? 'bg-white text-amber-700 shadow-2xs'
                  : 'text-slate-600 hover:text-slate-900'
              )}
              title="ทดสอบระบบ RAG ล้วน (แสดง Chunks ดิบ + คะแนน Rerank/Lexical)"
            >
              <Cpu className="w-3 h-3" />
              <span>ทดสอบ RAG</span>
            </button>
          </div>

          {/* Optional LLM API Settings */}
          <button
            type="button"
            onClick={() => setShowLlmSettings((v) => !v)}
            className={cn(
              'p-1.5 rounded-full border transition-colors cursor-pointer',
              showLlmSettings || llmConfig.apiKey
                ? 'bg-blue-50 text-blue-700 border-blue-200'
                : 'bg-[#F0F4F9] text-slate-600 hover:bg-slate-200/80 border-slate-200/70'
            )}
            title="ตั้งค่า OpenAI / OpenRouter / Ollama LLM API"
          >
            <Settings2 className="w-3.5 h-3.5" />
          </button>

          {/* Hidden PDF Upload Input */}
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf"
            onChange={handleFileChange}
            className="hidden"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium text-slate-700 bg-[#F0F4F9] hover:bg-slate-200/80 border border-slate-200/70 transition-colors cursor-pointer"
            title="Upload your own PDF file for live Q&A"
          >
            <FileUp className="w-3.5 h-3.5 text-blue-600" />
            <span>Load Custom PDF</span>
          </button>

          {!isPdfOpen && (
            <button
              type="button"
              onClick={onTogglePdfPane}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium text-blue-700 bg-blue-50 hover:bg-blue-100 border border-blue-200/80 transition-colors cursor-pointer"
            >
              <PanelRightOpen className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Inspect PDF ({docs.length})</span>
            </button>
          )}
        </div>
      </header>

      {/* Collapsible LLM Config Bar */}
      {showLlmSettings && (
        <div className="px-4 sm:px-6 py-2.5 bg-[#F8FAFC] border-b border-slate-200/80 flex flex-wrap items-center gap-2 text-xs z-10">
          <span className="font-semibold text-slate-700">LLM Config (Optional):</span>
          <input
            type="password"
            value={llmConfig.apiKey}
            onChange={(e) => onChangeLlmConfig({ ...llmConfig, apiKey: e.target.value })}
            placeholder="OpenAI / OpenRouter API Key (เว้นว่าง = ใช้ Local Ollama / Synthesizer)"
            className="flex-1 min-w-[200px] px-2.5 py-1 rounded-lg border border-slate-200 bg-white text-slate-800 focus:outline-none focus:border-blue-400"
          />
          <input
            type="text"
            value={llmConfig.baseUrl}
            onChange={(e) => onChangeLlmConfig({ ...llmConfig, baseUrl: e.target.value })}
            placeholder="Base URL (e.g. http://127.0.0.1:11434/v1)"
            className="w-52 px-2.5 py-1 rounded-lg border border-slate-200 bg-white text-slate-800 focus:outline-none focus:border-blue-400"
          />
          <input
            type="text"
            value={llmConfig.model}
            onChange={(e) => onChangeLlmConfig({ ...llmConfig, model: e.target.value })}
            placeholder="Model (e.g. gpt-4o-mini)"
            className="w-36 px-2.5 py-1 rounded-lg border border-slate-200 bg-white text-slate-800 focus:outline-none focus:border-blue-400"
          />
        </div>
      )}

      {/* Scrollable Chat Stream */}
      <div className="flex-1 overflow-y-auto px-4 sm:px-6 pt-6 pb-36">
        <div className="mx-auto max-w-[768px]">
          {isEmpty ? (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, ease: 'easeOut' }}
              className="py-10 sm:py-16 text-center"
            >
              <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-blue-50 text-blue-700 text-xs font-medium border border-blue-100 mb-4">
                <Sparkles className="w-3.5 h-3.5" />
                <span>Grounded RAG with Coordinate Bounding Boxes</span>
              </div>
              <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight font-display">
                ถามคำถามจากเอกสาร PDF ของคุณ
              </h2>
              <p className="mt-2.5 text-sm text-slate-500 max-w-lg mx-auto leading-relaxed">
                ทุกคำตอบมาพร้อม <strong className="text-slate-700">Citation Badge</strong>{' '}
                ที่คลิกเพื่อเปิดหน้า PDF จริงและไฮไลต์กรอบสีเหลืองตรงบรรทัดต้นฉบับทันที
                หรืออัปโหลดไฟล์ PDF ใหม่เพื่อทดสอบสดได้ตลอดเวลา
              </p>

              {/* Loaded Docs Pills */}
              <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
                {docs.map((doc) => (
                  <button
                    key={doc.id}
                    type="button"
                    onClick={onTogglePdfPane}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#F0F4F9] hover:bg-slate-200/70 text-xs text-slate-700 border border-slate-200/60 transition-colors cursor-pointer"
                  >
                    <span className="w-2 h-2 rounded-full bg-emerald-500" />
                    <span className="font-medium">{doc.name}</span>
                    <span className="text-slate-400">({doc.numPages} หน้า)</span>
                  </button>
                ))}
              </div>
            </motion.div>
          ) : (
            <div className="space-y-6">
              {messages.map((msg) => (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.25 }}
                  className={cn(
                    'flex gap-3.5',
                    msg.role === 'user' ? 'justify-end' : 'justify-start'
                  )}
                >
                  {msg.role === 'assistant' && (
                    <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 text-white flex items-center justify-center shrink-0 mt-0.5 shadow-xs">
                      <Bot className="w-4 h-4" />
                    </div>
                  )}

                  <div
                    className={cn(
                      'min-w-0',
                      msg.role === 'user'
                        ? 'max-w-[82%] rounded-2xl rounded-tr-sm px-4 py-3 bg-[#F0F4F9] text-slate-900 text-[14.5px] leading-relaxed border border-slate-200/60'
                        : 'flex-1'
                    )}
                  >
                    {msg.role === 'assistant' ? (
                      <div>
                        {msg.pipelineSteps && (
                          <PipelineStepper steps={msg.pipelineSteps} />
                        )}
                        <MarkdownMessage
                          content={msg.content}
                          citations={msg.citations}
                          activeCitationId={activeCitationId}
                          onSelectCitation={onSelectCitation}
                        />
                      </div>
                    ) : (
                      <p className="whitespace-pre-wrap">{msg.content}</p>
                    )}
                  </div>

                  {msg.role === 'user' && (
                    <div className="w-8 h-8 rounded-xl bg-slate-800 text-white flex items-center justify-center shrink-0 mt-0.5">
                      <User className="w-4 h-4" />
                    </div>
                  )}
                </motion.div>
              ))}

              {isGenerating && (
                <div className="flex items-start gap-3.5">
                  <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 text-white flex items-center justify-center shrink-0">
                    <Loader2 className="w-4 h-4 animate-spin" />
                  </div>
                  <div className="flex-1 rounded-xl border border-slate-200/80 bg-[#F8FAFC] px-4 py-3 text-xs text-slate-600 flex items-center gap-2">
                    <span className="inline-block w-2 h-2 rounded-full bg-blue-500 animate-ping" />
                    <span>
                      กำลังค้นหา Vector &amp; สแกนพิกัด Text Layer ในเอกสาร PDF...
                    </span>
                  </div>
                </div>
              )}
              <div ref={scrollEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* Bottom Floating Input Zone (Fitts's Law + Glassmorphism) */}
      <div className="absolute bottom-0 inset-x-0 pointer-events-none bg-gradient-to-t from-white via-white/95 to-transparent pt-6 pb-4 px-4 sm:px-6">
        <div className="mx-auto max-w-[768px] pointer-events-auto space-y-2.5">
          {/* Preset Prompt Chips */}
          <div className="flex flex-wrap items-center gap-2">
            {PRESET_PROMPTS.map((preset) => (
              <button
                key={preset.id}
                type="button"
                disabled={isGenerating}
                onClick={() => onSendQuery(preset.query)}
                className="group inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs bg-white/90 hover:bg-blue-50/90 text-slate-700 hover:text-blue-700 border border-slate-200/90 hover:border-blue-300 shadow-2xs transition-all cursor-pointer disabled:opacity-50"
              >
                <span className="px-1.5 py-0.5 rounded-full text-[10px] font-semibold bg-slate-100 group-hover:bg-blue-100 text-slate-600 group-hover:text-blue-700">
                  {preset.tag}
                </span>
                <span className="truncate max-w-[260px] sm:max-w-[340px]">
                  {preset.label}
                </span>
              </button>
            ))}
          </div>

          {/* Pill-Shaped Floating Prompt Bar */}
          <form
            onSubmit={handleSubmit}
            className="flex items-center gap-2 rounded-full border border-slate-200/90 bg-white/80 backdrop-blur-md pl-5 pr-2 py-2 shadow-[0_8px_30px_rgb(0,0,0,0.06)] focus-within:border-blue-400 focus-within:ring-2 focus-within:ring-blue-100 transition-all"
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="ถามคำถามจากเอกสาร PDF ที่โหลดอยู่ (รองรับทั้งไทยและอังกฤษ)..."
              className="flex-1 bg-transparent text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none"
            />
            <button
              type="submit"
              disabled={!input.trim() || isGenerating}
              className="flex items-center justify-center w-9 h-9 rounded-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-200 text-white disabled:text-slate-400 transition-colors cursor-pointer shrink-0"
              title="Send Question"
            >
              <ArrowUp className="w-4 h-4" />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
