import { useCallback, useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type {
  ChatMessage,
  CitationItem,
  LoadedPdfDoc,
} from './types/rag';
import { ChatPane } from './components/ChatPane';
import { PdfViewerPane } from './components/PdfViewerPane';

export type RagMode = 'ai_chat' | 'rag_only';

export function App() {
  const [docs, setDocs] = useState<LoadedPdfDoc[]>([]);
  const [embedderLabel, setEmbedderLabel] = useState<string>('Connecting...');
  const [ragMode, setRagMode] = useState<RagMode>('ai_chat');
  const [llmConfig, setLlmConfig] = useState<{
    apiKey: string;
    baseUrl: string;
    model: string;
  }>({
    apiKey: '',
    baseUrl: '',
    model: '',
  });

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);

  // Split-screen & PDF inspector state
  const [isPdfOpen, setIsPdfOpen] = useState(false);
  const [activeDocId, setActiveDocId] = useState<string>('cpall-2025');
  const [activeCitation, setActiveCitation] = useState<CitationItem | null>(null);
  const [highlightKey, setHighlightKey] = useState(0);

  // Resizable split ratio (default 55% chat / 45% PDF)
  const [chatWidthPct, setChatWidthPct] = useState(55);
  const isDraggingRef = useRef(false);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Load real documents from FastAPI RAG Backend
  useEffect(() => {
    let cancelled = false;

    async function fetchDocs() {
      try {
        const res = await fetch('/api/docs');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (cancelled) return;

        setEmbedderLabel(data.embedder || 'Live RAG Pipeline');
        const loaded: LoadedPdfDoc[] = (data.docs || []).map(
          (d: { id: string; name: string; numPages: number; isCustom?: boolean }) => ({
            id: d.id,
            name: d.name,
            data: new Uint8Array(0), // Lazy-loaded via /api/docs/:id/pdf
            numPages: d.numPages,
            isCustom: d.isCustom,
          })
        );
        setDocs(loaded);
        if (loaded.length > 0) {
          setActiveDocId(loaded[0].id);
        }
      } catch {
        if (!cancelled) {
          setEmbedderLabel('Backend Offline (Start web_demo_server.py)');
        }
      }
    }

    fetchDocs();
    return () => {
      cancelled = true;
    };
  }, []);

  // Resizable divider drag handlers
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDraggingRef.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const nextPct = ((e.clientX - rect.left) / rect.width) * 100;
      setChatWidthPct(Math.min(72, Math.max(35, nextPct)));
    };

    const handleMouseUp = () => {
      if (isDraggingRef.current) {
        isDraggingRef.current = false;
        document.body.style.cursor = '';
      }
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  const handleSelectCitation = useCallback((citation: CitationItem) => {
    setIsPdfOpen(true);
    setActiveDocId(citation.docId);
    setActiveCitation(citation);
    setHighlightKey((k) => k + 1);
  }, []);

  const handleUploadPdf = useCallback(async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    setIsGenerating(true);
    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error(`Upload failed (${res.status})`);
      const meta = await res.json();

      const newDoc: LoadedPdfDoc = {
        id: meta.id,
        name: meta.name,
        data: new Uint8Array(0),
        numPages: meta.numPages,
        isCustom: true,
      };

      setDocs((prev) => [newDoc, ...prev]);
      setActiveDocId(newDoc.id);
      setActiveCitation(null);
      setIsPdfOpen(true);

      setMessages((prev) => [
        ...prev,
        {
          id: `sys-${Date.now()}`,
          role: 'assistant',
          timestamp: new Date().toLocaleTimeString(),
          content: `### นำเข้าและ Index เอกสารจริงสำเร็จ: ${meta.name}

ระบบได้สกัดข้อความด้วย \`PyMuPDF + Thai Normalizer\`, ตัด Chunk ด้วย \`make_chunks\`, และ Index ลง \`Qdrant VectorStore\` จำนวน **${meta.numPages} หน้า** (${meta.chunksCount} Chunks) เรียบร้อยแล้ว สามารถถามคำถามเกี่ยวกับไฟล์นี้ได้ทันที`,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          role: 'assistant',
          timestamp: new Date().toLocaleTimeString(),
          content: `อัปโหลดไฟล์ไม่สำเร็จ: ${err instanceof Error ? err.message : 'Unknown error'}`,
        },
      ]);
    } finally {
      setIsGenerating(false);
    }
  }, []);

  const handleSendQuery = useCallback(async (query: string) => {
    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: query,
      timestamp: new Date().toLocaleTimeString(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setIsGenerating(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          top_k: 3,
          mode: ragMode,
          api_key: llmConfig.apiKey || undefined,
          base_url: llmConfig.baseUrl || undefined,
          model: llmConfig.model || undefined,
        }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      const assistantMsg: ChatMessage = {
        id: `asst-${Date.now()}`,
        role: 'assistant',
        content: data.answer,
        citations: data.citations || [],
        pipelineSteps: data.pipelineSteps || [],
        timestamp: new Date().toLocaleTimeString(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          role: 'assistant',
          timestamp: new Date().toLocaleTimeString(),
          content: `ไม่สามารถเชื่อมต่อ RAG Backend ได้ (${err instanceof Error ? err.message : 'Error'}) กรุณาตรวจสอบว่า \`web_demo_server.py\` กำลังทำงานอยู่ที่พอร์ต 8000`,
        },
      ]);
    } finally {
      setIsGenerating(false);
    }
  }, [ragMode, llmConfig]);

  return (
    <div
      ref={containerRef}
      className="flex h-screen w-screen overflow-hidden bg-white text-slate-900"
    >
      {/* Left / Main Chat Pane */}
      <motion.div
        layout
        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
        style={{
          width: isPdfOpen ? `${chatWidthPct}%` : '100%',
        }}
        className="h-full min-w-[320px] flex flex-col"
      >
        <ChatPane
          messages={messages}
          docs={docs}
          embedderLabel={embedderLabel}
          ragMode={ragMode}
          onChangeRagMode={setRagMode}
          llmConfig={llmConfig}
          onChangeLlmConfig={setLlmConfig}
          isGenerating={isGenerating}
          isPdfOpen={isPdfOpen}
          activeCitationId={activeCitation?.id ?? null}
          onSendQuery={handleSendQuery}
          onSelectCitation={handleSelectCitation}
          onTogglePdfPane={() => setIsPdfOpen((open) => !open)}
          onUploadPdf={handleUploadPdf}
        />
      </motion.div>

      {/* Draggable Resizer + Right PDF Inspector Pane */}
      <AnimatePresence initial={false}>
        {isPdfOpen && (
          <>
            <div
              role="separator"
              aria-orientation="vertical"
              onMouseDown={() => {
                isDraggingRef.current = true;
                document.body.style.cursor = 'col-resize';
              }}
              className="w-1.5 h-full bg-slate-100 hover:bg-blue-400/70 active:bg-blue-500 cursor-col-resize transition-colors flex items-center justify-center shrink-0 z-30 group"
              title="Drag to resize split pane"
            >
              <div className="w-0.5 h-8 rounded-full bg-slate-300 group-hover:bg-white transition-colors" />
            </div>

            <motion.div
              key="pdf-inspector-pane"
              initial={{ opacity: 0, x: 36 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 36 }}
              transition={{ type: 'spring', stiffness: 300, damping: 30 }}
              style={{ width: `${100 - chatWidthPct}%` }}
              className="h-full min-w-[320px]"
            >
              <PdfViewerPane
                docs={docs}
                activeDocId={activeDocId}
                onSelectDoc={(docId) => {
                  setActiveDocId(docId);
                  if (activeCitation && activeCitation.docId !== docId) {
                    setActiveCitation(null);
                  }
                }}
                activeCitation={activeCitation}
                highlightKey={highlightKey}
                onClose={() => setIsPdfOpen(false)}
              />
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
