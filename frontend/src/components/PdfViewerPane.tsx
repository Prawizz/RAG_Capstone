import { useCallback, useEffect, useRef, useState } from 'react';
import * as pdfjsLib from 'pdfjs-dist';
import type { PDFDocumentProxy } from 'pdfjs-dist';

// Configure PDF.js worker for Vite
pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url
).toString();
import {
  ChevronLeft,
  ChevronRight,
  FileText,
  Maximize2,
  Minus,
  Plus,
  Sparkles,
  X,
} from 'lucide-react';
import type { CitationItem, LoadedPdfDoc } from '../types/rag';
import { cn } from '../lib/utils';

interface PdfViewerPaneProps {
  docs: LoadedPdfDoc[];
  activeDocId: string;
  onSelectDoc: (docId: string) => void;
  activeCitation: CitationItem | null;
  highlightKey: number;
  onClose: () => void;
}

export function PdfViewerPane({
  docs,
  activeDocId,
  onSelectDoc,
  activeCitation,
  highlightKey,
  onClose,
}: PdfViewerPaneProps) {
  const activeDoc = docs.find((d) => d.id === activeDocId) ?? docs[0];
  const totalPages = activeDoc?.numPages ?? 1;

  const [currentPage, setCurrentPage] = useState<number>(() =>
    activeCitation && activeCitation.docId === activeDoc?.id ? activeCitation.page : 1
  );
  const [pdfProxy, setPdfProxy] = useState<PDFDocumentProxy | null>(null);
  const [isLoadingPage, setIsLoadingPage] = useState<boolean>(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [scale, setScale] = useState<number>(1.05);
  const [dimensions, setDimensions] = useState<{ width: number; height: number }>({
    width: 595 * 1.05,
    height: 842 * 1.05,
  });

  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const textLayerRef = useRef<HTMLDivElement | null>(null);
  const bboxRef = useRef<HTMLDivElement | null>(null);

  // Sync currentPage when user clicks a citation
  useEffect(() => {
    if (activeCitation && activeCitation.docId === activeDoc?.id) {
      setCurrentPage(activeCitation.page);
    } else if (activeDoc) {
      setCurrentPage((prev) => Math.min(Math.max(1, prev), activeDoc.numPages));
    }
  }, [activeCitation, activeDoc?.id, activeDoc?.numPages]);

  // Fetch ONLY the single cited/current page from backend (/api/docs/:id/page/:page/pdf)
  useEffect(() => {
    if (!activeDoc) return;
    let cancelled = false;
    let loadedPdf: PDFDocumentProxy | null = null;

    async function loadSinglePage() {
      setIsLoadingPage(true);
      setLoadError(null);
      setPdfProxy(null);

      try {
        const source =
          activeDoc.data && activeDoc.data.length > 0
            ? { data: activeDoc.data.slice() }
            : {
                url: `/api/docs/${encodeURIComponent(activeDoc.id)}/page/${currentPage}/pdf`,
              };
        const task = pdfjsLib.getDocument(source);
        const pdf = await task.promise;
        if (cancelled) {
          await task.destroy();
          return;
        }
        loadedPdf = pdf;
        setPdfProxy(pdf);
      } catch (err) {
        if (!cancelled) {
          setLoadError(err instanceof Error ? err.message : 'Failed to load PDF page');
        }
      } finally {
        if (!cancelled) setIsLoadingPage(false);
      }
    }

    loadSinglePage();
    return () => {
      cancelled = true;
      if (loadedPdf) {
        void loadedPdf.cleanup();
      }
    };
  }, [activeDoc?.id, currentPage]);

  // Render the loaded single page onto canvas + selectable TextLayer
  useEffect(() => {
    if (!pdfProxy) return;
    let cancelled = false;
    let renderTask: { cancel: () => void; promise: Promise<void> } | null = null;

    async function render() {
      // Single-page endpoint returns a 1-page PDF (index 1); full-data fallback uses currentPage
      const targetIndex = pdfProxy!.numPages === 1 ? 1 : Math.min(currentPage, pdfProxy!.numPages);
      const page = await pdfProxy!.getPage(targetIndex);
      if (cancelled) return;

      const viewport = page.getViewport({ scale });
      setDimensions({ width: viewport.width, height: viewport.height });

      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
      canvas.width = Math.floor(viewport.width * dpr);
      canvas.height = Math.floor(viewport.height * dpr);
      canvas.style.width = `${viewport.width}px`;
      canvas.style.height = `${viewport.height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      renderTask = page.render({ canvasContext: ctx, viewport, canvas });
      try {
        await renderTask.promise;
      } catch {
        return;
      }
      if (cancelled) return;

      const textContainer = textLayerRef.current;
      if (textContainer) {
        textContainer.innerHTML = '';
        textContainer.style.setProperty('--scale-factor', String(scale));
        const textContent = await page.getTextContent();
        if (cancelled) return;
        try {
          const textLayer = new pdfjsLib.TextLayer({
            textContentSource: textContent,
            container: textContainer,
            viewport,
          });
          await textLayer.render();
        } catch {
          // Graceful fallback
        }
      }

      // Smooth center-scroll to the bounding box once the page has rendered
      if (
        !cancelled &&
        activeCitation &&
        activeCitation.docId === activeDoc?.id &&
        activeCitation.page === currentPage &&
        bboxRef.current
      ) {
        window.setTimeout(() => {
          bboxRef.current?.scrollIntoView({
            behavior: 'smooth',
            block: 'center',
            inline: 'nearest',
          });
        }, 80);
      }
    }

    render();
    return () => {
      cancelled = true;
      if (renderTask) {
        try {
          renderTask.cancel();
        } catch {
          // ignore
        }
      }
    };
  }, [pdfProxy, scale, currentPage, activeCitation, activeDoc?.id, highlightKey]);

  const handleFitToWidth = useCallback(async () => {
    if (!pdfProxy || !scrollContainerRef.current) return;
    const targetIndex = pdfProxy.numPages === 1 ? 1 : Math.min(currentPage, pdfProxy.numPages);
    const page = await pdfProxy.getPage(targetIndex);
    const viewport = page.getViewport({ scale: 1 });
    const availableWidth = scrollContainerRef.current.clientWidth - 48;
    const nextScale = Math.max(0.5, Math.min(2.2, availableWidth / viewport.width));
    setScale(Number(nextScale.toFixed(2)));
  }, [pdfProxy, currentPage]);

  const isCitedPage =
    activeCitation !== null &&
    activeCitation.docId === activeDoc?.id &&
    activeCitation.page === currentPage;

  return (
    <div className="flex flex-col h-full bg-[#F3F4F6] border-l border-slate-200/90 select-none">
      {/* Document Switcher Tabs */}
      {docs.length > 1 && (
        <div className="flex items-center gap-1 px-3 pt-2 bg-white border-b border-slate-200/80 overflow-x-auto">
          {docs.map((doc) => {
            const isSelected = doc.id === activeDoc?.id;
            return (
              <button
                key={doc.id}
                type="button"
                onClick={() => {
                  onSelectDoc(doc.id);
                  setCurrentPage(1);
                }}
                className={cn(
                  'group flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-t-lg border-b-2 transition-colors whitespace-nowrap cursor-pointer',
                  isSelected
                    ? 'border-blue-600 text-blue-700 bg-blue-50/50'
                    : 'border-transparent text-slate-500 hover:text-slate-800 hover:bg-slate-50'
                )}
              >
                <FileText className="w-3.5 h-3.5 shrink-0" />
                <span className="max-w-[160px] truncate">{doc.name}</span>
                {doc.isCustom && (
                  <span className="px-1.5 py-0.2 rounded-full text-[9.5px] bg-violet-100 text-violet-700 font-semibold">
                    Custom
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}

      {/* PDF Viewer Toolbar */}
      <div className="flex items-center justify-between gap-2 px-3 py-2 bg-white/95 backdrop-blur-sm border-b border-slate-200/80 shadow-2xs z-20">
        {/* Left: Page Nav (loads 1 page on demand) */}
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            disabled={currentPage <= 1}
            className="p-1.5 rounded-lg text-slate-600 hover:bg-slate-100 disabled:opacity-35 disabled:pointer-events-none cursor-pointer transition-colors"
            title="Previous Page"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <span className="text-xs font-medium text-slate-700 tabular-nums px-1.5">
            Page <strong className="font-semibold text-slate-900">{currentPage}</strong> of{' '}
            {totalPages}
          </span>
          <button
            type="button"
            onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
            disabled={currentPage >= totalPages}
            className="p-1.5 rounded-lg text-slate-600 hover:bg-slate-100 disabled:opacity-35 disabled:pointer-events-none cursor-pointer transition-colors"
            title="Next Page"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
          <span className="hidden lg:inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium bg-amber-50 text-amber-700 border border-amber-200/70 ml-1">
            Cited Page Only
          </span>
        </div>

        {/* Center: Zoom Controls */}
        <div className="flex items-center gap-1 bg-slate-100/80 rounded-lg p-0.5 border border-slate-200/70">
          <button
            type="button"
            onClick={() => setScale((s) => Math.max(0.5, Number((s - 0.15).toFixed(2))))}
            className="p-1 rounded-md text-slate-600 hover:bg-white hover:shadow-2xs cursor-pointer transition-all"
            title="Zoom Out"
          >
            <Minus className="w-3.5 h-3.5" />
          </button>
          <span className="text-[11.5px] font-semibold text-slate-700 tabular-nums w-11 text-center">
            {Math.round(scale * 100)}%
          </span>
          <button
            type="button"
            onClick={() => setScale((s) => Math.min(2.4, Number((s + 0.15).toFixed(2))))}
            className="p-1 rounded-md text-slate-600 hover:bg-white hover:shadow-2xs cursor-pointer transition-all"
            title="Zoom In"
          >
            <Plus className="w-3.5 h-3.5" />
          </button>
          <div className="w-px h-3.5 bg-slate-300 mx-0.5" />
          <button
            type="button"
            onClick={handleFitToWidth}
            className="flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium text-slate-600 hover:bg-white hover:text-slate-900 hover:shadow-2xs cursor-pointer transition-all"
            title="Fit to Width"
          >
            <Maximize2 className="w-3 h-3" />
            <span className="hidden xl:inline">Fit</span>
          </button>
        </div>

        {/* Right: Close Button */}
        <button
          type="button"
          onClick={onClose}
          className="p-1.5 rounded-lg text-slate-500 hover:text-slate-800 hover:bg-slate-100 cursor-pointer transition-colors"
          title="Close PDF Inspector"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Single-Page PDF Canvas Viewport */}
      <div
        ref={scrollContainerRef}
        className="flex-1 overflow-auto px-4 py-4 bg-[#E5E7EB]/85 relative"
      >
        {loadError ? (
          <div className="flex items-center justify-center h-full text-sm text-red-600 px-4 text-center">
            ไม่สามารถแสดงหน้า PDF ได้: {loadError}
          </div>
        ) : isLoadingPage && !pdfProxy ? (
          <div className="flex items-center justify-center h-full text-sm text-slate-500">
            กำลังโหลดหน้า {currentPage} จาก {activeDoc?.name}...
          </div>
        ) : (
          <div
            className="relative mx-auto bg-white shadow-[0_4px_20px_rgba(15,23,42,0.10)] ring-1 ring-slate-900/5 rounded-xs overflow-hidden"
            style={{ width: dimensions.width, height: dimensions.height }}
          >
            <canvas ref={canvasRef} className="block select-none" />
            <div ref={textLayerRef} className="pdf-text-layer" />

            {/* Coordinate-Based Normalized Bounding Box Overlay */}
            {isCitedPage && activeCitation && (
              <div
                key={`${activeCitation.id}-${highlightKey}`}
                ref={bboxRef}
                className="absolute z-10 pointer-events-none rounded-md animate-bbox-pulse"
                style={{
                  top: `${activeCitation.coords.top}%`,
                  left: `${activeCitation.coords.left}%`,
                  width: `${activeCitation.coords.width}%`,
                  height: `${activeCitation.coords.height}%`,
                  backgroundColor: 'rgba(250, 204, 21, 0.35)',
                  border: '2px solid #EAB308',
                }}
              >
                <div className="-top-6 left-0 absolute inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-amber-500 text-white text-[10.5px] font-semibold shadow-sm whitespace-nowrap">
                  <Sparkles className="w-2.5 h-2.5" />
                  <span>Citation [{activeCitation.label}]</span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Bottom Active Citation Status Strip */}
      {activeCitation && (
        <div className="px-3.5 py-2 bg-white border-t border-slate-200 flex items-center justify-between gap-2 text-xs">
          <div className="flex items-center gap-2 min-w-0">
            <span className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-amber-500 text-white text-[10px] font-bold shrink-0">
              {activeCitation.label}
            </span>
            <span className="text-slate-700 font-medium truncate">
              ชี้เป้าพิกัดหน้า {activeCitation.page} • {activeCitation.docName}
            </span>
          </div>
          <span className="text-[11px] font-mono text-slate-400 shrink-0">
            [{activeCitation.coords.left.toFixed(0)}%, {activeCitation.coords.top.toFixed(0)}%]
          </span>
        </div>
      )}
    </div>
  );
}
