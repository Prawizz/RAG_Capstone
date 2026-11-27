import * as Tooltip from '@radix-ui/react-tooltip';
import { FileText, Sparkles } from 'lucide-react';
import type { CitationItem } from '../types/rag';
import { cn } from '../lib/utils';

interface CitationBadgeProps {
  citation: CitationItem;
  isActive: boolean;
  onSelect: (citation: CitationItem) => void;
}

export function CitationBadge({ citation, isActive, onSelect }: CitationBadgeProps) {
  const shortDoc =
    citation.docName.length > 20
      ? `${citation.docName.slice(0, 17)}...pdf`
      : citation.docName;

  return (
    <Tooltip.Provider delayDuration={180}>
      <Tooltip.Root>
        <Tooltip.Trigger asChild>
          <button
            type="button"
            onClick={() => onSelect(citation)}
            className={cn(
              'inline-flex items-center gap-1 px-2 py-0.5 mx-0.5 rounded-full text-[11.5px] font-medium align-baseline transition-all duration-150 cursor-pointer border',
              isActive
                ? 'bg-amber-100 text-amber-950 border-amber-400 ring-2 ring-amber-300/60 shadow-xs'
                : 'bg-[#EEF4FF] text-[#1D4ED8] border-blue-200/80 hover:bg-amber-50 hover:text-amber-900 hover:border-amber-300'
            )}
          >
            <span
              className={cn(
                'inline-flex items-center justify-center w-4 h-4 rounded-full text-[10px] font-bold',
                isActive
                  ? 'bg-amber-500 text-white'
                  : 'bg-blue-600 text-white'
              )}
            >
              {citation.label}
            </span>
            <span>หน้า {citation.page}</span>
            <span className="text-slate-400">•</span>
            <span className="max-w-[120px] truncate">{shortDoc}</span>
          </button>
        </Tooltip.Trigger>

        <Tooltip.Portal>
          <Tooltip.Content
            side="top"
            sideOffset={6}
            className="z-50 max-w-xs rounded-xl border border-slate-200 bg-white/95 backdrop-blur-md p-3 shadow-xl text-left animate-in fade-in-0 zoom-in-95"
          >
            <div className="flex items-center justify-between gap-2 mb-1.5 pb-1.5 border-b border-slate-100">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-800 truncate">
                <FileText className="w-3.5 h-3.5 text-blue-600 shrink-0" />
                <span className="truncate">{citation.docName}</span>
              </div>
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md bg-emerald-50 text-emerald-700 text-[10px] font-semibold shrink-0">
                <Sparkles className="w-2.5 h-2.5" />
                Sim {(citation.similarity * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-[11.5px] leading-relaxed text-slate-600 line-clamp-4">
              “{citation.snippet}”
            </p>
            <div className="mt-2 flex items-center justify-between text-[10.5px] text-blue-600 font-medium">
              <span>หน้าที่ {citation.page}</span>
              <span>คลิกเพื่อชี้เป้าบน PDF →</span>
            </div>
            <Tooltip.Arrow className="fill-white" />
          </Tooltip.Content>
        </Tooltip.Portal>
      </Tooltip.Root>
    </Tooltip.Provider>
  );
}
