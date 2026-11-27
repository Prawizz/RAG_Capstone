import { useState } from 'react';
import { CheckCircle2, ChevronDown, Cpu, Layers, Loader2, Sparkles } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import type { PipelineStep } from '../types/rag';
import { cn } from '../lib/utils';

interface PipelineStepperProps {
  steps: PipelineStep[];
}

export function PipelineStepper({ steps }: PipelineStepperProps) {
  const [expanded, setExpanded] = useState(false);

  if (!steps || steps.length === 0) return null;

  const totalMs = steps.reduce((acc, s) => acc + s.durationMs, 0);
  const allDone = steps.every((s) => s.status === 'done');

  return (
    <div className="mb-3 rounded-xl border border-slate-200/80 bg-[#F8FAFC] overflow-hidden">
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        className="w-full flex items-center justify-between px-3.5 py-2 text-left hover:bg-slate-100/70 transition-colors cursor-pointer"
      >
        <div className="flex items-center gap-2 min-w-0">
          <div className="flex items-center justify-center w-5 h-5 rounded-full bg-blue-50 text-blue-600 shrink-0">
            {allDone ? (
              <Sparkles className="w-3 h-3" />
            ) : (
              <Loader2 className="w-3 h-3 animate-spin" />
            )}
          </div>
          <span className="text-xs font-semibold text-slate-700 truncate">
            RAG Retrieval Pipeline
          </span>
          <span className="hidden sm:inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200/60">
            {steps.length} stages • {totalMs} ms
          </span>
        </div>

        <div className="flex items-center gap-1.5 text-[11px] font-medium text-slate-500 shrink-0">
          <span>{expanded ? 'ซ่อนรายละเอียด' : 'ดูขั้นตอน'}</span>
          <ChevronDown
            className={cn(
              'w-3.5 h-3.5 transition-transform duration-200',
              expanded && 'rotate-180'
            )}
          />
        </div>
      </button>

      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
            className="overflow-hidden border-t border-slate-200/70"
          >
            <div className="px-3.5 py-2.5 space-y-2">
              {steps.map((step, idx) => (
                <div key={step.id} className="flex items-start gap-2.5">
                  <div className="mt-0.5 shrink-0">
                    {step.status === 'done' ? (
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                    ) : idx === 0 ? (
                      <Cpu className="w-3.5 h-3.5 text-blue-500 animate-pulse" />
                    ) : (
                      <Layers className="w-3.5 h-3.5 text-slate-400" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-medium text-slate-800">
                        {step.title}
                      </span>
                      <span className="text-[11px] tabular-nums text-slate-400 shrink-0">
                        {step.durationMs} ms
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-500 leading-relaxed">
                      {step.detail}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
