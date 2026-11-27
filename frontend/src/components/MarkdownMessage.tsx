import React from 'react';
import type { CitationItem } from '../types/rag';
import { CitationBadge } from './CitationBadge';

interface MarkdownMessageProps {
  content: string;
  citations?: CitationItem[];
  activeCitationId?: string | null;
  onSelectCitation: (citation: CitationItem) => void;
}

/**
 * Lightweight, zero-dependency Markdown + inline [[cit-id]] badge renderer
 * tailored for structured RAG answers (headings, bullets, bold, inline code).
 */
export function MarkdownMessage({
  content,
  citations = [],
  activeCitationId,
  onSelectCitation,
}: MarkdownMessageProps) {
  const citationMap = new Map(citations.map((c) => [c.id, c]));

  const renderInline = (text: string, keyPrefix: string): React.ReactNode[] => {
    // Split by [[cit-id]], **bold**, and `code`
    const tokenRegex = /(\[\[[^\]]+\]\]|\*\*[^*]+\*\*|`[^`]+`)/g;
    const parts = text.split(tokenRegex);

    return parts.map((part, idx) => {
      const key = `${keyPrefix}-${idx}`;
      if (part.startsWith('[[') && part.endsWith(']]')) {
        const citId = part.slice(2, -2).trim();
        const cit = citationMap.get(citId);
        if (cit) {
          return (
            <CitationBadge
              key={key}
              citation={cit}
              isActive={activeCitationId === cit.id}
              onSelect={onSelectCitation}
            />
          );
        }
        return null;
      }
      if (part.startsWith('**') && part.endsWith('**')) {
        return (
          <strong key={key} className="font-semibold text-slate-900">
            {part.slice(2, -2)}
          </strong>
        );
      }
      if (part.startsWith('`') && part.endsWith('`')) {
        return (
          <code
            key={key}
            className="px-1.5 py-0.5 rounded-md bg-slate-100 text-slate-800 font-mono text-[12px] border border-slate-200/70"
          >
            {part.slice(1, -1)}
          </code>
        );
      }
      return <React.Fragment key={key}>{part}</React.Fragment>;
    });
  };

  const lines = content.split('\n');
  const blocks: React.ReactNode[] = [];
  let listItems: string[] = [];

  const flushList = (keyIdx: number) => {
    if (listItems.length === 0) return;
    blocks.push(
      <ul key={`list-${keyIdx}`} className="my-2.5 space-y-2 pl-1">
        {listItems.map((item, i) => (
          <li key={i} className="flex items-start gap-2.5 text-[14.5px] leading-relaxed text-slate-700">
            <span className="mt-2 w-1.5 h-1.5 rounded-full bg-blue-500 shrink-0" />
            <span className="flex-1">{renderInline(item, `li-${keyIdx}-${i}`)}</span>
          </li>
        ))}
      </ul>
    );
    listItems = [];
  };

  lines.forEach((rawLine, idx) => {
    const line = rawLine.trim();
    if (!line) {
      flushList(idx);
      return;
    }

    if (line.startsWith('### ')) {
      flushList(idx);
      blocks.push(
        <h3
          key={`h3-${idx}`}
          className="text-[15.5px] font-semibold text-slate-900 mt-1 mb-2 tracking-tight"
        >
          {renderInline(line.slice(4), `h3-${idx}`)}
        </h3>
      );
      return;
    }

    if (line.startsWith('- ') || line.startsWith('* ')) {
      listItems.push(line.slice(2));
      return;
    }

    flushList(idx);
    blocks.push(
      <p key={`p-${idx}`} className="text-[14.5px] leading-relaxed text-slate-700 my-2">
        {renderInline(line, `p-${idx}`)}
      </p>
    );
  });

  flushList(lines.length);

  return <div className="space-y-1">{blocks}</div>;
}
