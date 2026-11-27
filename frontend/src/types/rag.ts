export interface BoundingBoxCoords {
  /** Top offset as percentage of page height (0-100) */
  top: number;
  /** Left offset as percentage of page width (0-100) */
  left: number;
  /** Width as percentage of page width (0-100) */
  width: number;
  /** Height as percentage of page height (0-100) */
  height: number;
}

export interface CitationItem {
  id: string;
  label: string;
  docId: string;
  docName: string;
  page: number;
  coords: BoundingBoxCoords;
  snippet: string;
  similarity: number;
  highlightText?: string;
}

export interface PipelineStep {
  id: string;
  title: string;
  detail: string;
  durationMs: number;
  status: 'done' | 'running' | 'pending';
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  citations?: CitationItem[];
  pipelineSteps?: PipelineStep[];
  isStreaming?: boolean;
}

export interface LoadedPdfDoc {
  id: string;
  name: string;
  data: Uint8Array;
  numPages: number;
  isCustom?: boolean;
}

export interface IndexedParagraph {
  docId: string;
  docName: string;
  page: number;
  text: string;
  coords: BoundingBoxCoords;
}
