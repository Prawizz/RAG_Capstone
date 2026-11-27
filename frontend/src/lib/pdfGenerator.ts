import { PDFDocument, StandardFonts, rgb, type PDFFont, type PDFPage } from 'pdf-lib';
import type { BoundingBoxCoords } from '../types/rag';

export interface GeneratedDemoPdfs {
  archPdf: Uint8Array;
  govPdf: Uint8Array;
  /** Exact normalized bounding boxes computed during PDF generation */
  knownCoords: Record<string, BoundingBoxCoords>;
}

interface DrawBlockOptions {
  page: PDFPage;
  font: PDFFont;
  boldFont: PDFFont;
  pageWidth: number;
  pageHeight: number;
  x: number;
  y: number;
  maxWidth: number;
  title?: string;
  lines: string[];
  fontSize?: number;
  lineHeight?: number;
}

/**
 * Draws a structured section block onto a PDF page and returns:
 * - nextY: the Y coordinate after the block
 * - bbox: normalized percentage [top, left, width, height] for exact citation overlay
 */
function drawSectionBlock(opts: DrawBlockOptions): { nextY: number; bbox: BoundingBoxCoords } {
  const {
    page,
    font,
    boldFont,
    pageWidth,
    pageHeight,
    x,
    maxWidth,
    title,
    lines,
    fontSize = 10.5,
    lineHeight = 15,
  } = opts;

  let cursorY = opts.y;
  const startTopPdf = cursorY + (title ? 14 : 10);

  if (title) {
    page.drawText(title, {
      x,
      y: cursorY,
      size: 12,
      font: boldFont,
      color: rgb(0.09, 0.16, 0.31),
    });
    cursorY -= 20;
  }

  // Word-wrap each paragraph line so text stays cleanly within maxWidth
  const wrappedLines: string[] = [];
  for (const rawLine of lines) {
    const words = rawLine.split(' ');
    let current = '';
    for (const word of words) {
      const candidate = current ? `${current} ${word}` : word;
      if (font.widthOfTextAtSize(candidate, fontSize) > maxWidth && current) {
        wrappedLines.push(current);
        current = word;
      } else {
        current = candidate;
      }
    }
    if (current) wrappedLines.push(current);
  }

  for (const line of wrappedLines) {
    page.drawText(line, {
      x,
      y: cursorY,
      size: fontSize,
      font,
      color: rgb(0.15, 0.2, 0.28),
    });
    cursorY -= lineHeight;
  }

  const endBottomPdf = cursorY + lineHeight - 5;

  // Convert PDF bottom-left coordinates into normalized top-left percentage coordinates
  const topPct = ((pageHeight - startTopPdf) / pageHeight) * 100;
  const heightPct = ((startTopPdf - endBottomPdf) / pageHeight) * 100;
  const leftPct = ((x - 8) / pageWidth) * 100;
  const widthPct = ((maxWidth + 16) / pageWidth) * 100;

  return {
    nextY: cursorY - 14,
    bbox: {
      top: Number(Math.max(0, topPct).toFixed(2)),
      left: Number(Math.max(0, leftPct).toFixed(2)),
      width: Number(Math.min(96, widthPct).toFixed(2)),
      height: Number(Math.max(3, heightPct).toFixed(2)),
    },
  };
}

function drawPageChrome(
  page: PDFPage,
  font: PDFFont,
  boldFont: PDFFont,
  docTitle: string,
  pageNum: number,
  totalPages: number
) {
  const { width, height } = page.getSize();

  // Header bar
  page.drawRectangle({
    x: 48,
    y: height - 52,
    width: width - 96,
    height: 1,
    color: rgb(0.84, 0.87, 0.91),
  });
  page.drawText(docTitle, {
    x: 48,
    y: height - 42,
    size: 8.5,
    font: boldFont,
    color: rgb(0.39, 0.45, 0.55),
  });
  page.drawText('CONFIDENTIAL - ACADEMIC DEMO', {
    x: width - 195,
    y: height - 42,
    size: 8,
    font,
    color: rgb(0.55, 0.6, 0.68),
  });

  // Footer
  page.drawRectangle({
    x: 48,
    y: 44,
    width: width - 96,
    height: 0.75,
    color: rgb(0.88, 0.9, 0.93),
  });
  page.drawText(`Page ${pageNum} of ${totalPages}`, {
    x: width - 102,
    y: 28,
    size: 8.5,
    font,
    color: rgb(0.45, 0.5, 0.58),
  });
  page.drawText('Enterprise Domain-Agnostic RAG System (2026)', {
    x: 48,
    y: 28,
    size: 8.5,
    font,
    color: rgb(0.45, 0.5, 0.58),
  });
}

/**
 * Generates two real, multi-page, selectable PDF documents in-memory using pdf-lib
 * so the prototype works 100% offline with zero broken links.
 */
export async function generateDemoPdfs(): Promise<GeneratedDemoPdfs> {
  const knownCoords: Record<string, BoundingBoxCoords> = {};

  // =========================================================================
  // Document 1: Enterprise_RAG_Architecture_2026.pdf (4 pages)
  // =========================================================================
  const archDoc = await PDFDocument.create();
  const font = await archDoc.embedFont(StandardFonts.Helvetica);
  const boldFont = await archDoc.embedFont(StandardFonts.HelveticaBold);
  const W = 595.28; // A4 width
  const H = 841.89; // A4 height

  // Page 1
  const p1 = archDoc.addPage([W, H]);
  drawPageChrome(p1, font, boldFont, 'Enterprise_RAG_Architecture_2026.pdf', 1, 4);

  p1.drawText('Enterprise Domain-Agnostic RAG System', {
    x: 48,
    y: H - 96,
    size: 20,
    font: boldFont,
    color: rgb(0.06, 0.12, 0.25),
  });
  p1.drawText('Technical Architecture, Hybrid Retrieval & Citation Grounding Whitepaper', {
    x: 48,
    y: H - 118,
    size: 11,
    font,
    color: rgb(0.32, 0.4, 0.52),
  });

  let y = H - 160;
  const s1 = drawSectionBlock({
    page: p1,
    font,
    boldFont,
    pageWidth: W,
    pageHeight: H,
    x: 48,
    y,
    maxWidth: W - 96,
    title: '1.1 Hybrid Retrieval & BGE-M3 Multi-Vector Indexing',
    lines: [
      'Our retrieval engine utilizes BAAI/bge-m3 to generate three complementary vector representations per document chunk: dense semantic embeddings (1024-dim), learned sparse lexical weights (sparse_bge), and BM25 statistical term vectors.',
      'During query execution, Qdrant performs Reciprocal Rank Fusion (RRF) across dense and sparse_bge channels with a default candidate pool of Top-30 chunks before downstream reranking. This hybrid fusion improves Thai-English cross-lingual recall by 24.6% compared to dense-only search.',
    ],
  });
  knownCoords['cit-1'] = s1.bbox;
  y = s1.nextY;

  const s2 = drawSectionBlock({
    page: p1,
    font,
    boldFont,
    pageWidth: W,
    pageHeight: H,
    x: 48,
    y,
    maxWidth: W - 96,
    title: '1.2 Sentence-Boundary Chunking & Coordinate Provenance',
    lines: [
      'Documents are segmented along natural sentence boundaries using CHUNK_TARGET_CHARS = 1000 with a 1-sentence sliding overlap. Each chunk preserves exact character offsets and normalized PDF bounding box coordinates [top, left, width, height].',
      'Preserving layout provenance enables deterministic visual verification: every generated claim links directly to its exact paragraph coordinates on the rendered PDF canvas.',
    ],
  });
  knownCoords['cit-1b'] = s2.bbox;
  y = s2.nextY;

  drawSectionBlock({
    page: p1,
    font,
    boldFont,
    pageWidth: W,
    pageHeight: H,
    x: 48,
    y,
    maxWidth: W - 96,
    title: '1.3 Query Expansion & Salience-Weighted Reranking',
    lines: [
      'Before vector lookup, the Query Expansion module generates domain-normalized synonyms and bilingual Thai-English variants. Retrieved chunks are then scored by a Cross-Encoder Reranker combined with Salience Weighting to prioritize primary evidence over boilerplate headers.',
    ],
  });

  // Page 2
  const p2 = archDoc.addPage([W, H]);
  drawPageChrome(p2, font, boldFont, 'Enterprise_RAG_Architecture_2026.pdf', 2, 4);
  y = H - 96;

  const s3 = drawSectionBlock({
    page: p2,
    font,
    boldFont,
    pageWidth: W,
    pageHeight: H,
    x: 48,
    y,
    maxWidth: W - 96,
    title: '2.1 Consistency Detector & Hallucination Mitigation',
    lines: [
      'Every candidate response passes through our Consistency Detector before reaching the user. The detector decomposes the generated answer into atomic claims and verifies entailment against the retrieved source chunks.',
      'Claims lacking sufficient grounding score (entailment < 0.72) are flagged or abstained automatically, reducing unsupported hallucinations to under 1.8% across our 300-question Thai enterprise benchmark.',
    ],
  });
  knownCoords['cit-2'] = s3.bbox;
  y = s3.nextY;

  drawSectionBlock({
    page: p2,
    font,
    boldFont,
    pageWidth: W,
    pageHeight: H,
    x: 48,
    y,
    maxWidth: W - 96,
    title: '2.2 Dual-Stage Access Control (ACL) Enforcement',
    lines: [
      'Security is enforced at two independent layers: (1) Qdrant pre-filtering restricts vector search to collections and document IDs matching the principal department and clearance level; (2) Post-retrieval ACL re-verification checks the live permission store prior to LLM context injection.',
    ],
  });

  // Page 3
  const p3 = archDoc.addPage([W, H]);
  drawPageChrome(p3, font, boldFont, 'Enterprise_RAG_Architecture_2026.pdf', 3, 4);
  y = H - 96;

  const s4 = drawSectionBlock({
    page: p3,
    font,
    boldFont,
    pageWidth: W,
    pageHeight: H,
    x: 48,
    y,
    maxWidth: W - 96,
    title: '3.1 Evaluation Benchmark Results (Thai Enterprise Corpus)',
    lines: [
      'Evaluated on 300 curated questions spanning CPALL (546 pages), PTT (464 pages), and KBANK (180 OCR pages): Hybrid RRF + Salience Reranking achieved 91.4% Recall@5 and 0.89 Mean Reciprocal Rank (MRR), outperforming baseline dense retrieval (76.2% Recall@5).',
      'End-to-end citation precision reached 94.1% when verified against raw PDF ground-truth page numbers and bounding-box regions.',
    ],
  });
  knownCoords['cit-3'] = s4.bbox;

  // Page 4
  const p4 = archDoc.addPage([W, H]);
  drawPageChrome(p4, font, boldFont, 'Enterprise_RAG_Architecture_2026.pdf', 4, 4);
  drawSectionBlock({
    page: p4,
    font,
    boldFont,
    pageWidth: W,
    pageHeight: H,
    x: 48,
    y: H - 96,
    maxWidth: W - 96,
    title: '4.1 Production Deployment & Latency Budget',
    lines: [
      'Total P95 retrieval latency is 185 ms on an NVIDIA RTX-class GPU with BGE-M3 in bf16 precision: 22 ms for query embedding, 38 ms for Qdrant RRF search, and 125 ms for salience reranking and citation alignment.',
    ],
  });

  const archPdf = await archDoc.save();

  // =========================================================================
  // Document 2: AI_Governance_Compliance_2026.pdf (2 pages)
  // =========================================================================
  const govDoc = await PDFDocument.create();
  const gFont = await govDoc.embedFont(StandardFonts.Helvetica);
  const gBold = await govDoc.embedFont(StandardFonts.HelveticaBold);

  const gp1 = govDoc.addPage([W, H]);
  drawPageChrome(gp1, gFont, gBold, 'AI_Governance_Compliance_2026.pdf', 1, 2);
  gp1.drawText('Enterprise AI Governance & Audit Policy', {
    x: 48,
    y: H - 96,
    size: 19,
    font: gBold,
    color: rgb(0.06, 0.12, 0.25),
  });

  const gs1 = drawSectionBlock({
    page: gp1,
    font: gFont,
    boldFont: gBold,
    pageWidth: W,
    pageHeight: H,
    x: 48,
    y: H - 146,
    maxWidth: W - 96,
    title: 'Section 1: Mandatory Grounding & Auditability',
    lines: [
      'All high-stakes enterprise AI responses in financial, legal, and regulatory workflows must include verifiable inline citations pointing to the exact source document and page coordinates.',
      'Unverifiable claims or out-of-scope queries must trigger a transparent abstention message rather than speculative generation.',
    ],
  });
  knownCoords['cit-4'] = gs1.bbox;

  const gp2 = govDoc.addPage([W, H]);
  drawPageChrome(gp2, gFont, gBold, 'AI_Governance_Compliance_2026.pdf', 2, 2);
  drawSectionBlock({
    page: gp2,
    font: gFont,
    boldFont: gBold,
    pageWidth: W,
    pageHeight: H,
    x: 48,
    y: H - 96,
    maxWidth: W - 96,
    title: 'Section 2: Data Privacy & Zero-Leakage Guarantee',
    lines: [
      'Cross-department document retrieval is strictly prohibited unless explicit read permissions are verified at both index-filter time and response-assembly time. Unauthorized queries must return the exact same empty-evidence state as non-existent documents.',
    ],
  });

  const govPdf = await govDoc.save();

  return { archPdf, govPdf, knownCoords };
}
