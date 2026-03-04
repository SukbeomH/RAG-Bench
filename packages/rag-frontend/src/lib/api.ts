const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface PageResult {
  page_num: number;
  markdown: string;
  backend: string;
  has_bbox: boolean;
}

export interface ParseResponse {
  doc_id: string;
  pdf_path: string;
  pages: PageResult[];
  total_time_s: number;
  chunk_count: number;
}

export interface CitationItem {
  chunk_id: string;
  source_path: string;
  page_number: number;
  text_snippet: string;
  bbox: number[] | null;
  relevance_score: number;
}

export interface AskResponse {
  answer: string;
  citations: CitationItem[];
}

export async function parsePDF(
  file: File,
  backend = "pymupdf",
): Promise<ParseResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("backend", backend);
  formData.append("chunk", "true");

  const res = await fetch(`${API_BASE}/api/parse`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(`Parse failed: ${res.statusText}`);
  return res.json();
}

export async function askQuestion(
  query: string,
  docId: string,
  k = 5,
): Promise<AskResponse> {
  const res = await fetch(`${API_BASE}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, doc_id: docId, k }),
  });
  if (!res.ok) throw new Error(`Ask failed: ${res.statusText}`);
  return res.json();
}
