"use client";

import { useState, useCallback } from "react";
import SourceSidebar, { type DocumentEntry } from "@/components/SourceSidebar";
import ChatInterface from "@/components/ChatInterface";
import CitationPanel from "@/components/CitationPanel";
import PDFViewer from "@/components/PDFViewer";
import { parsePDF, type CitationItem } from "@/lib/api";

export default function Home() {
  const [documents, setDocuments] = useState<DocumentEntry[]>([]);
  const [activeDocId, setActiveDocId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [citations, setCitations] = useState<CitationItem[]>([]);
  const [activeCitation, setActiveCitation] = useState<CitationItem | null>(
    null,
  );
  const [pdfUrls, setPdfUrls] = useState<Record<string, string>>({});

  const handleUpload = useCallback(async (file: File) => {
    setUploading(true);
    try {
      const resp = await parsePDF(file);

      const entry: DocumentEntry = {
        docId: resp.doc_id,
        fileName: file.name,
        pageCount: resp.pages.length,
        chunkCount: resp.chunk_count,
      };
      setDocuments((prev) => [...prev, entry]);
      setActiveDocId(resp.doc_id);

      // Create object URL for PDF viewer
      const url = URL.createObjectURL(file);
      setPdfUrls((prev) => ({ ...prev, [resp.doc_id]: url }));
    } catch (err) {
      alert(`업로드 실패: ${err}`);
    } finally {
      setUploading(false);
    }
  }, []);

  const handleCitationClick = useCallback((citation: CitationItem) => {
    setActiveCitation(citation);
    setCitations((prev) => {
      // Add to list if not already there
      if (!prev.find((c) => c.chunk_id === citation.chunk_id)) {
        return [...prev, citation];
      }
      return prev;
    });
  }, []);

  return (
    <div className="flex h-screen">
      {/* Source Sidebar */}
      <div className="w-56 shrink-0">
        <SourceSidebar
          documents={documents}
          activeDocId={activeDocId}
          onDocSelect={setActiveDocId}
          onUpload={handleUpload}
          uploading={uploading}
        />
      </div>

      {/* Chat Interface */}
      <div className="flex flex-1 flex-col border-r">
        <div className="border-b bg-white px-4 py-2">
          <h1 className="text-sm font-semibold">AutoRAG Chat</h1>
        </div>
        <div className="flex-1">
          <ChatInterface
            docId={activeDocId}
            onCitationClick={handleCitationClick}
          />
        </div>
      </div>

      {/* Right panel: Citations + PDF */}
      <div className="flex w-[45%] shrink-0 flex-col">
        {/* Citation Panel */}
        <div className="h-48 shrink-0 border-b">
          <CitationPanel
            citations={citations}
            activeCitation={activeCitation}
            onSelect={setActiveCitation}
          />
        </div>

        {/* PDF Viewer */}
        <div className="flex-1">
          <PDFViewer
            pdfUrl={activeDocId ? pdfUrls[activeDocId] || null : null}
            activeCitation={activeCitation}
          />
        </div>
      </div>
    </div>
  );
}
