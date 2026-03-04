"use client";

import { useEffect, useRef, useState } from "react";
import type { CitationItem } from "@/lib/api";

interface PDFViewerProps {
  pdfUrl: string | null;
  activeCitation: CitationItem | null;
}

export default function PDFViewer({ pdfUrl, activeCitation }: PDFViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(0);
  const [pdfDoc, setPdfDoc] = useState<any>(null);

  // Load PDF
  useEffect(() => {
    if (!pdfUrl) return;

    let cancelled = false;

    async function loadPdf() {
      const pdfjsLib = await import("pdfjs-dist");
      pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.mjs`;

      const doc = await pdfjsLib.getDocument(pdfUrl!).promise;
      if (!cancelled) {
        setPdfDoc(doc);
        setTotalPages(doc.numPages);
        setCurrentPage(1);
      }
    }

    loadPdf();
    return () => {
      cancelled = true;
    };
  }, [pdfUrl]);

  // Render page
  useEffect(() => {
    if (!pdfDoc || !canvasRef.current) return;

    async function renderPage() {
      const page = await pdfDoc.getPage(currentPage);
      const scale = 1.5;
      const viewport = page.getViewport({ scale });

      const canvas = canvasRef.current!;
      const context = canvas.getContext("2d")!;
      canvas.width = viewport.width;
      canvas.height = viewport.height;

      await page.render({ canvasContext: context, viewport }).promise;

      // Draw bbox highlight if citation is on this page
      if (activeCitation?.bbox && activeCitation.page_number === currentPage) {
        const [x0, y0, x1, y1] = activeCitation.bbox;
        context.strokeStyle = "rgba(59, 130, 246, 0.8)";
        context.lineWidth = 2;
        context.fillStyle = "rgba(59, 130, 246, 0.15)";
        const sx = x0 * scale;
        const sy = y0 * scale;
        const sw = (x1 - x0) * scale;
        const sh = (y1 - y0) * scale;
        context.fillRect(sx, sy, sw, sh);
        context.strokeRect(sx, sy, sw, sh);
      }
    }

    renderPage();
  }, [pdfDoc, currentPage, activeCitation]);

  // Navigate to citation page
  useEffect(() => {
    if (activeCitation?.page_number) {
      setCurrentPage(activeCitation.page_number);
    }
  }, [activeCitation]);

  if (!pdfUrl) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-gray-400">
        PDF를 업로드하면 여기에 표시됩니다
      </div>
    );
  }

  return (
    <div ref={containerRef} className="flex h-full flex-col">
      {/* Navigation */}
      <div className="flex items-center justify-between border-b bg-white px-4 py-2">
        <button
          onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
          disabled={currentPage <= 1}
          className="rounded px-2 py-1 text-sm hover:bg-gray-100 disabled:opacity-30"
        >
          이전
        </button>
        <span className="text-xs text-gray-500">
          {currentPage} / {totalPages}
        </span>
        <button
          onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
          disabled={currentPage >= totalPages}
          className="rounded px-2 py-1 text-sm hover:bg-gray-100 disabled:opacity-30"
        >
          다음
        </button>
      </div>

      {/* Canvas */}
      <div className="flex-1 overflow-auto bg-gray-100 p-4">
        <canvas ref={canvasRef} className="mx-auto shadow-lg" />
      </div>
    </div>
  );
}
