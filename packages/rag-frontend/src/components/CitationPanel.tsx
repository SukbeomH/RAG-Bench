"use client";

import { FileText } from "lucide-react";
import type { CitationItem } from "@/lib/api";

interface CitationPanelProps {
  citations: CitationItem[];
  activeCitation: CitationItem | null;
  onSelect: (citation: CitationItem) => void;
}

export default function CitationPanel({
  citations,
  activeCitation,
  onSelect,
}: CitationPanelProps) {
  if (citations.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-gray-400">
        질문을 하면 출처가 여기에 표시됩니다
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-3">
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-500">
        출처 ({citations.length})
      </h3>
      <div className="space-y-2">
        {citations.map((c, i) => {
          const isActive = activeCitation?.chunk_id === c.chunk_id;
          return (
            <button
              key={c.chunk_id}
              onClick={() => onSelect(c)}
              className={`w-full rounded-lg border p-3 text-left transition ${
                isActive
                  ? "border-blue-500 bg-blue-50"
                  : "border-gray-200 bg-white hover:border-blue-300"
              }`}
            >
              <div className="mb-1 flex items-center gap-2">
                <FileText size={14} className="text-gray-400" />
                <span className="text-xs font-medium text-gray-700">
                  [{i + 1}] 페이지 {c.page_number}
                </span>
                {c.bbox && (
                  <span className="rounded bg-green-100 px-1 text-[10px] text-green-700">
                    bbox
                  </span>
                )}
              </div>
              <p className="line-clamp-3 text-xs text-gray-600">
                {c.text_snippet}
              </p>
              <div className="mt-1 text-[10px] text-gray-400">
                {c.source_path} / {c.chunk_id}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
