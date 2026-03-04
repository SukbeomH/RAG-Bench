"use client";

import { useRef } from "react";
import { Upload, FileText } from "lucide-react";

interface DocumentEntry {
  docId: string;
  fileName: string;
  pageCount: number;
  chunkCount: number;
}

interface SourceSidebarProps {
  documents: DocumentEntry[];
  activeDocId: string | null;
  onDocSelect: (docId: string) => void;
  onUpload: (file: File) => void;
  uploading: boolean;
}

export default function SourceSidebar({
  documents,
  activeDocId,
  onDocSelect,
  onUpload,
  uploading,
}: SourceSidebarProps) {
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onUpload(file);
    if (fileRef.current) fileRef.current.value = "";
  };

  return (
    <div className="flex h-full flex-col border-r bg-gray-50">
      {/* Header */}
      <div className="flex items-center justify-between border-b p-3">
        <h2 className="text-sm font-semibold">문서</h2>
        <button
          onClick={() => fileRef.current?.click()}
          disabled={uploading}
          className="rounded bg-blue-600 px-2 py-1 text-xs text-white hover:bg-blue-700 disabled:opacity-50"
        >
          <Upload size={12} className="mr-1 inline" />
          업로드
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".pdf"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>

      {/* Document list */}
      <div className="flex-1 overflow-y-auto p-2">
        {documents.length === 0 && (
          <div className="py-10 text-center text-xs text-gray-400">
            PDF를 업로드하세요
          </div>
        )}
        {documents.map((doc) => (
          <button
            key={doc.docId}
            onClick={() => onDocSelect(doc.docId)}
            className={`mb-1 flex w-full items-start gap-2 rounded-lg p-2 text-left transition ${
              activeDocId === doc.docId
                ? "bg-blue-50 text-blue-700"
                : "hover:bg-gray-100"
            }`}
          >
            <FileText size={16} className="mt-0.5 shrink-0" />
            <div className="min-w-0">
              <div className="truncate text-xs font-medium">{doc.fileName}</div>
              <div className="text-[10px] text-gray-400">
                {doc.pageCount}p / {doc.chunkCount} chunks
              </div>
            </div>
          </button>
        ))}
      </div>

      {uploading && (
        <div className="border-t p-3 text-center text-xs text-gray-500">
          <span className="animate-pulse">파싱 중...</span>
        </div>
      )}
    </div>
  );
}

export type { DocumentEntry };
