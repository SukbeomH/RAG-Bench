"use client";

import { useState } from "react";
import { Send } from "lucide-react";
import { askQuestion, type AskResponse, type CitationItem } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: CitationItem[];
}

interface ChatInterfaceProps {
  docId: string | null;
  onCitationClick: (citation: CitationItem) => void;
}

export default function ChatInterface({
  docId,
  onCitationClick,
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !docId) return;

    const userMsg: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const resp: AskResponse = await askQuestion(input, docId);
      const assistantMsg: Message = {
        role: "assistant",
        content: resp.answer,
        citations: resp.citations,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `오류: ${err}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-full flex-col">
      {/* Messages */}
      <div className="flex-1 space-y-4 overflow-y-auto p-4">
        {messages.length === 0 && (
          <div className="py-20 text-center text-gray-400">
            {docId ? "문서에 대해 질문하세요" : "먼저 PDF를 업로드하세요"}
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] rounded-lg px-4 py-2 ${
                msg.role === "user"
                  ? "bg-blue-600 text-white"
                  : "bg-white shadow"
              }`}
            >
              <p className="whitespace-pre-wrap text-sm">{msg.content}</p>
              {msg.citations && msg.citations.length > 0 && (
                <div className="mt-2 space-y-1 border-t pt-2">
                  {msg.citations.map((c, ci) => (
                    <button
                      key={ci}
                      onClick={() => onCitationClick(c)}
                      className="block w-full rounded bg-gray-50 px-2 py-1 text-left text-xs text-blue-600 hover:bg-blue-50"
                    >
                      [{ci + 1}] p.{c.page_number} —{" "}
                      {c.text_snippet.slice(0, 60)}...
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="rounded-lg bg-white px-4 py-2 shadow">
              <span className="animate-pulse text-sm text-gray-400">
                답변 생성 중...
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <form
        onSubmit={handleSubmit}
        className="flex gap-2 border-t bg-white p-4"
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={
            docId ? "질문을 입력하세요..." : "PDF를 먼저 업로드하세요"
          }
          disabled={!docId || loading}
          className="flex-1 rounded-lg border px-3 py-2 text-sm focus:border-blue-500 focus:outline-none disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={!docId || loading || !input.trim()}
          className="rounded-lg bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 disabled:opacity-50"
        >
          <Send size={16} />
        </button>
      </form>
    </div>
  );
}
