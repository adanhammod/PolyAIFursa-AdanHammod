import type { ChatMessage } from "./types";

const AGENT_URL = process.env.NEXT_PUBLIC_AGENT_URL ?? "http://localhost:8000";

export interface AgentReply {
  response: string;
  annotated_image_base64?: string;
}

export async function sendMessage(messages: ChatMessage[]): Promise<AgentReply> {
  // Find the index of the most recent user message so we keep its image_base64.
  const lastUserIdx = messages.reduce<number>(
    (last, msg, i) => (msg.role === "user" ? i : last),
    -1,
  );

  // Strip image_base64 from all but the last user message to prevent stale
  // uploads from previous turns bleeding into new requests.
  // Strip annotated_image_base64 from assistant messages — the backend ignores
  // it and it can be a very large base64 payload.
  const payload = messages.map((msg, i) => {
    if (msg.role === "assistant") {
      return { role: msg.role, content: msg.content };
    }
    if (msg.role === "user" && i !== lastUserIdx) {
      return { role: msg.role, content: msg.content };
    }
    return msg;
  });

  const res = await fetch(`${AGENT_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: payload }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || res.statusText);
  }
  const data = await res.json();
  return {
    response: data.response as string,
    annotated_image_base64: data.annotated_image_base64 ?? undefined,
  };
}
