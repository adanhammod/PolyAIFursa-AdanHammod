import type { ChatMessage } from "./types";

const AGENT_URL = process.env.NEXT_PUBLIC_AGENT_URL ?? "http://localhost:8000";

export interface AgentReply {
  response: string;
  annotated_image_base64?: string;
}

export async function sendMessage(messages: ChatMessage[]): Promise<AgentReply> {
  // Each image-processing request is self-contained. Sending conversation history
  // causes the LLM to echo previous assistant turns or re-run earlier operations.
  // Only the most recent user message is needed.
  const lastMsg = messages[messages.length - 1];

  const res = await fetch(`${AGENT_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: lastMsg ? [lastMsg] : [] }),
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
