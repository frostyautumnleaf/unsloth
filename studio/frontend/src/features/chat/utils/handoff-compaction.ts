// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Handoff compaction: at the threshold the model writes a <handoff_note> that the checkpoint reset
 *  carries in place of the deterministic carried_forward block.
 *
 *  The reset itself is the server's (see auto-compaction.ts); only the note is model-written, so it
 *  needs a round-trip the stateless fitter cannot make. That request happens here, ahead of the real
 *  turn, and its result rides along on that turn as `handoffNote`. Nothing is published to the thread:
 *  the note request is a side call, never a message in the chat box. */

import { streamChatCompletions } from "../api/chat-api";
import type { OpenAIChatCompletionsRequest, OpenAIChatMessage } from "../types/api";
import { extractDeltaText } from "./parse-assistant-content";

export const HANDOFF_NOTE_OPEN = "<handoff_note>";
export const HANDOFF_NOTE_CLOSE = "</handoff_note>";

/** Floor for the note request: below this a summary cannot name the work it is replacing. */
export const HANDOFF_NOTE_MIN_TOKENS = 128;
/** The header, the searchability sentence and the two delimiters the server wraps the note in. */
const HANDOFF_NOTE_BLOCK_OVERHEAD_TOKENS = 128;
/** `UNSLOTH_CHECKPOINT_MAX_TOKENS` / `UNSLOTH_CHECKPOINT_MAX_FRACTION` in checkpoint.py. */
const CHECKPOINT_BLOCK_MAX_TOKENS = 1024;
const CHECKPOINT_BLOCK_FRACTION = 0.1;
/** The reply floor `prompt_budget` sets aside in context_window.py. */
const REPLY_FLOOR_DIVISOR = 4;

/** The note request's Max Tokens: what the block the server will render has room for.
 *
 *  Mirrors the server rather than trusting it to truncate: `fit_checkpoint_context` prices the carried
 *  block at a fraction of the prompt budget and drops what does not fit WHOLE, so a note a token too
 *  long loses the handoff entirely and the reset falls back to its deterministic carry. */
export function handoffNoteTokenBudget(options: {
  contextLength: number;
  maxTokens: number | null;
}): number {
  const contextLength = Math.floor(options.contextLength);
  if (contextLength <= 1) return HANDOFF_NOTE_MIN_TOKENS;
  const requested =
    options.maxTokens != null && options.maxTokens > 0
      ? options.maxTokens
      : contextLength;
  const promptTarget =
    contextLength -
    Math.min(requested, Math.max(1, Math.floor(contextLength / REPLY_FLOOR_DIVISOR)));
  const block = Math.min(
    CHECKPOINT_BLOCK_MAX_TOKENS,
    Math.floor(promptTarget * CHECKPOINT_BLOCK_FRACTION),
  );
  return Math.max(
    HANDOFF_NOTE_MIN_TOKENS,
    block - HANDOFF_NOTE_BLOCK_OVERHEAD_TOKENS,
  );
}

// Closed-pair extraction, the same shape tool_call_parser._TOOL_CLOSED_PATS uses for
// `< tool_call>...</ /tool_call>`; tolerate a missing close at end-of-turn like that parser, since a
// note cut off by Max Tokens still beats no note at all.
const HANDOFF_NOTE_RE = /<handoff_note>([\s\S]*?)(?:<\/handoff_note>|$)/i;

/** What the checkpoint reset keeps, stated for the model before it writes, so the note does not
 *  spend itself restating the two things that survive the reset. */
export const HANDOFF_KEPT =
  "your system prompt and the most recent user message";

/** The model wrote a usable note, or null for "compact the ordinary way". */
export function parseHandoffNote(text: string): string | null {
  const match = HANDOFF_NOTE_RE.exec(text);
  if (!match) return null;
  const note = match[1].trim();
  return note ? note : null;
}

/** The invisible message injected at the tail of the note request: stop, no more tools, write the
 *  note in the block, the user's own instructions appended, and what SURVIVES the reset up front so
 *  the model does not repeat it. Tailed rather than headed because it is the turn's last instruction
 *  and has to outrank everything the summary is being drawn from. */
export function buildHandoffSystemMessage(
  instructions: string,
  kept: string = HANDOFF_KEPT,
): string {
  return (
    "Stop what you are doing and do not call any more tools. " +
    `Write your handoff note inside a ${HANDOFF_NOTE_OPEN} ${HANDOFF_NOTE_CLOSE} block. ` +
    `${instructions.trim()} ` +
    `After compaction, ${kept} will remain in your context, so the note only needs ` +
    "everything else required to continue without re-reading the dropped turns."
  );
}

/** True when a handoff note should be generated now. Local GGUF and the mode are checked by the
 *  caller, which already owns both verdicts; this is the arithmetic on the window. */
export function shouldRunHandoff(opts: {
  contextUsage: { promptTokens: number } | null;
  contextLength: number | null;
  threshold: number;
  alreadyHandedOffAt: number | null;
}): boolean {
  const { contextUsage, contextLength, threshold } = opts;
  // No window or no count means no idea where the threshold is; the server's own compaction still
  // applies to the turn, so declining here loses nothing.
  if (!contextUsage || !contextLength || contextLength <= 0) return false;
  if (!(contextUsage.promptTokens / contextLength >= threshold)) return false;
  // Re-fire only after the context has grown past the point where we last handed off, so the note's
  // own tokens cannot re-trigger a loop on the very next send.
  return contextUsage.promptTokens > (opts.alreadyHandedOffAt ?? 0);
}

// Thread → prompt tokens at the last handoff. In-memory only, like continuation.ts's `spent` map: a
// handoff is a property of a live context, and a reload that empties the window resets the claim.
const handedOffAtByThreadId = new Map<string, number>();

export function lastHandoffAt(threadId: string | null): number | null {
  return threadId === null ? null : (handedOffAtByThreadId.get(threadId) ?? null);
}

export function recordHandoff(threadId: string | null, promptTokens: number): void {
  if (threadId === null) return;
  handedOffAtByThreadId.set(threadId, promptTokens);
}

/** The transient request that gets the note: the run's own outbound history with the handoff
 *  instruction appended, tools off so the model spends the request writing instead of calling, and a
 *  bounded Max Tokens.
 *
 *  Null on any failure — a model that ignored the block, an abort, an endpoint that refused. The
 *  caller sends the real turn either way, and without a note the server compacts exactly as it did
 *  before handoff existed. */
export async function generateHandoffNote(options: {
  model: string;
  messages: readonly OpenAIChatMessage[];
  instructions: string;
  /** From `handoffNoteTokenBudget`: the room the rendered block has, not the room the chat has. */
  maxTokens: number;
  /** The turn's own request fields — sampling, reasoning, replay — copied in verbatim below, so the
   *  note is written under the settings the chat is actually running with. */
  fields: Partial<OpenAIChatCompletionsRequest>;
  abortSignal: AbortSignal;
}): Promise<string | null> {
  const payload: OpenAIChatCompletionsRequest = {
    // The run's fields go in FIRST, so everything below is the only thing this request does
    // differently. The server's own compaction is handed the live request and inherits its sampling
    // from it; a side call that used the server defaults would quietly reconfigure a chat the user
    // configured, and the note is the one artefact that survives into every later turn. What is
    // deliberately NOT copied is the run's plumbing rather than its settings: `thread_id` (this call
    // must not archive against the thread nor move its sticky compaction boundary), `cancel_id` and
    // `session_id` (this call ends when the caller's signal does), this turn's media, and the tool
    // wiring, which is switched off below.
    ...options.fields,
    model: options.model,
    messages: [
      ...options.messages,
      {
        role: "system",
        content: buildHandoffSystemMessage(options.instructions),
      },
    ],
    stream: true,
    // The one number that cannot be the run's: the room the note's block has in the reset, not the
    // room the chat has.
    max_tokens: options.maxTokens,
    // The server's default offers tools to a GGUF that was started with --enable-tools, and a model
    // that opens a call here writes no note at all.
    enable_tools: false,
  };

  let text = "";
  try {
    for await (const chunk of streamChatCompletions(payload, options.abortSignal)) {
      const delta = chunk.choices?.[0]?.delta?.content;
      if (delta == null) continue;
      text += extractDeltaText(delta).text;
    }
  } catch {
    // Aborted or refused: no note, and the turn goes out without one. A Stop lands here too, and the
    // real request that follows is cancelled by the same signal.
    return null;
  }
  return parseHandoffNote(text);
}
