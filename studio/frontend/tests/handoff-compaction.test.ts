// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

// The module also owns the transient note request, so its one import of the API layer is what puts a
// request on the wire. Cut it here rather than stub the auth barrel: everything below drives the real
// source with a transport that only records what it was handed.
const sent: Record<string, unknown>[] = [];
const NOTE_REPLY =
  "<handoff_note>Task: retune the sampler. Done: the gate. Next: ship it.</handoff_note>";

const {
  HANDOFF_KEPT,
  HANDOFF_NOTE_OPEN,
  buildHandoffSystemMessage,
  generateHandoffNote,
  handoffNoteTokenBudget,
  lastHandoffAt,
  parseHandoffNote,
  recordHandoff,
  shouldRunHandoff,
} = loadWithStubs<{
  HANDOFF_KEPT: string;
  HANDOFF_NOTE_OPEN: string;
  buildHandoffSystemMessage: (instructions: string, kept?: string) => string;
  generateHandoffNote: (options: {
    model: string;
    messages: { role: string; content: string }[];
    instructions: string;
    maxTokens: number;
    fields: Record<string, unknown>;
    abortSignal: AbortSignal;
  }) => Promise<string | null>;
  handoffNoteTokenBudget: (options: {
    contextLength: number;
    maxTokens: number | null;
  }) => number;
  lastHandoffAt: (threadId: string | null) => number | null;
  parseHandoffNote: (text: string) => string | null;
  recordHandoff: (threadId: string | null, promptTokens: number) => void;
  shouldRunHandoff: (options: {
    contextUsage: { promptTokens: number } | null;
    contextLength: number | null;
    threshold: number;
    alreadyHandedOffAt: number | null;
  }) => boolean;
}>(new URL("../src/features/chat/utils/handoff-compaction.ts", import.meta.url), {
  "../api/chat-api": {
    streamChatCompletions: (payload: Record<string, unknown>) => {
      sent.push(payload);
      return (async function* () {
        yield { choices: [{ delta: { content: NOTE_REPLY } }] };
      })();
    },
  },
  "./parse-assistant-content": {
    // The real one unwraps a plain string, a provider's part list, or an object carrying reasoning
    // text. A local GGUF note reply is always the plain string.
    extractDeltaText: (delta: unknown) => ({
      text: typeof delta === "string" ? delta : "",
    }),
  },
});

test("the note is extracted the way a tool call is", () => {
  const text =
    "Preamble the run discards.\n<handoff_note>\n  Task: paint the widget. Done: the mask. \n</handoff_note>\n";
  assert.equal(
    parseHandoffNote(text),
    "Task: paint the widget. Done: the mask.",
  );
});

test("a note cut off by Max Tokens is still a note", () => {
  // Same tolerance tool_call_parser gives a `< tool_call>` whose pair never arrived: half a note
  // beats the deterministic block this one was written to replace.
  assert.equal(
    parseHandoffNote("<handoff_note>wrote the parser, still"),
    "wrote the parser, still",
  );
});

test("no block, or an empty one, is no note", () => {
  assert.equal(parseHandoffNote("the model answered normally"), null);
  assert.equal(parseHandoffNote("<handoff_note>   \n  </handoff_note>"), null);
});

test("the injected message says stop, no tools, the block, and what survives", () => {
  const message = buildHandoffSystemMessage(
    "Name every file you touched.",
    HANDOFF_KEPT,
  );
  assert.match(message, /do not call any more tools/i);
  assert.ok(message.includes(`${HANDOFF_NOTE_OPEN}`));
  assert.ok(message.includes("Name every file you touched."));
  // The requirement is that the model is told BEFORE it writes what the reset keeps, so the note
  // does not spend itself restating the system prompt and the newest turn.
  assert.ok(message.includes(HANDOFF_KEPT));
});

test("the gate fires at the threshold and not before", () => {
  const at = (tokens: number) => ({
    contextUsage: { promptTokens: tokens },
    contextLength: 100_000,
    threshold: 0.9,
    alreadyHandedOffAt: null,
  });
  assert.equal(shouldRunHandoff(at(89_999)), false);
  assert.equal(shouldRunHandoff(at(90_000)), true);
});

test("no window or no count means no handoff", () => {
  // The server's own compaction still applies to the turn, so declining here loses nothing.
  assert.equal(
    shouldRunHandoff({
      contextUsage: null,
      contextLength: 100_000,
      threshold: 0.9,
      alreadyHandedOffAt: null,
    }),
    false,
  );
  assert.equal(
    shouldRunHandoff({
      contextUsage: { promptTokens: 95_000 },
      contextLength: null,
      threshold: 0.9,
      alreadyHandedOffAt: null,
    }),
    false,
  );
});

test("the note's own tokens cannot re-trigger it on the next send", () => {
  // Recorded at 90,000, so a turn that lands at 90,000 again (or below) is the same growth step;
  // only a prompt that has actually grown past it hands off a second time.
  const guard = (tokens: number) =>
    shouldRunHandoff({
      contextUsage: { promptTokens: tokens },
      contextLength: 100_000,
      threshold: 0.9,
      alreadyHandedOffAt: 90_000,
    });
  assert.equal(guard(90_000), false);
  assert.equal(guard(94_000), true);
});

test("the guard is per thread, and a threadless run has none", () => {
  recordHandoff("thread-a", 1234);
  assert.equal(lastHandoffAt("thread-a"), 1234);
  assert.equal(lastHandoffAt("thread-b"), null);
  recordHandoff(null, 5678);
  assert.equal(lastHandoffAt(null), null);
});

test("the note request samples with the turn's settings, not the server's defaults", async () => {
  // The point of the `fields` argument: the server's own compaction is handed the live request and
  // inherits its sampling from it, so the side call has to carry the same numbers. A note written at
  // defaults would be a settings change nobody made, in text that outlives the turns it describes.
  const note = await generateHandoffNote({
    model: "unsloth/Model-GGUF",
    messages: [{ role: "user", content: "keep going" }],
    instructions: "Name every file you touched.",
    maxTokens: 512,
    fields: {
      temperature: 0.21,
      top_p: 0.9,
      top_k: 40,
      min_p: 0.05,
      repetition_penalty: 1.07,
      presence_penalty: -0.1,
      seed: 7,
      max_tokens: 8192,
      enable_thinking: false,
      preserve_thinking: true,
      studio_tool_history: true,
      context_overflow: "truncate_oldest",
    },
    abortSignal: new AbortController().signal,
  });
  const body = sent[0];
  assert.equal(note, "Task: retune the sampler. Done: the gate. Next: ship it.");
  for (const [key, value] of [
    ["temperature", 0.21],
    ["top_p", 0.9],
    ["top_k", 40],
    ["min_p", 0.05],
    ["repetition_penalty", 1.07],
    ["presence_penalty", -0.1],
    ["seed", 7],
    ["enable_thinking", false],
    ["preserve_thinking", true],
    ["studio_tool_history", true],
    ["context_overflow", "truncate_oldest"],
  ] as const) {
    assert.deepEqual([key, body[key]], [key, value], `${key} changed en route`);
  }
  // The two the note request owns: its own room, and no tools to spend it on.
  assert.equal(body.max_tokens, 512);
  assert.equal(body.enable_tools, false);
  // And the run's plumbing, which belongs to the turn rather than to this call: the thread it must
  // not archive against, its cancel identity, its attachments.
  assert.equal("thread_id" in body, false);
  assert.equal("cancel_id" in body, false);
  assert.equal("image_base64" in body, false);
  // The instruction is the LAST message: it has to outrank everything the summary is drawn from.
  const messages = body.messages as { role: string; content: string }[];
  assert.equal(messages[messages.length - 1].role, "system");
  assert.ok(
    messages[messages.length - 1].content.includes("Name every file you touched."),
  );
});

test("the note's room is priced against the block the reset renders", () => {
  // `fit_checkpoint_context` drops a carried block WHOLE when it does not fit, so asking for more
  // than the block has room for loses the handoff instead of trimming it.
  const budget = handoffNoteTokenBudget({ contextLength: 100_000, maxTokens: null });
  assert.ok(budget >= 128);
  assert.ok(budget < 1024);
  assert.equal(
    handoffNoteTokenBudget({ contextLength: 1, maxTokens: null }),
    128,
  );
});
