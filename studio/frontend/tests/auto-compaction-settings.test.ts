// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { ggufCompactionRequestFields } from "../src/features/chat/utils/auto-compaction.ts";

import { readSrc } from "./helpers/kit.ts";

test("auto sends truncate_oldest and no policy of its own", () => {
  // No context_policy: the server applies UNSLOTH_CONTEXT_POLICY. Studio used to offer that
  // choice as a setting and no longer does, so this is the only shape an enabled request takes.
  assert.deepEqual(
    ggufCompactionRequestFields({ isGguf: true, compactionMode: "auto" }),
    { context_overflow: "truncate_oldest" },
  );
});

test("off sends an explicit error overflow policy", () => {
  assert.deepEqual(
    ggufCompactionRequestFields({ isGguf: true, compactionMode: "off" }),
    { context_overflow: "error" },
  );
});

test("handoff moves the reset to the point the user set", () => {
  // Handoff REPLACES the default compaction rather than decorating it, so it has to own when the
  // window is rewritten, not only what survives. Without the threshold the server resets at its own
  // reservation for the reply (75% of the window at the default 90% setting), so the conversation is
  // compacted the default way long before the handoff point and never reaches it.
  assert.deepEqual(
    ggufCompactionRequestFields({
      isGguf: true,
      compactionMode: "handoff",
      handoffThreshold: 0.9,
    }),
    { context_overflow: "truncate_oldest", handoffThreshold: 0.9 },
  );
});

test("handoff with no threshold named keeps the server's own point", () => {
  // An older client, or a caller that has no window to measure: the field is omitted rather than
  // guessed, because a guessed fraction silently moves a reset the user did not ask for.
  assert.deepEqual(
    ggufCompactionRequestFields({ isGguf: true, compactionMode: "handoff" }),
    { context_overflow: "truncate_oldest" },
  );
});

test("archive does not borrow the handoff threshold", () => {
  // The threshold belongs to handoff. Sent with archive it would move the default compaction off its
  // own formula for a mode whose note is never asked for, which is a chat that stops compacting early
  // and refuses instead.
  assert.deepEqual(
    ggufCompactionRequestFields({
      isGguf: true,
      compactionMode: "auto",
      handoffThreshold: 0.9,
    }),
    { context_overflow: "truncate_oldest" },
  );
});

test("external models never opt into GGUF compaction", () => {
  assert.deepEqual(
    ggufCompactionRequestFields({ isGguf: false, compactionMode: "auto" }),
    {},
  );
});

test("nothing still offers the removed compaction style", () => {
  // The row is gone, so its state, its request fields and its strings should be too: a leftover
  // setter or key is a setting the user cannot reach but the code still carries.
  for (const file of [
    "features/settings/tabs/chat-tab.tsx",
    "features/chat/stores/chat-runtime-store.ts",
    "features/chat/utils/chat-settings-storage.ts",
    "features/chat/utils/queued-chat-run-settings.ts",
    "features/chat/api/chat-settings-api.ts",
    "features/chat/api/chat-adapter.ts",
    "features/chat/utils/auto-compaction.ts",
    "features/settings/settings-search.ts",
    "i18n/locales/en.ts",
  ]) {
    assert.doesNotMatch(
      readSrc(file),
      /contextPolicy|compactionHeadroomRatio|compactionStyle|compactionDescription/,
      `${file} still carries the removed compaction style`,
    );
  }
});

test("the chat adapter sends compaction fields through the shared helper", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(adapter, /ggufCompactionRequestFields\(/);
  // Through isServedByLlamaCpp, not a catalog row: /api/models/list can replace the row a
  // load minted, and the panel that shows these settings asks the same owner.
  assert.match(adapter, /isGguf: isGgufForCompaction/);
  assert.match(adapter, /loadedIsGguf: runtime\.loadedIsGguf/);
  assert.match(adapter, /isServedByLlamaCpp\(/);
// One request object for both streams, so a media turn cannot lose these fields; the value is
  // derived from THIS turn's messages only (durable-gate.ts), not a variable hoisted earlier.
  assert.match(adapter, /image_base64: findLatestUserImageBase64\(currentTurnMessages\)/);
});

test("a queued run keeps its own model's llama.cpp verdict after the picker moves on", async () => {
  const { registerBundlerResolver, installLocalStorageFake } = await import(
    "./helpers/kit.ts"
  );
  registerBundlerResolver();
  installLocalStorageFake();
  const { snapshotQueuedChatRunSettings } = await import(
    "../src/features/chat/utils/queued-chat-run-settings.ts"
  );
  const { isServedByLlamaCpp, loadedContextFields } = await import(
    "../src/features/model-picker/model-config/per-model-config.ts"
  );

  // An Ollama GGUF: the backend keeps the opaque inventory ref public, so the checkpoint
  // carries no .gguf suffix, and the load reports no quant. loadedIsGguf is the only
  // evidence llama.cpp serves it.
  const resident = {
    params: { checkpoint: "ollama-manifest:%2Fhome%2Fu%2F.ollama%2Fmanifests%2Fq" },
    activeGgufVariant: null,
    activeNativePathToken: null,
    loadedIsGguf: true,
    loadedContextLength: 8192,
    compactionMode: "auto",
  };
  const queued = snapshotQueuedChatRunSettings(
    resident as unknown as Parameters<typeof snapshotQueuedChatRunSettings>[0],
  );

  // Selecting an external provider clears the local residency fields without unloading
  // the model that the queued turn is still going to be served by.
  const live = {
    ...resident,
    params: { checkpoint: "external::openai::gpt-5" },
    activeGgufVariant: null,
    activeNativePathToken: null,
    ...loadedContextFields(null),
  };
  const runtime = { ...live, ...queued };

  const isGguf = isServedByLlamaCpp({
    loadedIsGguf: runtime.loadedIsGguf,
    activeGgufVariant: runtime.activeGgufVariant,
    activeNativePathToken: runtime.activeNativePathToken,
    checkpoint: runtime.params.checkpoint,
  });
  assert.equal(isGguf, true);
  assert.deepEqual(
    ggufCompactionRequestFields({
      isGguf,
      compactionMode: runtime.compactionMode,
    }),
    { context_overflow: "truncate_oldest" },
  );
});

test("a saved autoCompactEnabled survives as compactionMode", async () => {
  // The switch became a tri-state, and a stored "off" was an explicit refusal of the archive:
  // falling back to the new default here would switch compaction back on for everyone who turned it
  // off, on an upgrade nobody asked for. The store stubs keep the storage layer off the auth barrel.
  const { registerStoreStubResolver, installLocalStorageFake } = await import(
    "./helpers/kit.ts"
  );
  registerStoreStubResolver();
  installLocalStorageFake();
  const { normalizeSavedChatSettings } = await import(
    "../src/features/chat/utils/chat-settings-storage.ts"
  );

  assert.equal(
    normalizeSavedChatSettings({ autoCompactEnabled: false }).compactionMode,
    "off",
  );
  assert.equal(
    normalizeSavedChatSettings({ autoCompactEnabled: true }).compactionMode,
    "auto",
  );
  // An explicit mode outranks the legacy key, and a legacy key of the wrong type goes the way of
  // every other unsanitizable value rather than guessing a mode.
  assert.equal(
    normalizeSavedChatSettings({
      autoCompactEnabled: false,
      compactionMode: "handoff",
    }).compactionMode,
    "handoff",
  );
  assert.equal(
    normalizeSavedChatSettings({ autoCompactEnabled: "yes" }).compactionMode,
    undefined,
  );
});
