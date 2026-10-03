// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Local GGUF compaction. Off, or the server's own reset, or a handoff the model writes first; which
 *  reset algorithm runs is still the server's call.
 *
 *  Studio used to offer the policy as a setting, but the choice needed the reader to know what a
 *  checkpoint epoch and a rolling window were before it meant anything, and both sides of it were
 *  already the server's to configure. It now always follows the server, which is what the setting
 *  shipped as anyway (UNSLOTH_CONTEXT_POLICY, default "checkpoint"). Handoff is the exception to that:
 *  it asks the model for a note, swaps that note in for the deterministic carried block, AND moves the
 *  reset to the fraction the user set — because a compaction that fires before the handoff point is not
 *  handoff, it is the old one with a feature bolted on the side. */

export const DEFAULT_COMPACTION_MODE = "auto";
export const DEFAULT_HANDOFF_THRESHOLD = 0.9;
export const HANDOFF_INSTRUCTIONS_MAX_CHARS = 4096;
export const HANDOFF_THRESHOLD_MIN = 0.1;
export const HANDOFF_THRESHOLD_MAX = 0.95;

/** Pre-filled into the settings text box; appended verbatim to the hidden note-request message. */
export const DEFAULT_HANDOFF_INSTRUCTIONS =
  "Write a concise but detailed handoff summary of the work so far: the task, what is done, key decisions and constraints, the current state (files, code, open questions), and the exact next steps. Be specific about names, paths, and values so work can continue without re-reading the dropped turns.";

export type CompactionMode = "off" | "auto" | "handoff";

export function ggufCompactionRequestFields(options: {
  isGguf: boolean;
  compactionMode: CompactionMode;
  /** Only meaningful with `handoff`, and the reason the server needs it: see below. */
  handoffThreshold?: number;
}): {
  context_overflow?: "error" | "truncate_oldest";
  handoffThreshold?: number;
} {
  if (!options.isGguf) return {};
  if (options.compactionMode === "off") {
    // An omitted field falls back to UNSLOTH_CONTEXT_OVERFLOW, which may still compact. "error" is an
    // explicit refusal of that fallback.
    return { context_overflow: "error" };
  }
  if (options.compactionMode === "handoff") {
    // Handoff replaces the default compaction rather than decorating it, so it has to own WHEN the
    // window is rewritten as well as what survives. Left to itself the server resets at its own
    // reservation for the reply — 75% of the window at the default 90% setting — so the conversation
    // was compacted long before the point the user chose, and every reset pulled the usage back under
    // the threshold that gates the note. No context_policy: the server applies UNSLOTH_CONTEXT_POLICY.
    return {
      context_overflow: "truncate_oldest",
      ...(options.handoffThreshold != null
        ? { handoffThreshold: options.handoffThreshold }
        : {}),
    };
  }
  return { context_overflow: "truncate_oldest" };
}
