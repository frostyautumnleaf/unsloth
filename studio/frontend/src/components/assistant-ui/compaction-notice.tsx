// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ArchiveIcon } from "lucide-react";
import type { FC } from "react";

import {
  type ContextTruncation,
  promptWasShortened,
} from "@/features/chat/utils/context-truncation";

/**
 * Tells the user their conversation was compacted, on the turn where it STARTED.
 *
 * Deliberately NOT a message: it renders inside the assistant message's container from
 * `metadata.custom.contextTruncation`, so it is never sent to the model, editable, or
 * exported as content, yet unlike a toast it survives a reload and stays attached to the
 * turn it describes.
 *
 * Rendered once per COMPACTION, gated by the caller, not once per compacted turn: a
 * thread past its window refits on every request, so per-turn would mean a notice on
 * every reply forever. The caller shows this when the boundary moved or a checkpoint started.
 */
export const CompactionNotice: FC<{ truncation: ContextTruncation }> = ({
  truncation,
}) => {
  if (!promptWasShortened(truncation)) return null;

  const dropped = truncation.dropped_messages;
  const archived = truncation.archived_messages ?? 0;
  const recalled = truncation.recalled_chunks ?? 0;
  // The Handoff mode's note. Trimmed because a note of only whitespace was never worth a section,
  // and the server sends the field only when a note was carried.
  const handoffNote = truncation.handoff_note?.trim() ?? "";

  const detail = archived
    ? "They are saved and searchable, and the parts relevant to each question are brought back automatically."
    : "The full conversation is still visible and saved here.";

  return (
    <div
      className="aui-compaction-notice mb-3 flex items-start gap-2 rounded-lg border border-border/60 bg-muted/40 px-3 py-2 text-ui-13 text-muted-foreground"
      data-testid="compaction-notice"
      data-dropped={dropped}
      data-archived={archived}
      data-recalled={recalled}
    >
      <ArchiveIcon className="mt-0.5 size-3.5 shrink-0" aria-hidden />
      <div className="min-w-0">
        <span className="font-medium text-foreground/80">
          This conversation got long, so it was compacted.
        </span>{" "}
        <span>
          Older messages were dropped from the model&apos;s context to make room. {detail}
        </span>
        <span>
          {" "}
          ({dropped} {dropped === 1 ? "message" : "messages"} dropped here
          {recalled > 0
            ? `, ${recalled} earlier ${recalled === 1 ? "passage" : "passages"} recalled`
            : ""}
          .)
        </span>
        {handoffNote ? (
          // Collapsed by default: the note is the dropped turns' replacement in the MODEL's context,
          // and most readers of a compacted chat want to know it happened, not to re-read the whole
          // conversation summarised at them. A native <details> keeps it shut with no state to
          // restore and no JS, so it is shut on a reload too.
          <details
            className="mt-2"
            data-testid="compaction-notice-handoff"
            data-handoff-chars={handoffNote.length}
          >
            <summary className="w-fit cursor-pointer select-none text-ui-12 font-medium text-foreground/70 underline-offset-2 hover:underline focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-foreground/40">
              What the model handed over
            </summary>
            <div className="mt-1.5 max-h-64 overflow-y-auto whitespace-pre-wrap break-words rounded-md border border-border/50 bg-background/60 px-2.5 py-2 text-ui-12">
              {handoffNote}
            </div>
          </details>
        ) : null}
      </div>
    </div>
  );
};
