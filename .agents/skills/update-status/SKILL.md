---
name: update-status
description: Summarize recent substantive work in the current thread, reconcile it with repository evidence, and update the repository root STATUS.md. Use after meaningful implementation, debugging, or planning work. Do not use for trivial edits, purely conversational summaries, or when there is no substantive work to record.
---

# Update Status

Use this skill when the user reports completion of a logical block of work and wants `STATUS.md` refreshed as a handoff record.

The work may be committed, staged, or only present in the working tree.

Keep the update factual. The goal is to leave `STATUS.md` as a short, accurate handoff artifact grounded in both the thread and the repository state.

## Inputs

Use two evidence sources:

1. The current thread: what was implemented, investigated, decided, deferred, or blocked.
2. Repository evidence:
   - committed, staged, and unstaged changes
   - current branch, HEAD commit, and relevant commit range since the last status entry if available
   - associated logs, test commands, and test results when available

Do not rely on thread memory alone if repository evidence is available.

## Target file

Update the repository root `STATUS.md`.

If `STATUS.md` does not exist, create it at the repository root. 
Do not update unrelated `status.md` files elsewhere in the tree.

## Workflow

1. Identify the repository root and create `STATUS.md` if necessary.
2. Review the current thread for substantive work only.
3. Inspect the latest relevant entry in `STATUS.md` if the file already exists.
4. If an earlier status entry records a commit, use it as the lower bound for the repository review when that is still relevant to the current branch. Otherwise review the current branch state directly without inventing a baseline.
5. Ingest all relevant committed, staged, and unstaged changes that contribute to the current work summary.
6. Reconcile discrepancies explicitly:
   - prefer repository evidence for claims about code state
   - prefer the thread for intent, rationale, and deferred work
7. Update `STATUS.md` with a concise snapshot of:
   - current branch, commit hash when available, and git author
   - the goal of the work: issue addressed, feature implemented, bug investigated, or plan produced
   - what changed
   - what was verified
   - any notable compromises, odd code, technical debt, or follow-up work introduced to solve the issue
8. Re-read the final `STATUS.md` and check that every claim is still supported.

## Content rules
- Head each status item by: Current commit hash, branch, and git author.
- The snapshot should be minimalistic but informative.
- Prefer current facts over historical narrative.
- Include concrete file paths, commands, or tests only when they materially help the handoff.
- Call out missing verification clearly.
- Record blockers and partial work explicitly rather than smoothing them over.
- Do not include speculation or filler.
- Do not claim tests passed unless you verified that in this thread or from repository evidence.
- If there is no commit yet for the current work, say so plainly instead of inventing one.

## Suggested structure
Put the new items at the top.
Use a compact structure such as:

```markdown
# Status summary (goals)
`<date>`:  `<hash 6 digits>` @ `<branch>` by `author`

## Goal(s) 
Can be omitted if clear from the title.

## Changes summary

## Open Items
Can be omitted.
```

## When not to use

Do not use this skill:

- after trivial edits
- when there is no meaningful work in the current thread
- when the user wants a conversational summary only and did not ask to update repository state

## Completion check

Before finishing:

- confirm `STATUS.md` reflects the current repository state
- confirm unresolved items are listed plainly
- confirm the update is short enough for a human handoff
