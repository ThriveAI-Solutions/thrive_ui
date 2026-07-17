# Repository Working Agreement

These instructions apply to the entire repository.

## GitHub delivery workflow

Unless the user explicitly asks for a different workflow, substantive changes
must be delivered through GitHub:

1. Identify the existing issue before coding, or create a narrowly scoped issue
   when the work is not tracked.
2. Check the GitHub Project and issue labels before assuming ownership. An issue
   assigned to Thrive may still be a HEALTHeLINK responsibility. Treat
   `thrive`, `healthelink`, `joint`, and `external-dependency` labels as the
   ownership source of truth, and call out contradictions.
3. Work on a dedicated branch. Codex-created branches use the `codex/` prefix.
4. Keep unrelated working-tree changes out of the branch, commit, and pull
   request. Stage explicit paths in a mixed worktree; do not use `git add -A`.
5. Run focused tests for the changed behavior and a proportionate broader
   suite. Record the exact commands and results in the pull request.
6. Open a pull request that explains the problem, root cause, implementation,
   impact, and validation. Use `Closes #N` only when the merged change fully
   satisfies that issue; otherwise use `Refs #N` and state what remains.
7. Merge only after required checks pass and review requirements are satisfied.
8. After merge, reconcile GitHub: comment with evidence, close completed issues,
   and update Project status (`Todo`, `In Progress`, `Blocked`, `In Review`, or
   `Done`). Never close an external dependency merely because Thrive's portion
   is complete.

## Safety and scope

- Preserve user changes already present in the worktree.
- Do not commit credentials, PHI, patient rosters, evaluation results, local
  databases, screenshots, or generated reports containing sensitive data.
- Consent, patient identity resolution, Public Health role mapping, and
  compliance acceptance criteria require the documented business decision;
  do not invent one to unblock implementation.
- Prefer evidence over status claims: cite tests, merged code, deployed checks,
  or an explicit external sign-off.
