Analyze and fix GitHub issue: $ARGUMENTS

Steps:
1. Fetch details: `gh issue view $ARGUMENTS` (or read linked ticket) to capture requirements.
2. Identify scope: locate relevant assignment folder (`CAs/`, `homeworks/`, or `Other_Assisments/`) and read the nearest CLAUDE.md.
3. Search codebase with `rg` for symbols and configs tied to the issue.
4. Plan a minimal fix; avoid cross-assignment churn. Keep seeds and APIs consistent.
5. Implement changes following folder patterns (agents/environments/utils/notebooks as appropriate).
6. Add/update tests or lightweight smoke checks (pytest or py_compile) relevant to touched files.
7. Strip notebook outputs if modified.
8. Run targeted checks: `python -m py_compile <files>` and `python -m pytest <tests>` where available.
9. Craft a descriptive commit message summarizing scope and checks.
10. If using GitHub flow, open PR with context and risks noted.

Always respect safety rules (no secrets, no destructive commands, no force push without approval).
