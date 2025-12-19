Perform a comprehensive code review of recent changes:

1. Check code follows our Python, Gym, and notebook conventions from CLAUDE.md and sub-CLAUDEs.
2. Verify proper error handling, seeding, and device placement; no side effects on import.
3. Ensure notebooks stay lightweight (outputs stripped unless requested) and heavy cells are marked.
4. Review test coverage or smoke runs for changed assignments; confirm pytest/py_compile when relevant.
5. Check security: no secrets, no large artifacts committed, answers/ PDFs untouched unless intended.
6. Validate performance implications (avoid runaway rollouts, GPU/CPU thrash, memory growth).
7. Confirm documentation/README updates match code defaults and CLI flags.

Provide specific, actionable feedback with file references and suggested fixes.
