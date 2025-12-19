Create a database migration (use only if a DB-backed task is introduced): $ARGUMENTS

1. Confirm a migration tool exists; if not, pause and ask for approval/stack.
2. Create migration file with the project tool (placeholder): `python manage.py makemigrations "$ARGUMENTS"` or similar — adjust to actual stack.
3. Write safe up/down steps; avoid data loss and provide defaults for new columns.
4. Run migration locally (dry-run if available): `python manage.py migrate`.
5. Verify schema and backwards compatibility; snapshot before/after if tools exist.
6. Run relevant tests/smoke checks touching the affected codepaths.
7. Document breaking changes and env vars; update CLAUDE/README if new steps are required.
8. Commit migration files and notes only after validation.

CRITICAL: Never run migrations against production or shared datasets without explicit approval.
