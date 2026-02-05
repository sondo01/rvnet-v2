---
trigger: always_on
---

Environment Configuration

When executing Python commands or running tests (pytest/unittest), ALWAYS use the interpreter located at: ./.venv/bin/python (or .\.venv\Scripts\python.exe on Windows). Do not run python or pytest directly from the global path.
