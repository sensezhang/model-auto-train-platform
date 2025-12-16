# Repository Guidelines

## Project Structure & Modules
- `backend/`: FastAPI app and ML services.
  - `backend/app/main.py`: FastAPI entry; mounts routers and /files.
  - `backend/app/routers/`: API routes (projects, images, annotations, autolabel, training).
  - `backend/app/services/`: YOLO training, auto-label helpers.
  - `backend/requirements.txt`: Python deps.
- `frontend/`: React + Vite app.
  - `frontend/src/`: UI pages and bootstrap.
- `app.db`: SQLite database for dev.
- `Prd.md`, `README.md`: product and setup notes.

## Build, Test, Run
- Backend (venv recommended):
  - `cd backend && python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt` — install deps.
  - `uvicorn app.main:app --reload --port 8000` — run API.
- Frontend:
  - `cd frontend && npm install` — install deps.
  - `npm run dev` — start Vite dev server on 5173.
  - Proxy API via `vite.config.ts` to `http://localhost:8000`.

## Coding Style & Naming
- Python: Black-ish style, 4-space indent, type hints where practical. Modules: `snake_case.py`; classes `PascalCase`; functions/vars `snake_case`.
- TypeScript/React: 2-space indent, functional components, `PascalCase` for components, `camelCase` for vars.
- Keep routers small and focused; shared logic lives in `services/` or `utils/`.

## Testing Guidelines
- Backend: prefer `pytest`. Place tests under `backend/tests/` mirroring app layout, e.g., `tests/routers/test_projects.py`.
- Frontend: add Vitest/RTL tests under `frontend/src/__tests__/`.
- Run: `pytest` (backend), `npm test` (frontend) once configured. Target meaningful coverage for routers and services.

## Commit & PR Guidelines
- Commits: imperative, scoped messages, e.g., `feat(training): add ONNX export flag`.
- PRs: include summary, linked issues, before/after notes or screenshots for UI, and steps to verify.
- Keep diffs minimal; update README if commands or flows change.

## Security & Config Tips
- Do not commit real datasets, secrets, or large weights. Use `/files` only for local dev.
- For YOLO training, install `ultralytics` and a CUDA-matched `torch`. Cache weights locally to avoid downloads in CI.
