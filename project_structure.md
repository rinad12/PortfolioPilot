# PortfolioPilot - Project Structure

This is a monorepo containing both the backend (FastAPI) and frontend (Next.js) applications.

## Root Directory

```
PortfolioPilot/
├── backend/              # Python FastAPI backend application
├── frontend/             # Next.js frontend application
├── .cursor/              # Cursor IDE configuration and rules
├── .gitignore           # Git ignore patterns
├── README.md            # Project documentation
└── project_structure.md # This file
```

## Backend Structure (`backend/`)

```
backend/
├── src/
│   └── portfoliopilot/  # Main Python package
│       └── __init__.py
├── tests/               # Pytest test files
├── main.py              # Application entry point
├── pyproject.toml       # Python project configuration (uv)
└── uv.lock              # Dependency lock file (if using uv)
```

### Backend Details

- **Package Manager**: `uv` (configured in `pyproject.toml`)
- **Source Layout**: Uses `src/` layout pattern (`src = "src"` in `[tool.uv]`)
- **Package Name**: `portfoliopilot`
- **Python Version**: >=3.12
- **Framework**: FastAPI
- **Testing**: Pytest with async support

### Key Backend Files

- `backend/pyproject.toml`: Project metadata, dependencies, and tool configurations
- `backend/main.py`: Application entry point
- `backend/src/portfoliopilot/`: Main package source code

## Frontend Structure (`frontend/`)

```
frontend/
├── app/                 # Next.js App Router pages
├── components/          # React components
├── lib/                 # Utility functions and API clients
├── public/              # Static assets
├── styles/              # Global styles
├── package.json         # Node.js dependencies
├── tsconfig.json        # TypeScript configuration
├── tailwind.config.ts   # Tailwind CSS configuration
└── next.config.js       # Next.js configuration
```

### Frontend Details

- **Framework**: Next.js (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Package Manager**: npm/pnpm/yarn (to be configured)

## Development Workflow

### Backend Development

1. Navigate to `backend/` directory
2. Install dependencies: `uv sync` (or `uv pip install -e .`)
3. Run application: `python main.py` or `uvicorn portfoliopilot.main:app`
4. Run tests: `pytest` (from `backend/` directory)

### Frontend Development

1. Navigate to `frontend/` directory
2. Install dependencies: `npm install` (or `pnpm install`)
3. Run dev server: `npm run dev`
4. Build for production: `npm run build`

## Architecture Notes

- **Backend**: FastAPI REST API with async/await patterns
- **Frontend**: Next.js with Server Actions or typed API calls
- **Type Safety**: TypeScript interfaces in frontend should match Pydantic schemas in backend
- **Monorepo**: Both applications share the same repository but maintain separate dependency management

## Important Paths

- Backend source: `backend/src/portfoliopilot/`
- Backend config: `backend/pyproject.toml`
- Frontend source: `frontend/app/` (Next.js App Router)
- Frontend config: `frontend/package.json`
- Cursor rules: `.cursor/rules/`

