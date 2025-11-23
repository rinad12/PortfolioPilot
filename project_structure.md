# PortfolioPilot - Project Structure

This is a monorepo containing both the backend (FastAPI) and frontend (Next.js) applications.

## Root Directory

```
PortfolioPilot/
├── backend/              # Python FastAPI backend application
├── frontend/             # Next.js frontend application
├── .cursor/              # Cursor IDE configuration and rules
├── db/                   # Database initialization scripts
├── .env                  # Environment variables (gitignored)
├── .env.example          # Environment template for reference
├── .gitignore            # Git ignore patterns
├── docker-compose.yml    # Docker compose configuration for local database
├── README.md             # Project documentation
└── project_structure.md  # This file
```

## Backend Structure (`backend/`)

```
backend/
├── src/
│   ├── portfoliopilot/  # Main Python package
│   │   └── __init__.py
│   ├── core/            # Infrastructure and database configuration
│   │   ├── __init__.py
│   │   ├── config.py    # Pydantic settings for environment variables
│   │   ├── database.py  # SQLModel engine and session management
│   │   └── models/      # Database model definitions
│   │       └── __init__.py
│   ├── agents/          # Agent implementations
│   ├── tools/           # Tool implementations
│   └── data/            # Data processing modules
├── alembic/             # Database migration scripts
│   ├── versions/        # Migration files
│   ├── env.py           # Alembic environment configuration
│   ├── script.py.mako   # Migration template
│   └── README           # Alembic documentation
├── tests/               # Pytest test files
├── alembic.ini          # Alembic configuration
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
- Backend core infrastructure: `backend/src/core/`
- Database migrations: `backend/alembic/`
- Migration configuration: `backend/alembic.ini`
- Frontend source: `frontend/app/` (Next.js App Router)
- Frontend config: `frontend/package.json`
- Cursor rules: `.cursor/rules/`
- Environment template: `.env.example`

## Database Migration Workflow

The project uses Alembic for database schema management:

1. **Initial Setup**: The `db/init.sql` script creates the pgvector extension on container startup
2. **Migrations**: All schema and data changes are managed via Alembic migrations in `backend/alembic/versions/`
3. **Running Migrations**: 
   - Start database: `docker-compose up -d db`
   - Apply migrations: `cd backend && uv run alembic upgrade head`
   - Check current state: `cd backend && uv run alembic current`
4. **Creating New Migrations**: After defining new models, run `cd backend && uv run alembic revision --autogenerate -m "description"`

See `backend/MIGRATIONS.md` for detailed migration documentation.

