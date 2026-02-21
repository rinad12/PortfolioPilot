# PortfolioPilot

An AI-powered portfolio management platform featuring automated financial market analysis and intelligent investment insights.

## Overview

PortfolioPilot is a full-stack application that combines FastAPI backend services with a Next.js frontend to deliver comprehensive portfolio management capabilities. The platform leverages LLM agents, sentiment analysis, and automated data pipelines to provide daily market insights and investment recommendations.

## Key Features

### 🤖 LLM Agent for Financial Market Analysis

An automated "investment digest" system that:

- **Morning Data Collection**: Automatically gathers financial data from multiple sources:
  - Yahoo Finance (market data, stock prices, indices)
  - FRED (Federal Reserve Economic Data)
  - REST APIs (real-time market feeds)

- **News Sentiment Analysis**: 
  - Uses T5/HuggingFace models for sentiment analysis
  - Processes financial news and market updates
  - Identifies key market trends and sentiment shifts

- **Intelligent Digest Generation**:
  - LLM-powered analysis via LangChain or MCP
  - Generates concise daily market summaries
  - Highlights key risks and opportunities
  - Example output: *"S&P500 grows by 1.2% due to CPI report... Here are the key risks..."*

- **Multi-Channel Distribution**:
  - Notion integration for structured documentation
  - Telegram notifications for real-time alerts
  - Automated workflows via n8n

## Technology Stack

### Backend
- **Framework**: FastAPI (Python 3.12+)
- **Agent Framework**: LangChain, LangGraph
- **LLM Integration**: OpenAI, Anthropic, Google Gemini (Vertex AI)
- **ML/AI**: 
  - PyTorch, Transformers (HuggingFace)
  - T5 models for sentiment analysis
- **Data Processing**: Pandas, NumPy, scikit-learn
- **Cloud Infrastructure**: 
  - Google Cloud Platform (GCP)
  - Cloud Functions for serverless execution
  - BigQuery for data warehousing
  - Cloud Storage for document storage
- **Workflow Automation**: n8n
- **Database**: PostgreSQL (via SQLModel)
- **Package Management**: uv

### Frontend
- **Framework**: Next.js (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **State Management**: React Server Components, Server Actions

## Project Structure

This is a monorepo containing both backend and frontend applications:

```
PortfolioPilot/
├── backend/              # Python FastAPI backend
│   ├── src/
│   │   └── portfoliopilot/
│   ├── tests/
│   └── pyproject.toml
├── frontend/            # Next.js frontend
│   ├── app/
│   ├── components/
│   └── lib/
└── README.md
```

## Getting Started

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Set up environment variables (create `.env` file):
```bash
# API Keys
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_APPLICATION_CREDENTIALS=path/to/gcp-credentials.json

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/portfoliopilot

# n8n Webhook URLs
N8N_NOTION_WEBHOOK=your_webhook_url
N8N_TELEGRAM_WEBHOOK=your_webhook_url
```

4. Run the application:
```bash
uvicorn portfoliopilot.main:app --reload
```

5. Run tests:
```bash
uv run pytest
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
# or
pnpm install
```

3. Set up environment variables (create `.env.local`):
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

4. Run the development server:
```bash
npm run dev
```

## Architecture Highlights

### LLM Pipeline Components

1. **Data Collection Layer**: Scheduled jobs that fetch market data from Yahoo Finance, FRED, and other APIs
2. **Sentiment Analysis**: T5-based models process news articles and market updates
3. **LLM Analysis**: LangChain agents synthesize data and generate insights
4. **ETL Pipeline**: Transform and load data into BigQuery for historical analysis
5. **Distribution Layer**: n8n workflows push insights to Notion and Telegram

### Key Capabilities for CV/Portfolio

- ✅ **Automation**: Fully automated daily market analysis pipeline
- ✅ **LLM Pipeline**: End-to-end LangChain-based agentic workflows
- ✅ **Retrieval & Summarization**: RAG patterns for financial data analysis
- ✅ **ETL**: Data extraction, transformation, and loading from multiple sources
- ✅ **Cloud Infrastructure**: Serverless functions, data warehousing, vector search
- ✅ **ML Integration**: Fine-tuned sentiment analysis models

## Development Guidelines

- **Backend**: Follow FastAPI best practices, use async/await patterns, strict Pydantic validation
- **Frontend**: Functional React components, TypeScript interfaces matching backend schemas
- **Type Safety**: Ensure TypeScript interfaces match Pydantic models exactly
- **Testing**: Write comprehensive pytest tests for backend, maintain test coverage

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Running Scripts

```bash
uv run script_name.py
```

## Running Tests for backend

```bash
 uv run --env-file ../.env pytest
```

## Running Tests for frontend
```bash
```