from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agents.base import Agent

# Create FastAPI application
app = FastAPI(
    title="PortfolioPilot API",
    description="AI-powered portfolio optimization and analysis",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure this based on environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "PortfolioPilot API"}


@app.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {"message": "PortfolioPilot API is running"}


def main() -> None:
    """Main entry point for non-server execution."""
    agent = Agent()
    print(agent.run())


if __name__ == "__main__":
    main()
