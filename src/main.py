import logging
import os
import time # import time for sleep
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from src.config import get_settings
from src.db.factory import make_database
from src.routers import hybrid_search, ping
from src.routers.ask import ask_router, stream_router
from src.services.arxiv.factory import make_arxiv_client
from src.services.cache.factory import make_cache_client
from src.services.embeddings.factory import make_embeddings_service
from src.services.langfuse.factory import make_langfuse_tracer
from src.services.ollama.factory import make_ollama_client
from src.services.opensearch.factory import make_opensearch_client
from src.services.pdf_parser.factory import make_pdf_parser_service
from opensearchpy.exceptions import NotFoundError # import NotFoundError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan for the API.
    """
    logger.info("Starting RAG API...")

    settings = get_settings()
    app.state.settings = settings

    # Database connection
    database = make_database()
    app.state.database = database
    logger.info("Database connected")

    # Initialize OpenSearch
    opensearch_client = make_opensearch_client()
    app.state.opensearch_client = opensearch_client

    # Determine if we should force recreate indices (dev only)
    force_index = settings.environment.lower() == "development"

    # Verify OpenSearch connectivity and setup index safely
    if opensearch_client.health_check():
        logger.info("OpenSearch connected successfully")

        # Retry logic for OpenSearch index setup
        max_retries = 5
        retry_delay = 5  # seconds
        for i in range(max_retries):
            try:
                setup_results = opensearch_client.setup_indices(force=force_index)
                if setup_results.get("hybrid_index"):
                    if force_index:
                        logger.info("Hybrid index created/recreated (dev environment)")
                    else:
                        logger.info("Hybrid index exists (production)")
                else:
                    logger.info("Hybrid index already exists")
                break  # If successful, break out of the retry loop
            except NotFoundError as e:
                logger.warning(
                    f"OpenSearch index setup attempt {i+1}/{max_retries} failed with NotFoundError: {e}"
                )
                if i < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        "Max retries reached for OpenSearch index setup. Continuing anyway."
                    )
            except Exception as e:
                logger.error(
                    f"OpenSearch index setup failed with unexpected error: {e}"
                )
                break # For other errors, no point in retrying this specific block

        # Get simple statistics
        try:
            stats = opensearch_client.client.count(index=opensearch_client.index_name)
            logger.info(f"OpenSearch ready: {stats['count']} documents indexed")
        except Exception as e:
            logger.warning(f"OpenSearch index ready, but stats unavailable due to: {e}")
    else:
        logger.warning("OpenSearch connection failed - search features will be limited")

    # Initialize other services
    app.state.arxiv_client = make_arxiv_client()
    app.state.pdf_parser = make_pdf_parser_service()
    app.state.embeddings_service = make_embeddings_service()
    app.state.ollama_client = make_ollama_client()
    app.state.langfuse_tracer = make_langfuse_tracer()
    app.state.cache_client = make_cache_client(settings)
    logger.info("Services initialized: arXiv API client, PDF parser, OpenSearch, Embeddings, Ollama, Langfuse, Cache")

    logger.info("API ready")
    yield

    # Cleanup
    database.teardown()
    logger.info("API shutdown complete")

app = FastAPI(
    title="arXiv Paper Curator API",
    description="Personal arXiv CS.AI paper curator with RAG capabilities",
    version=os.getenv("APP_VERSION", "0.1.0"),
    lifespan=lifespan,
)

# Include routers
app.include_router(ping.router, prefix="/api/v1")  # Health check endpoint
app.include_router(hybrid_search.router, prefix="/api/v1")  # Search chunks with BM25/hybrid
app.include_router(ask_router, prefix="/api/v1")  # RAG question answering with LLM
app.include_router(stream_router, prefix="/api/v1")  # Streaming RAG responses


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")