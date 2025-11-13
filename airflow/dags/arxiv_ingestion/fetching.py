import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import random

from .common import get_cached_services

logger = logging.getLogger(__name__)


def _filter_unprocessed_papers_by_db(papers: List[Any]) -> List[Any]:
    """Filter out papers that are already parsed/processed in PostgreSQL.

    A paper is considered processed if its record exists with pdf_processed=True.
    This avoids re-downloading and re-parsing PDFs unnecessarily.
    """
    try:
        _arxiv_client, _pdf_parser, database, _metadata_fetcher, _opensearch_client = get_cached_services()

        # Extract arxiv_ids from fetched metadata
        arxiv_ids = [p.arxiv_id for p in papers if getattr(p, "arxiv_id", None)]
        if not arxiv_ids:
            return papers

        with database.get_session() as session:
            from sqlalchemy import select
            from src.models.paper import Paper

            # Find those already processed
            stmt = select(Paper.arxiv_id).where(Paper.arxiv_id.in_(arxiv_ids), Paper.pdf_processed == True)
            processed_ids = set(x[0] for x in session.execute(stmt).all())

        if not processed_ids:
            return papers

        remaining = [p for p in papers if p.arxiv_id not in processed_ids]
        logger.info(
            f"Filtered already processed papers by DB: {len(processed_ids)} skipped, {len(remaining)}/{len(papers)} remaining"
        )
        return remaining
    except Exception as e:
        # Fail-open: if filtering fails, proceed with full list but log the issue
        logger.warning(f"DB filter for processed papers failed, proceeding without filter: {e}")
        return papers


async def run_paper_ingestion_pipeline(
    target_date: str,
    process_pdfs: bool = True,
) -> dict:
    """Async wrapper for the paper ingestion pipeline.

    :param target_date: Date to fetch papers for (YYYYMMDD format)
    :param process_pdfs: Whether to download and process PDFs
    :returns: Dictionary with ingestion statistics
    """
    arxiv_client, _, database, metadata_fetcher, _ = get_cached_services()

    max_results = arxiv_client.max_results
    logger.info(f"Using default max_results from config: {max_results}")

    with database.get_session() as session:
        return await metadata_fetcher.fetch_and_process_papers(
            max_results=max_results,
            from_date=target_date,
            to_date=target_date,
            process_pdfs=process_pdfs,
            store_to_db=True,
            db_session=session,
        )


def fetch_daily_papers(**context):
    """Fetch daily papers from arXiv and store in PostgreSQL.

    This task:
    1. Determines the target date (defaults to yesterday)
    2. Fetches papers from arXiv API
    3. Downloads and processes PDFs using Docling
    4. Stores metadata and parsed content in PostgreSQL

    Note: OpenSearch indexing is handled by a separate dedicated task
    """
    logger.info("Starting daily paper fetching task")

    execution_date = context.get("execution_date")
    if execution_date:
        target_dt = execution_date - timedelta(days=1)
        target_date = target_dt.strftime("%Y%m%d")
    else:
        yesterday = datetime.now() - timedelta(days=1)
        target_date = yesterday.strftime("%Y%m%d")

    logger.info(f"Fetching papers for date: {target_date}")

    results = asyncio.run(
        run_paper_ingestion_pipeline(
            target_date=target_date,
            process_pdfs=True,
        )
    )

    logger.info(f"Daily fetch complete: {results['papers_fetched']} papers for {target_date}")

    results["date"] = target_date
    ti = context.get("ti")
    if ti:
        ti.xcom_push(key="fetch_results", value=results)

    return results


async def run_historical_paper_ingestion_pipeline(
    from_date: str,
    to_date: str,
    categories: List[str],
    process_pdfs: bool = True,
    max_results: Optional[int] = None,
) -> dict:
    """Async wrapper for historical paper ingestion pipeline with multiple categories.

    :param from_date: Start date to fetch papers for (YYYYMMDD format)
    :param to_date: End date to fetch papers for (YYYYMMDD format)
    :param categories: List of arXiv categories to fetch (e.g., ["cs.AI", "stat.ML", "cs.LG"])
    :param process_pdfs: Whether to download and process PDFs
    :param max_results: Maximum number of papers to fetch (uses config default if None)
    :returns: Dictionary with ingestion statistics
    """
    arxiv_client, _, database, metadata_fetcher, _ = get_cached_services()

    if max_results is None:
        max_results = arxiv_client.max_results

    # Build search query with multiple categories
    # Format: (cat:cs.AI OR cat:stat.ML OR cat:cs.LG) AND submittedDate:[YYYYMMDD0000 TO YYYYMMDD2359]
    category_query = " OR ".join([f"cat:{cat}" for cat in categories])
    
    # Add date filtering
    date_from = f"{from_date}0000"
    date_to = f"{to_date}2359"
    search_query = f"({category_query}) AND submittedDate:[{date_from}+TO+{date_to}]"

    logger.info(f"Fetching historical papers with query: {search_query}")
    logger.info(f"Categories: {categories}, Date range: {from_date} to {to_date}")

    # Fetch papers using custom query (request more, then sample for variability)
    target_count = max_results
    request_limit = min(2000, max(1, target_count * 5))
    papers = await arxiv_client.fetch_papers_with_query(
        search_query=search_query,
        max_results=request_limit,
        sort_by="submittedDate",
        sort_order="ascending",  # Use ascending for historical data
    )

    # Shuffle and sample down to target_count to vary selection across runs
    random.shuffle(papers)
    if len(papers) > target_count:
        papers = papers[:target_count]
    logger.info(f"Fetched {len(papers)} papers from arXiv (after shuffle/sample)")

    # Filter out papers already processed in DB to avoid re-work
    papers = _filter_unprocessed_papers_by_db(papers)
    total_papers = len(papers)
    if total_papers == 0:
        logger.info("All fetched papers are already processed. Nothing to do.")
        return {
            "papers_fetched": 0,
            "pdfs_downloaded": 0,
            "pdfs_parsed": 0,
            "papers_stored": 0,
            "errors": [],
            "processing_time": 0,
        }

    if not papers:
        logger.warning("No papers found for the given criteria")
        return {
            "papers_fetched": 0,
            "pdfs_downloaded": 0,
            "pdfs_parsed": 0,
            "papers_stored": 0,
            "errors": [],
            "processing_time": 0,
        }

    # Process PDFs if requested
    pdf_results = {}
    if process_pdfs:
        # Track progress with a counter
        progress_counter = {"parsed": 0}
        
        async def process_with_progress(paper, download_sem, parse_sem):
            """Wrapper to track progress during PDF processing."""
            result = await metadata_fetcher._download_and_parse_pipeline(paper, download_sem, parse_sem)
            if result and isinstance(result, tuple) and result[1]:  # If parsed successfully
                progress_counter["parsed"] += 1
                logger.info(f"PDF processed: {progress_counter['parsed']}/{total_papers}")
            return result
        
        # Process with progress tracking
        download_semaphore = asyncio.Semaphore(metadata_fetcher.max_concurrent_downloads)
        parse_semaphore = asyncio.Semaphore(metadata_fetcher.max_concurrent_parsing)
        
        pipeline_tasks = [process_with_progress(paper, download_semaphore, parse_semaphore) for paper in papers]
        pipeline_results = await asyncio.gather(*pipeline_tasks, return_exceptions=True)
        
        # Convert results to the format expected by _store_papers_to_db
        parsed_papers = {}
        downloaded_count = 0
        parsed_count = 0
        errors = []
        
        for paper, result in zip(papers, pipeline_results):
            if isinstance(result, Exception):
                errors.append(f"Pipeline error for {paper.arxiv_id}: {str(result)}")
            elif result and isinstance(result, tuple):
                download_success, parsed_paper = result
                if download_success:
                    downloaded_count += 1
                if parsed_paper:
                    parsed_count += 1
                    parsed_papers[paper.arxiv_id] = parsed_paper
        
        pdf_results = {
            "downloaded": downloaded_count,
            "parsed": parsed_count,
            "parsed_papers": parsed_papers,
            "errors": errors,
        }

    # Store to database
    with database.get_session() as session:
        stored_count = metadata_fetcher._store_papers_to_db(
            papers=papers,
            parsed_papers=pdf_results.get("parsed_papers", {}),
            db_session=session,
        )

    results = {
        "papers_fetched": len(papers),
        "pdfs_downloaded": pdf_results.get("downloaded", 0) if process_pdfs else 0,
        "pdfs_parsed": pdf_results.get("parsed", 0) if process_pdfs else 0,
        "papers_stored": stored_count,
        "errors": pdf_results.get("errors", []) if process_pdfs else [],
        "processing_time": 0,  # Could add timing if needed
    }

    return results


def fetch_historical_papers(
    from_date: str,
    to_date: str,
    categories: List[str],
    max_results: Optional[int] = None,
    **context
):
    """Fetch historical papers from arXiv for a date range and multiple categories.

    This task:
    1. Fetches papers from arXiv API for the specified date range and categories
    2. Downloads and processes PDFs using Docling
    3. Stores metadata and parsed content in PostgreSQL

    :param from_date: Start date (YYYYMMDD format)
    :param to_date: End date (YYYYMMDD format)
    :param categories: List of arXiv categories (e.g., ["cs.AI", "stat.ML", "cs.LG"])
    :param max_results: Maximum number of papers to fetch (uses config default if None)
    """
    logger.info(f"Starting historical paper fetching task")
    logger.info(f"Date range: {from_date} to {to_date}")
    logger.info(f"Categories: {categories}")

    results = asyncio.run(
        run_historical_paper_ingestion_pipeline(
            from_date=from_date,
            to_date=to_date,
            categories=categories,
            process_pdfs=True,
            max_results=max_results,
        )
    )

    logger.info(
        f"Historical fetch complete: {results['papers_fetched']} papers fetched, "
        f"{results['papers_stored']} papers stored"
    )

    results["from_date"] = from_date
    results["to_date"] = to_date
    results["categories"] = categories
    
    ti = context.get("ti")
    if ti:
        ti.xcom_push(key="fetch_results", value=results)

    return results
