from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from arxiv_ingestion.fetching import fetch_historical_papers
from arxiv_ingestion.indexing import index_papers_hybrid_with_task_id, verify_hybrid_index
from arxiv_ingestion.reporting import generate_report_with_task_ids

# Import task functions from modular structure
from arxiv_ingestion.setup import setup_environment

# Default DAG arguments
default_args = {
    "owner": "arxiv-curator",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=30),
    "catchup": False,
}

# Historical date range: January 2024 to October 31, 2024
HISTORICAL_FROM_DATE = "20240101"
HISTORICAL_TO_DATE = "20251031"

# Categories to fetch: cs.AI (default) + stat.ML + cs.LG
HISTORICAL_CATEGORIES = ["cs.AI", "stat.ML", "cs.LG"]


def fetch_historical_papers_task(**context):
    """Wrapper function to call fetch_historical_papers with the configured parameters.
    
    Note: For 10 months of data across 3 categories, we use max_results=2000 (arXiv API limit).
    If more papers are needed, consider splitting the date range or implementing pagination.
    """
    return fetch_historical_papers(
        from_date=HISTORICAL_FROM_DATE,
        to_date=HISTORICAL_TO_DATE,
        categories=HISTORICAL_CATEGORIES,
        max_results=1000,  # Use API limit for historical data
        **context
    )


# Create the DAG
dag = DAG(
    "arxiv_historical_paper_ingestion",
    default_args=default_args,
    description=f"Historical arXiv paper pipeline ({HISTORICAL_FROM_DATE} to {HISTORICAL_TO_DATE}): "
                f"fetch → store to PostgreSQL → chunk & embed → hybrid OpenSearch indexing. "
                f"Categories: {', '.join(HISTORICAL_CATEGORIES)}",
    schedule=None,  # Manual trigger only - this is a one-time historical ingestion
    max_active_runs=1,
    catchup=False,
    tags=["arxiv", "papers", "ingestion", "hybrid-search", "embeddings", "chunks", "historical"],
)

# Task definitions
setup_task = PythonOperator(
    task_id="setup_environment",
    python_callable=setup_environment,
    dag=dag,
)

fetch_task = PythonOperator(
    task_id="fetch_historical_papers",
    python_callable=fetch_historical_papers_task,
    dag=dag,
)

# Hybrid search indexing task (replaces old OpenSearch task)
# Use the flexible version that accepts task_id parameter
def index_historical_papers_hybrid(**context):
    """Wrapper to index historical papers using the correct task ID."""
    return index_papers_hybrid_with_task_id(task_id="fetch_historical_papers", **context)


index_hybrid_task = PythonOperator(
    task_id="index_papers_hybrid",
    python_callable=index_historical_papers_hybrid,
    dag=dag,
)

# Reporting task - use flexible version with correct task IDs
def generate_historical_report(**context):
    """Wrapper to generate report using the correct task IDs."""
    return generate_report_with_task_ids(
        fetch_task_id="fetch_historical_papers",
        index_task_id="index_papers_hybrid",
        **context
    )


report_task = PythonOperator(
    task_id="generate_daily_report",
    python_callable=generate_historical_report,
    dag=dag,
)

cleanup_task = BashOperator(
    task_id="cleanup_temp_files",
    bash_command="""
    echo "Cleaning up temporary files..."
    # Remove PDFs older than 30 days to manage disk space
    find /tmp -name "*.pdf" -type f -mtime +30 -delete 2>/dev/null || true
    echo "Cleanup completed"
    """,
    dag=dag,
)

# Task dependencies
# Simplified pipeline: setup -> fetch -> hybrid index -> report -> cleanup
setup_task >> fetch_task >> index_hybrid_task >> report_task >> cleanup_task

