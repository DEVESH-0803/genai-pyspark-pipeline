"""Execution script for running PySpark sales analytics.

This script initializes the SalesAnalytics engine, loads Parquet datasets
from data/raw/, executes category sales, top customer rankings, and monthly
trends with MoM growth calculations, profiles the execution time for each
operation, displays formatted results, and safely stops the SparkSession.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from src.config import config, setup_logging
from src.spark_analytics import SalesAnalytics

logger = setup_logging(log_level="INFO", logger_name="RunAnalytics")


def format_duration(seconds: float) -> str:
    """Formats an elapsed duration into a human-readable string.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        str: Formatted duration (e.g., '1.24s', '1m 12.30s').
    """
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}m {remaining_seconds:.2f}s"


def main() -> int:
    """Orchestrates loading, analytical queries, timing profiling, and display.

    Returns:
        int: Process exit code (0 for success, 1 on failure).
    """
    print("\n" + "=" * 90)
    print(" " * 26 + "PYSPARK SALES ANALYTICS RUNNER")
    print("=" * 90)

    total_start_time = time.perf_counter()
    timings: Dict[str, str] = {}

    # Define paths to raw Parquet files
    raw_dir = config.paths.raw_dir
    customers_path = raw_dir / "customers.parquet"
    products_path = raw_dir / "products.parquet"
    orders_path = raw_dir / "orders.parquet"

    # Verify input datasets exist
    missing_files = [p.name for p in (customers_path, products_path, orders_path) if not p.exists()]
    if missing_files:
        logger.error(
            "Missing required input files in %s: %s. Run `python main.py` first.",
            raw_dir,
            ", ".join(missing_files),
        )
        return 1

    analytics: Optional[SalesAnalytics] = None

    try:
        # ------------------------------------------------------------------
        # 1. Create Spark Session
        # ------------------------------------------------------------------
        logger.info("Initializing PySpark session (4GB memory, AQE, Kryo)...")
        spark_start = time.perf_counter()
        analytics = SalesAnalytics(app_name="EcommerceSalesAnalytics")
        spark_init_time = time.perf_counter() - spark_start
        timings["Spark Session Init"] = format_duration(spark_init_time)
        logger.info("SparkSession ready in %s", timings["Spark Session Init"])

        # ------------------------------------------------------------------
        # 2. Load Parquet Datasets
        # ------------------------------------------------------------------
        logger.info("Loading Parquet datasets from %s...", raw_dir)
        load_start = time.perf_counter()

        customers_df = analytics.load_parquet(customers_path)
        products_df = analytics.load_parquet(products_path)
        orders_df = analytics.load_parquet(orders_path)

        # Trigger count to profile actual parquet load/metadata scan
        cust_count = customers_df.count()
        prod_count = products_df.count()
        ord_count = orders_df.count()

        load_time = time.perf_counter() - load_start
        timings["Load Parquet Data"] = format_duration(load_time)

        logger.info(
            "Loaded datasets in %s: Customers=%s | Products=%s | Orders=%s",
            timings["Load Parquet Data"],
            f"{cust_count:,}",
            f"{prod_count:,}",
            f"{ord_count:,}",
        )

        # ------------------------------------------------------------------
        # 3. Analytics Method 1: Sales by Category
        # ------------------------------------------------------------------
        print("\n" + "-" * 90)
        print(" [1/3] SALES & UNITS SOLD BY PRODUCT CATEGORY")
        print("-" * 90)

        cat_start = time.perf_counter()
        category_df = analytics.sales_by_category(orders_df, products_df)
        category_df.show(truncate=False)
        cat_time = time.perf_counter() - cat_start
        timings["Sales by Category"] = format_duration(cat_time)
        print(f"--> Computation & Display Time: {timings['Sales by Category']}")

        # ------------------------------------------------------------------
        # 4. Analytics Method 2: Top Customers by Lifetime Revenue
        # ------------------------------------------------------------------
        print("\n" + "-" * 90)
        print(" [2/3] TOP 10 CUSTOMERS BY REVENUE (Pareto 80/20 Analyzed)")
        print("-" * 90)

        cust_start = time.perf_counter()
        top_cust_df = analytics.top_customers_by_revenue(orders_df, products_df, n=10)
        top_cust_df.show(truncate=False)
        cust_time = time.perf_counter() - cust_start
        timings["Top Customers (n=10)"] = format_duration(cust_time)
        print(f"--> Computation & Display Time: {timings['Top Customers (n=10)']}")

        # ------------------------------------------------------------------
        # 5. Analytics Method 3: Monthly Trends & MoM Growth %
        # ------------------------------------------------------------------
        print("\n" + "-" * 90)
        print(" [3/3] MONTHLY SALES TRENDS & MoM GROWTH PERCENTAGE")
        print("-" * 90)

        trend_start = time.perf_counter()
        monthly_df = analytics.monthly_trends(orders_df, products_df)
        monthly_df.show(24, truncate=False)
        trend_time = time.perf_counter() - trend_start
        timings["Monthly Trends & MoM"] = format_duration(trend_time)
        print(f"--> Computation & Display Time: {timings['Monthly Trends & MoM']}")

        # ------------------------------------------------------------------
        # 6. Overall Performance Timing Summary
        # ------------------------------------------------------------------
        total_time = time.perf_counter() - total_start_time

        print("\n" + "=" * 90)
        print(" " * 30 + "ANALYTICS TIMING SUMMARY")
        print("=" * 90)
        for operation, duration in timings.items():
            print(f" • {operation:<30} : {duration:>10}")
        print("-" * 90)
        print(f" • {'Total End-to-End Runtime':<30} : {format_duration(total_time):>10}")
        print("=" * 90 + "\n")

        return 0

    except KeyboardInterrupt:
        logger.warning("\nAnalytics execution interrupted by user.")
        return 130

    except Exception as exc:
        logger.error("An error occurred during analytics execution: %s", exc, exc_info=True)
        return 1

    finally:
        # ------------------------------------------------------------------
        # 7. Gracefully Stop Spark Session
        # ------------------------------------------------------------------
        if analytics is not None:
            logger.info("Stopping PySpark session...")
            analytics.stop()


if __name__ == "__main__":
    sys.exit(main())

