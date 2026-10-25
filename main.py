"""Main execution script for E-Commerce Synthetic Data Generation.

This script initializes the SyntheticDataGenerator using project configurations,
synthesizes customers, products, and orders datasets, saves them as compressed
Parquet files in data/raw/, displays execution timings and file sizes, and
implements robust exception handling.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src.config import config, setup_logging
from src.data_generator import SyntheticDataGenerator

logger = setup_logging(log_level="INFO", logger_name="DataPipelineMain")


def format_file_size(size_in_bytes: int) -> str:
    """Formats raw byte counts into human-readable file size strings.

    Args:
        size_in_bytes: File size in bytes.

    Returns:
        str: Formatted size string (e.g., '12.45 MB', '850.20 KB').

    Example:
        >>> format_file_size(1048576)
        '1.00 MB'
    """
    if size_in_bytes < 0:
        raise ValueError("File size cannot be negative.")

    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_in_bytes)
    unit_index = 0

    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1

    return f"{size:.2f} {units[unit_index]}"


def format_duration(seconds: float) -> str:
    """Formats a duration in seconds into a human-readable string.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        str: Formatted duration (e.g., '4.32s', '1m 24.15s').
    """
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}m {remaining_seconds:.2f}s"


def save_as_parquet(
    df: pd.DataFrame,
    destination_path: Path,
    compression: str = "snappy",
) -> int:
    """Saves a pandas DataFrame to a Parquet file and returns its size in bytes.

    Args:
        df: The pandas DataFrame to serialize.
        destination_path: Path where the Parquet file should be saved.
        compression: Compression codec ('snappy', 'gzip', 'zstd', or None).

    Returns:
        int: Size of the created file in bytes.

    Raises:
        IOError: If writing to the destination fails.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Writing %d rows to Parquet file: %s", len(df), destination_path)

    df.to_parquet(
        destination_path,
        index=False,
        engine="pyarrow",
        compression=compression,
    )

    if not destination_path.exists():
        raise IOError(f"Failed to create Parquet file at {destination_path}")

    return destination_path.stat().st_size


def generate_and_save_pipeline() -> int:
    """Orchestrates generation, serialization to Parquet, and metrics reporting.

    Returns:
        int: Exit status code (0 for success, 1 for failure).
    """
    logger.info("=" * 80)
    logger.info("STARTING SYNTHETIC DATA GENERATION & PARQUET EXPORT PIPELINE")
    logger.info("=" * 80)

    total_start_time = time.perf_counter()

    try:
        # 1. Ensure target directory structure exists
        config.paths.ensure_directories()
        raw_dir = config.paths.raw_dir
        gen_cfg = config.generator

        logger.info(
            "Configuration loaded: [Customers: %s | Products: %s | Orders: %s | Seed: %d]",
            f"{gen_cfg.num_customers:,}",
            f"{gen_cfg.num_products:,}",
            f"{gen_cfg.num_orders:,}",
            gen_cfg.random_seed,
        )
        logger.info("Destination directory: %s", raw_dir)

        # 2. Instantiate SyntheticDataGenerator
        generator = SyntheticDataGenerator(
            num_customers=gen_cfg.num_customers,
            num_products=gen_cfg.num_products,
            num_orders=gen_cfg.num_orders,
            seed=gen_cfg.random_seed,
        )

        metrics: Dict[str, Dict[str, str]] = {}

        # ------------------------------------------------------------------
        # Step A: Customers Generation & Parquet Serialization
        # ------------------------------------------------------------------
        logger.info("--> [1/3] Generating Customers...")
        cust_gen_start = time.perf_counter()
        customers_df = generator.generate_customers()
        cust_gen_time = time.perf_counter() - cust_gen_start

        customers_parquet_path = raw_dir / "customers.parquet"
        cust_save_start = time.perf_counter()
        cust_bytes = save_as_parquet(customers_df, customers_parquet_path)
        cust_save_time = time.perf_counter() - cust_save_start

        metrics["customers"] = {
            "rows": f"{len(customers_df):,}",
            "cols": str(len(customers_df.columns)),
            "gen_time": format_duration(cust_gen_time),
            "save_time": format_duration(cust_save_time),
            "file_size": format_file_size(cust_bytes),
            "file_name": customers_parquet_path.name,
        }

        # ------------------------------------------------------------------
        # Step B: Products Generation & Parquet Serialization
        # ------------------------------------------------------------------
        logger.info("--> [2/3] Generating Products...")
        prod_gen_start = time.perf_counter()
        products_df = generator.generate_products()
        prod_gen_time = time.perf_counter() - prod_gen_start

        products_parquet_path = raw_dir / "products.parquet"
        prod_save_start = time.perf_counter()
        prod_bytes = save_as_parquet(products_df, products_parquet_path)
        prod_save_time = time.perf_counter() - prod_save_start

        metrics["products"] = {
            "rows": f"{len(products_df):,}",
            "cols": str(len(products_df.columns)),
            "gen_time": format_duration(prod_gen_time),
            "save_time": format_duration(prod_save_time),
            "file_size": format_file_size(prod_bytes),
            "file_name": products_parquet_path.name,
        }

        # ------------------------------------------------------------------
        # Step C: Orders Generation (Pareto 80/20) & Parquet Serialization
        # ------------------------------------------------------------------
        logger.info("--> [3/3] Generating Orders (Pareto 80/20 distribution)...")
        ord_gen_start = time.perf_counter()
        orders_df = generator.generate_orders(
            customers_df=customers_df,
            products_df=products_df,
        )
        ord_gen_time = time.perf_counter() - ord_gen_start

        orders_parquet_path = raw_dir / "orders.parquet"
        ord_save_start = time.perf_counter()
        ord_bytes = save_as_parquet(orders_df, orders_parquet_path)
        ord_save_time = time.perf_counter() - ord_save_start

        metrics["orders"] = {
            "rows": f"{len(orders_df):,}",
            "cols": str(len(orders_df.columns)),
            "gen_time": format_duration(ord_gen_time),
            "save_time": format_duration(ord_save_time),
            "file_size": format_file_size(ord_bytes),
            "file_name": orders_parquet_path.name,
        }

        total_elapsed = time.perf_counter() - total_start_time

        # ------------------------------------------------------------------
        # Step D: Print Formatted Pipeline Summary Table
        # ------------------------------------------------------------------
        total_bytes = cust_bytes + prod_bytes + ord_bytes

        print("\n" + "=" * 90)
        print(" " * 28 + "PIPELINE GENERATION SUMMARY")
        print("=" * 90)
        header = f"{'Dataset':<12} | {'Rows':<12} | {'Cols':<6} | {'Gen Time':<10} | {'Save Time':<10} | {'Parquet Size':<14} | {'Target File'}"
        print(header)
        print("-" * 90)

        for name, data in metrics.items():
            print(
                f"{name.capitalize():<12} | "
                f"{data['rows']:<12} | "
                f"{data['cols']:<6} | "
                f"{data['gen_time']:<10} | "
                f"{data['save_time']:<10} | "
                f"{data['file_size']:<14} | "
                f"{data['file_name']}"
            )

        print("-" * 90)
        print(f"Total Combined Parquet Storage : {format_file_size(total_bytes)}")
        print(f"Total End-to-End Pipeline Time : {format_duration(total_elapsed)}")
        print("=" * 90 + "\n")

        logger.info(
            "Pipeline successfully completed in %s. Raw Parquet files stored in %s.",
            format_duration(total_elapsed),
            raw_dir,
        )
        return 0

    except KeyboardInterrupt:
        logger.warning("\nPipeline generation aborted by user (KeyboardInterrupt).")
        return 130

    except Exception as exc:
        logger.error(
            "Fatal error occurred during pipeline execution: %s",
            exc,
            exc_info=True,
        )
        print(f"\n[ERROR] Pipeline execution failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(generate_and_save_pipeline())

