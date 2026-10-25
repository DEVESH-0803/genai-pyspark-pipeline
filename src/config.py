"""Configuration module for the E-commerce PySpark Data Pipeline.

This module provides path management, data generation parameters,
Apache Spark configuration, and unified logging setup.
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class DataPaths:
    """Manages all file and directory paths used across the pipeline."""

    base_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )

    @property
    def raw_dir(self) -> Path:
        """Directory for generated raw CSV data files."""
        return self.base_dir / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        """Directory for analyzed output data files."""
        return self.base_dir / "data" / "processed"

    @property
    def customers_csv(self) -> Path:
        """File path for the raw customers CSV dataset."""
        return self.raw_dir / "customers.csv"

    @property
    def products_csv(self) -> Path:
        """File path for the raw products CSV dataset."""
        return self.raw_dir / "products.csv"

    @property
    def orders_csv(self) -> Path:
        """File path for the raw orders CSV dataset."""
        return self.raw_dir / "orders.csv"

    @property
    def customers_parquet(self) -> Path:
        """File path for the raw customers Parquet dataset."""
        return self.raw_dir / "customers.parquet"

    @property
    def products_parquet(self) -> Path:
        """File path for the raw products Parquet dataset."""
        return self.raw_dir / "products.parquet"

    @property
    def orders_parquet(self) -> Path:
        """File path for the raw orders Parquet dataset."""
        return self.raw_dir / "orders.parquet"

    @property
    def order_items_csv(self) -> Path:
        """File path for the raw order items CSV dataset."""
        return self.raw_dir / "order_items.csv"

    def ensure_directories(self) -> None:
        """Ensure that raw and processed directories exist on the filesystem."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class GeneratorConfig:
    """Configuration parameters for synthetic data generation."""

    num_customers: int = 100_000
    num_products: int = 10_000
    num_orders: int = 1_000_000
    min_items_per_order: int = 1
    max_items_per_order: int = 5
    random_seed: int = 42

    categories: List[str] = field(
        default_factory=lambda: [
            "Electronics",
            "Clothing",
            "Home",
            "Sports",
            "Books",
        ]
    )

    order_statuses: List[str] = field(
        default_factory=lambda: [
            "Completed",
            "Shipped",
            "Processing",
            "Cancelled",
            "Refunded",
        ]
    )

    order_status_weights: List[float] = field(
        default_factory=lambda: [0.65, 0.15, 0.10, 0.06, 0.04]
    )

    payment_methods: List[str] = field(
        default_factory=lambda: [
            "Credit Card",
            "Debit Card",
            "PayPal",
            "Apple Pay",
            "Bank Transfer",
        ]
    )

    customer_segments: List[str] = field(
        default_factory=lambda: [
            "Consumer",
            "Corporate",
            "Small Business",
            "VIP",
        ]
    )


@dataclass
class SparkConfig:
    """Configuration settings for Apache Spark session."""

    app_name: str = "EcommerceDataPipeline"
    master: str = "local[*]"
    driver_memory: str = "2g"
    shuffle_partitions: str = "4"
    log_level: str = "WARN"


@dataclass
class PipelineConfig:
    """Master pipeline configuration aggregating paths, generator, and Spark settings."""

    paths: DataPaths = field(default_factory=DataPaths)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    spark: SparkConfig = field(default_factory=SparkConfig)


def setup_logging(
    log_level: str = "INFO",
    logger_name: Optional[str] = None,
) -> logging.Logger:
    """Configures and returns a standardized logger for pipeline modules.

    Args:
        log_level: The threshold level for logging (e.g. 'DEBUG', 'INFO', 'WARNING').
        logger_name: Name of the logger instance. Defaults to root logger if None.

    Returns:
        logging.Logger: A configured Python logger instance.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger(logger_name)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if not logger.handlers:
        logger.setLevel(numeric_level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


# Global default configuration instance
config = PipelineConfig()

