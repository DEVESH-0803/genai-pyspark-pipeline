"""Unit tests for configuration module."""

from pathlib import Path

from src.config import DataPaths, GeneratorConfig, SparkConfig, setup_logging


def test_data_paths_resolution() -> None:
    """Verifies that DataPaths resolves expected directory structures."""
    paths = DataPaths()

    assert paths.raw_dir == paths.base_dir / "data" / "raw"
    assert paths.processed_dir == paths.base_dir / "data" / "processed"
    assert paths.customers_csv.name == "customers.csv"
    assert paths.products_csv.name == "products.csv"
    assert paths.orders_csv.name == "orders.csv"
    assert paths.order_items_csv.name == "order_items.csv"


def test_generator_config_defaults() -> None:
    """Verifies default parameters for data generation."""
    gen_cfg = GeneratorConfig()

    assert gen_cfg.num_customers > 0
    assert gen_cfg.num_products > 0
    assert gen_cfg.num_orders > 0
    assert len(gen_cfg.categories) > 0
    assert len(gen_cfg.order_statuses) == len(gen_cfg.order_status_weights)


def test_spark_config_defaults() -> None:
    """Verifies default parameters for Spark cluster setup."""
    spark_cfg = SparkConfig()

    assert spark_cfg.master == "local[*]"
    assert spark_cfg.app_name == "EcommerceDataPipeline"


def test_setup_logging() -> None:
    """Verifies logger initialization with appropriate log level."""
    logger = setup_logging(log_level="DEBUG", logger_name="TestLogger")
    assert logger.name == "TestLogger"

