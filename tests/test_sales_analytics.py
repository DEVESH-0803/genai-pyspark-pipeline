"""Unit tests for the SalesAnalytics PySpark module."""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from src.spark_analytics import SalesAnalytics


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """Session-scoped SparkSession fixture for testing.

    Returns:
        SparkSession: Active test session.
    """
    spark_session = SalesAnalytics.create_spark_session(
        app_name="TestSalesAnalytics",
        driver_memory="2g",
        master="local[2]",
    )
    yield spark_session
    spark_session.stop()


@pytest.fixture
def sample_data(spark: SparkSession):
    """Fixture creating small mock orders and products DataFrames for analytical testing."""
    products_schema = StructType(
        [
            StructField("product_id", StringType(), False),
            StructField("name", StringType(), True),
            StructField("category", StringType(), True),
            StructField("price", DoubleType(), True),
        ]
    )
    products_data = [
        ("P1", "Wireless Mouse", "Electronics", 20.0),
        ("P2", "Denim Jeans", "Clothing", 50.0),
        ("P3", "Yoga Mat", "Sports", 30.0),
    ]
    products_df = spark.createDataFrame(products_data, schema=products_schema)

    orders_schema = StructType(
        [
            StructField("order_id", StringType(), False),
            StructField("customer_id", StringType(), False),
            StructField("product_id", StringType(), False),
            StructField("quantity", IntegerType(), True),
            StructField("order_date", StringType(), True),
        ]
    )
    orders_data = [
        ("O1", "C1", "P1", 2, "2024-01-15 10:00:00"),  # $40
        ("O2", "C1", "P2", 1, "2024-01-20 12:00:00"),  # $50 -> C1 total $90
        ("O3", "C2", "P3", 3, "2024-02-10 14:00:00"),  # $90 -> C2 total $90
        ("O4", "C3", "P1", 1, "2024-02-15 16:00:00"),  # $20 -> C3 total $20
    ]
    orders_df = spark.createDataFrame(orders_data, schema=orders_schema)

    return orders_df, products_df


def test_sales_by_category(sample_data) -> None:
    """Tests revenue and units sold aggregations by category."""
    orders_df, products_df = sample_data
    result_df = SalesAnalytics.sales_by_category(orders_df, products_df)

    rows = {row["category"]: row for row in result_df.collect()}
    assert len(rows) == 3

    # Check Sports: 3 units * $30 = $90
    assert float(rows["Sports"]["total_revenue"]) == 90.0
    assert rows["Sports"]["total_units_sold"] == 3

    # Check Electronics: (2 * $20) + (1 * $20) = $60
    assert float(rows["Electronics"]["total_revenue"]) == 60.0
    assert rows["Electronics"]["total_units_sold"] == 3

    # Check Clothing: 1 unit * $50 = $50
    assert float(rows["Clothing"]["total_revenue"]) == 50.0
    assert rows["Clothing"]["total_units_sold"] == 1


def test_top_customers_by_revenue(sample_data) -> None:
    """Tests ranking of customers by lifetime spend."""
    orders_df, products_df = sample_data
    result_df = SalesAnalytics.top_customers_by_revenue(orders_df, products_df, n=2)

    rows = result_df.collect()
    assert len(rows) == 2

    # Top customers are C1 ($90) and C2 ($90)
    top_spend = float(rows[0]["total_revenue"])
    assert top_spend == 90.0
    assert rows[0]["customer_id"] in {"C1", "C2"}


def test_monthly_trends(sample_data) -> None:
    """Tests chronological monthly revenue aggregation and MoM growth calculation."""
    orders_df, products_df = sample_data
    result_df = SalesAnalytics.monthly_trends(orders_df, products_df)

    rows = result_df.collect()
    assert len(rows) == 2

    # 2024-01: O1 ($40) + O2 ($50) = $90
    assert rows[0]["year_month"] == "2024-01"
    assert float(rows[0]["monthly_revenue"]) == 90.0
    assert rows[0]["mom_growth_pct"] is None  # First month has no previous month

    # 2024-02: O3 ($90) + O4 ($20) = $110
    assert rows[1]["year_month"] == "2024-02"
    assert float(rows[1]["monthly_revenue"]) == 110.0
    # MoM growth: ((110 - 90) / 90) * 100 = 22.22%
    assert float(rows[1]["mom_growth_pct"]) == 22.22

