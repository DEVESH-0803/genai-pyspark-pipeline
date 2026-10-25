"""Unit tests for the SyntheticDataGenerator class."""

from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from src.data_generator import SyntheticDataGenerator


@pytest.fixture
def test_generator() -> SyntheticDataGenerator:
    """Fixture providing a scaled-down SyntheticDataGenerator for fast unit testing.

    Returns:
        SyntheticDataGenerator: Test generator with small record counts.
    """
    return SyntheticDataGenerator(
        num_customers=50,
        num_products=20,
        num_orders=200,
        seed=42,
        batch_size=25,
    )


def test_customer_generation_schema_and_distribution(
    test_generator: SyntheticDataGenerator,
) -> None:
    """Verifies customer DataFrame schema, types, and age distribution around 35."""
    df = test_generator.generate_customers(count=100)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100

    expected_cols = [
        "customer_id",
        "name",
        "email",
        "age",
        "city",
        "country",
        "registration_date",
    ]
    assert list(df.columns) == expected_cols

    # Check age distribution: Normal around 35
    assert 25 <= df["age"].mean() <= 45
    assert df["age"].min() >= 18
    assert df["age"].max() <= 75

    # Check formatting of customer_id
    assert df["customer_id"].iloc[0].startswith("CUST_")


def test_product_generation_schema_and_categories(
    test_generator: SyntheticDataGenerator,
) -> None:
    """Verifies product DataFrame schema, 5 categories, prices (10-500), and ratings (1-5)."""
    df = test_generator.generate_products(count=50)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50

    expected_cols = ["product_id", "name", "category", "price", "stock", "rating"]
    assert list(df.columns) == expected_cols

    expected_categories = {"Electronics", "Clothing", "Home", "Sports", "Books"}
    assert set(df["category"].unique()).issubset(expected_categories)

    # Price bounds: 10 - 500
    assert df["price"].min() >= 10.0
    assert df["price"].max() <= 500.0

    # Stock bounds: >= 0
    assert df["stock"].min() >= 0

    # Rating bounds: 1.0 - 5.0
    assert df["rating"].min() >= 1.0
    assert df["rating"].max() <= 5.0


def test_order_generation_and_pareto_distribution() -> None:
    """Verifies orders DataFrame schema, foreign keys, and Pareto 80/20 order distribution."""
    # Scale up slightly to check statistical Pareto distribution
    gen = SyntheticDataGenerator(
        num_customers=1000,
        num_products=50,
        num_orders=5000,
        seed=123,
    )

    customers_df = gen.generate_customers()
    products_df = gen.generate_products()
    orders_df = gen.generate_orders(
        customers_df=customers_df,
        products_df=products_df,
    )

    assert isinstance(orders_df, pd.DataFrame)
    assert len(orders_df) == 5000

    expected_cols = ["order_id", "customer_id", "product_id", "quantity", "order_date"]
    assert list(orders_df.columns) == expected_cols

    # Foreign key checks
    assert set(orders_df["customer_id"]).issubset(set(customers_df["customer_id"]))
    assert set(orders_df["product_id"]).issubset(set(products_df["product_id"]))

    # Quantity checks: 1-10
    assert orders_df["quantity"].min() >= 1
    assert orders_df["quantity"].max() <= 10

    # Pareto 80/20 Verification: Top 20% of customers should make ~80% of orders (within 70-90% tolerance)
    cust_order_counts = orders_df["customer_id"].value_counts().values
    top_20_pct_count = int(0.20 * len(cust_order_counts))
    top_20_share = (cust_order_counts[:top_20_pct_count].sum() / len(orders_df)) * 100

    assert 70.0 <= top_20_share <= 90.0, f"Expected ~80% Pareto share, got {top_20_share:.2f}%"


def test_order_dates_referential_integrity(
    test_generator: SyntheticDataGenerator,
) -> None:
    """Verifies that every order date occurs strictly on or after the customer registration date."""
    customers_df = test_generator.generate_customers(count=30)
    products_df = test_generator.generate_products(count=10)
    orders_df = test_generator.generate_orders(
        count=100,
        customers_df=customers_df,
        products_df=products_df,
    )

    reg_map = dict(zip(customers_df["customer_id"], customers_df["registration_date"]))

    for _, row in orders_df.iterrows():
        reg_dt = datetime.strptime(reg_map[row["customer_id"]], "%Y-%m-%d")
        order_dt = datetime.strptime(row["order_date"], "%Y-%m-%d %H:%M:%S")
        assert order_dt >= reg_dt


def test_main_utility_functions(tmp_path) -> None:
    """Tests file size formatting, duration formatting, and Parquet saving from main.py."""
    from main import format_duration, format_file_size, save_as_parquet

    assert format_file_size(500) == "500.00 B"
    assert format_file_size(1024) == "1.00 KB"
    assert format_file_size(1048576) == "1.00 MB"

    assert format_duration(15.5) == "15.50s"
    assert format_duration(95.0) == "1m 35.00s"

    sample_df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
    test_parquet_path = tmp_path / "test.parquet"
    file_size = save_as_parquet(sample_df, test_parquet_path)

    assert test_parquet_path.exists()
    assert file_size > 0
