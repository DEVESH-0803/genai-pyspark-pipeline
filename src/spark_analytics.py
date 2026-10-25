"""PySpark Sales Analytics Module for E-Commerce Pipeline.

This module provides the `SalesAnalytics` class, configured for high-performance
distributed batch analytics using Apache Spark with 4GB memory, Adaptive Query
Execution (AQE), Kryo serialization, and PySpark SQL Window functions.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType

from src.config import config, setup_logging

logger = setup_logging(log_level="INFO", logger_name="SalesAnalytics")


class SalesAnalytics:
    """Distributed sales analytics engine built with Apache Spark (PySpark).

    Provides scalable aggregations, window analytics, and business insights
    over e-commerce orders and product catalogs. Optimized for high throughput
    in local or cluster mode with:
      - 4GB driver and executor memory
      - Adaptive Query Execution (AQE) enabled
      - Automatic partition coalescing
      - Kryo serialization for low serialization overhead
      - Vectorized Arrow execution

    Attributes:
        spark: Active Apache Spark session.
    """

    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        app_name: str = "SalesAnalytics",
    ) -> None:
        """Initializes SalesAnalytics with an active or newly created SparkSession.

        Args:
            spark: Optional pre-configured SparkSession. If None, initializes
                a new optimized session via `create_spark_session()`.
            app_name: Name of the Spark application.
        """
        self.spark: SparkSession = spark or self.create_spark_session(app_name=app_name)
        logger.info(
            "SalesAnalytics initialized successfully. Spark version: %s",
            self.spark.version,
        )

    def __enter__(self) -> "SalesAnalytics":
        """Enables context manager entry (`with SalesAnalytics() as sa:`)."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Enables context manager exit, automatically terminating the SparkSession."""
        self.stop()

    @staticmethod
    def create_spark_session(
        app_name: str = "SalesAnalytics",
        driver_memory: str = "4g",
        master: str = "local[*]",
    ) -> SparkSession:
        """Configures and builds an optimized PySpark SparkSession for local execution.

        Settings applied:
          - Memory: 4GB driver and executor memory ('spark.driver.memory', '4g')
          - Adaptive Query Execution: Enabled ('spark.sql.adaptive.enabled', 'true')
          - AQE Partition Coalescing: Enabled ('spark.sql.adaptive.coalescePartitions.enabled', 'true')
          - Serialization: Kryo ('spark.serializer', 'org.apache.spark.serializer.KryoSerializer')
          - Arrow execution: Enabled ('spark.sql.execution.arrow.pyspark.enabled', 'true')
          - Master: Local mode utilizing all available cores ('local[*]')

        Args:
            app_name: Application name displayed in Spark UI (default: 'SalesAnalytics').
            driver_memory: Memory allocated to Spark driver (default: '4g').
            master: Cluster connection string (default: 'local[*]').

        Returns:
            SparkSession: Fully configured active SparkSession.

        Example:
            >>> spark = SalesAnalytics.create_spark_session()
            >>> spark.version
            '4.2.0'
        """
        # Ensure PySpark workers bind to the active Python interpreter
        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

        logger.info(
            "Configuring SparkSession '%s' [Master: %s, Memory: %s, AQE: True, Kryo: True]...",
            app_name,
            master,
            driver_memory,
        )

        spark = (
            SparkSession.builder.appName(app_name)
            .master(master)
            .config("spark.driver.memory", driver_memory)
            .config("spark.executor.memory", driver_memory)
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        )

        spark.sparkContext.setLogLevel("WARN")
        return spark

    def load_parquet(self, path: Union[str, Path]) -> DataFrame:
        """Loads a Parquet file or directory into a PySpark DataFrame.

        Args:
            path: String or Path object pointing to the Parquet file or folder.

        Returns:
            DataFrame: PySpark DataFrame containing the loaded records.

        Raises:
            FileNotFoundError: If the specified path does not exist on disk.

        Example:
            >>> sa = SalesAnalytics()
            >>> df = sa.load_parquet("data/raw/orders.parquet")
            >>> df.count() > 0
            True
        """
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"Parquet path not found: {target_path}")

        logger.info("Loading Parquet dataset from: %s", target_path)
        df = self.spark.read.parquet(str(target_path))
        logger.debug(
            "Successfully loaded '%s' (%d columns)",
            target_path.name,
            len(df.columns),
        )
        return df

    @staticmethod
    def top_customers_by_revenue(
        orders_df: DataFrame,
        products_df: DataFrame,
        n: int = 10,
    ) -> DataFrame:
        """Joins orders with products, computes customer lifetime spend, and returns top N.

        Calculation:
          line_revenue = quantity * price
          total_revenue = sum(line_revenue)
          total_orders = countDistinct(order_id)
          total_units = sum(quantity)
          avg_order_value = total_revenue / total_orders

        Args:
            orders_df: Orders DataFrame containing 'order_id', 'customer_id',
                'product_id', 'quantity'.
            products_df: Products DataFrame containing 'product_id', 'price'.
            n: Number of top revenue-generating customers to return (default: 10).

        Returns:
            DataFrame: Top N customer records sorted descending by total_revenue.

        Example:
            >>> top_cust = SalesAnalytics.top_customers_by_revenue(orders, products, n=5)
            >>> top_cust.count()
            5
        """
        logger.info("Computing top %d customers by lifetime revenue...", n)

        # 1. Join orders with products to calculate line revenue
        joined_df = orders_df.join(
            products_df, on="product_id", how="inner"
        ).withColumn("line_revenue", F.col("quantity") * F.col("price"))

        # 2. Group by customer and aggregate metrics
        customer_summary = (
            joined_df.groupBy("customer_id")
            .agg(
                F.round(F.sum("line_revenue"), 2).cast(DecimalType(18, 2)).alias("total_revenue"),
                F.countDistinct("order_id").alias("total_orders"),
                F.sum("quantity").alias("total_units_purchased"),
                F.round(
                    F.sum("line_revenue") / F.countDistinct("order_id"), 2
                ).cast(DecimalType(10, 2)).alias("avg_order_value"),
            )
            .orderBy(F.desc("total_revenue"))
            .limit(n)
        )

        return customer_summary

    @staticmethod
    def sales_by_category(
        orders_df: DataFrame,
        products_df: DataFrame,
    ) -> DataFrame:
        """Groups by product category and sums revenue, units sold, and order counts.

        Computes:
          total_revenue: Sum of (quantity * price) rounded to 2 decimals
          total_units_sold: Sum of quantity
          order_count: Count of distinct orders in the category
          avg_unit_price: Average price of items sold in that category

        Args:
            orders_df: Orders DataFrame containing 'product_id' and 'quantity'.
            products_df: Products DataFrame containing 'product_id', 'category', 'price'.

        Returns:
            DataFrame: Aggregated category metrics sorted descending by total_revenue.

        Example:
            >>> cat_sales = SalesAnalytics.sales_by_category(orders, products)
            >>> cat_sales.show()
        """
        logger.info("Computing sales aggregations grouped by category...")

        # 1. Join orders with products
        joined_df = orders_df.join(
            products_df, on="product_id", how="inner"
        ).withColumn("line_revenue", F.col("quantity") * F.col("price"))

        # 2. Group by category and compute aggregates
        category_summary = (
            joined_df.groupBy("category")
            .agg(
                F.round(F.sum("line_revenue"), 2).cast(DecimalType(18, 2)).alias("total_revenue"),
                F.sum("quantity").alias("total_units_sold"),
                F.countDistinct("order_id").alias("order_count"),
                F.round(F.avg("price"), 2).cast(DecimalType(10, 2)).alias("avg_unit_price"),
            )
            .orderBy(F.desc("total_revenue"))
        )

        return category_summary

    @staticmethod
    def monthly_trends(
        orders_df: DataFrame,
        products_df: DataFrame,
    ) -> DataFrame:
        """Calculates chronological monthly revenue and Month-over-Month (MoM) growth %.

        Computes:
          monthly_revenue: Total sales within the year-month (YYYY-MM).
          orders_count: Number of distinct orders in the month.
          units_sold: Total item volume sold.
          mom_growth_pct: Percentage change in revenue compared to previous month
            calculated via `pyspark.sql.Window.orderBy('year_month')` with `F.lag`.

        Args:
            orders_df: Orders DataFrame containing 'order_date', 'product_id', 'quantity'.
            products_df: Products DataFrame containing 'product_id', 'price'.

        Returns:
            DataFrame: Chronological monthly revenue trend with MoM growth percentage.

        Example:
            >>> trends = SalesAnalytics.monthly_trends(orders, products)
            >>> trends.show()
        """
        logger.info("Computing monthly revenue trends and MoM growth percentage...")

        # 1. Enrich orders with price and extract year-month
        enriched_orders = (
            orders_df.join(products_df, on="product_id", how="inner")
            .withColumn("order_timestamp", F.to_timestamp("order_date"))
            .withColumn("year_month", F.date_format("order_timestamp", "yyyy-MM"))
            .withColumn("line_revenue", F.col("quantity") * F.col("price"))
        )

        # 2. Aggregate monthly metrics
        monthly_summary = enriched_orders.groupBy("year_month").agg(
            F.round(F.sum("line_revenue"), 2).cast(DecimalType(18, 2)).alias("monthly_revenue"),
            F.countDistinct("order_id").alias("orders_count"),
            F.sum("quantity").alias("units_sold"),
        )

        # 3. Window specification for sequential lag calculation
        window_spec = Window.orderBy("year_month")

        trend_df = (
            monthly_summary.withColumn(
                "prev_month_revenue", F.lag("monthly_revenue", 1).over(window_spec)
            )
            .withColumn(
                "mom_growth_pct",
                F.round(
                    (
                        (F.col("monthly_revenue") - F.col("prev_month_revenue"))
                        / F.col("prev_month_revenue")
                    )
                    * 100,
                    2,
                ),
            )
            .drop("prev_month_revenue")
            .orderBy("year_month")
        )

        return trend_df

    @staticmethod
    def save_dataframe(
        df: DataFrame,
        output_path: Union[str, Path],
        file_format: str = "parquet",
    ) -> None:
        """Saves a PySpark DataFrame to disk.

        Attempts native PySpark write first. If running on a Windows environment
        lacking Hadoop winutils binaries, falls back smoothly to pandas/pyarrow.

        Args:
            df: DataFrame to serialize.
            output_path: Destination file or folder path.
            file_format: Format codec ('parquet' or 'csv').
        """
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving processed dataset to %s (format=%s)...", target, file_format)

        try:
            writer = df.coalesce(1).write.mode("overwrite")
            if file_format.lower() == "parquet":
                writer.parquet(str(target))
            elif file_format.lower() == "csv":
                writer.option("header", "true").csv(str(target))
            else:
                raise ValueError(f"Unsupported format '{file_format}'. Choose 'parquet' or 'csv'.")
            logger.info("Successfully exported via Spark: %s", target.name)
        except Exception as exc:
            logger.warning(
                "Native Spark file write encountered system limitation (%s). "
                "Saving via pandas/pyarrow exporter...",
                exc,
            )
            pdf = df.toPandas()
            if file_format.lower() == "parquet":
                target_file = target.with_suffix(".parquet") if target.suffix != ".parquet" else target
                pdf.to_parquet(target_file, index=False, engine="pyarrow")
            else:
                target_file = target.with_suffix(".csv") if target.suffix != ".csv" else target
                pdf.to_csv(target_file, index=False)
            logger.info("Successfully exported via pandas/pyarrow: %s", target_file.name)

    def stop(self) -> None:
        """Gracefully terminates the active SparkSession."""
        if self.spark:
            logger.info("Terminating active SparkSession...")
            self.spark.stop()
            logger.info("SparkSession stopped.")


# --------------------------------------------------------------------------
# Pipeline Orchestration Function & CLI Entry Point
# --------------------------------------------------------------------------

def run_sales_analytics(
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
) -> Dict[str, DataFrame]:
    """Coordinates ingestion, PySpark analytical transformations, and reporting.

    Args:
        raw_dir: Directory containing input Parquet datasets (default: data/raw).
        processed_dir: Directory for storing output analytics (default: data/processed).

    Returns:
        Dict[str, DataFrame]: Computed analytics DataFrames.
    """
    input_dir = raw_dir or config.paths.raw_dir
    output_dir = processed_dir or config.paths.processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    orders_path = input_dir / "orders.parquet"
    products_path = input_dir / "products.parquet"

    if not orders_path.exists() or not products_path.exists():
        raise FileNotFoundError(
            f"Required Parquet datasets missing in {input_dir}. "
            "Please run `python main.py` first to generate raw data."
        )

    with SalesAnalytics() as analytics:
        # 1. Load Parquet datasets
        orders_df = analytics.load_parquet(orders_path)
        products_df = analytics.load_parquet(products_path)

        # 2. Execute Analytics Queries
        print("\n" + "=" * 90)
        print(" " * 30 + "PYSPARK SALES ANALYTICS")
        print("=" * 90)

        # Query 1: Sales by Category
        category_df = analytics.sales_by_category(orders_df, products_df)
        print("\n>>> INSIGHT 1: SALES & UNITS SOLD BY PRODUCT CATEGORY")
        category_df.show(truncate=False)

        # Query 2: Top 10 Customers by Revenue
        top_cust_df = analytics.top_customers_by_revenue(orders_df, products_df, n=10)
        print("\n>>> INSIGHT 2: TOP 10 CUSTOMERS BY LIFETIME REVENUE (Pareto Driven)")
        top_cust_df.show(truncate=False)

        # Query 3: Monthly Trends & MoM Growth %
        monthly_df = analytics.monthly_trends(orders_df, products_df)
        print("\n>>> INSIGHT 3: MONTHLY REVENUE TRENDS & MoM GROWTH %")
        monthly_df.show(24, truncate=False)

        # 3. Save Processed Analytics to data/processed/
        analytics.save_dataframe(category_df, output_dir / "sales_by_category", "parquet")
        analytics.save_dataframe(top_cust_df, output_dir / "top_customers", "parquet")
        analytics.save_dataframe(monthly_df, output_dir / "monthly_trends", "parquet")

        print("=" * 90)
        logger.info("All analytics successfully generated and saved to %s", output_dir)

        return {
            "sales_by_category": category_df,
            "top_customers": top_cust_df,
            "monthly_trends": monthly_df,
        }


if __name__ == "__main__":
    try:
        run_sales_analytics()
    except Exception as error:
        logger.error("Analytics execution failed: %s", error, exc_info=True)
        sys.exit(1)
