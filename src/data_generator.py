"""Synthetic Data Generator for E-Commerce Data Pipeline.

This module defines the `SyntheticDataGenerator` class, which uses Faker,
NumPy distributions (including a Pareto 80/20 power law for customer orders,
and Normal distribution for customer age), and tqdm progress tracking to
generate realistic e-commerce datasets as pandas DataFrames.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from faker import Faker
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import PipelineConfig, config, setup_logging

logger = setup_logging(log_level="INFO", logger_name="SyntheticDataGenerator")


class SyntheticDataGenerator:
    """High-performance synthetic e-commerce data generator.

    Generates customers, products, and orders adhering to real-world statistical
    distributions:
      - Customer Age: Normal distribution centered around 35 years.
      - Customer Order Propensity: Pareto distribution (80/20 rule: 20% of
        customers generate 80% of total order volume).
      - Product Attributes: Realistic pricing ($10-$500), ratings (1-5), and
        category distributions.
      - Referential Integrity: Order timestamps are strictly on or after customer
        registration dates.

    Attributes:
        num_customers: Default total number of customer records (100,000).
        num_products: Default total number of product catalog items (10,000).
        num_orders: Default total number of order transactions (1,000,000).
        seed: Random seed for reproducible generation.
        batch_size: Chunk size for progress tracking and batched generation.
    """

    CATEGORIES: List[str] = [
        "Electronics",
        "Clothing",
        "Home",
        "Sports",
        "Books",
    ]

    PRODUCT_NAME_TEMPLATES: Dict[str, List[str]] = {
        "Electronics": [
            "Wireless Noise-Canceling Headphones",
            "Smart 4K Ultra HD TV",
            "Mechanical Gaming Keyboard",
            "Ergonomic Wireless Mouse",
            "Bluetooth Portable Speaker",
            "Fast-Charging Power Bank",
            "Curved Gaming Monitor",
            "Smart Fitness Watch",
            "High-Definition Streaming Webcam",
            "USB-C Multiport Docking Station",
        ],
        "Clothing": [
            "Classic Crewneck Cotton T-Shirt",
            "Slim Fit Stretch Denim Jeans",
            "All-Weather Insulated Jacket",
            "Breathable Athletic Running Shorts",
            "Cozy Fleece Pullover Hoodie",
            "Premium Leather Dress Belt",
            "Merino Wool Winter Beanie",
            "Tailored Business Casual Blazer",
            "Quick-Dry Training Top",
            "Casual Linen Button-Down Shirt",
        ],
        "Home": [
            "Espresso & Cappuccino Machine",
            "Digital Air Fryer Deluxe",
            "Smart Robot Vacuum Cleaner",
            "Stainless Steel 10-Piece Cookware Set",
            "Orthopedic Memory Foam Pillow",
            "Electric Temperature-Control Kettle",
            "Ceramic 16-Piece Dinnerware Set",
            "Ultrasonic Cool Mist Humidifier",
            "Modern Dimmable LED Desk Lamp",
            "High-Speed Countertop Blender",
        ],
        "Sports": [
            "Adjustable Cast Iron Dumbbell Set",
            "Extra Thick Non-Slip Yoga Mat",
            "Vacuum Insulated Stainless Water Bottle",
            "Heavy-Duty Resistance Exercise Bands",
            "Trail Running Hydration Backpack",
            "Pro Speed Jump Rope",
            "Waterproof 4-Person Camping Tent",
            "Carbon Fiber Tennis Racket",
            "High-Density Foam Roller",
            "Multi-Grip Pull-Up Bar",
        ],
        "Books": [
            "Modern Data Engineering Architecture",
            "Designing Distributed Systems with PySpark",
            "Hands-On Machine Learning with Python",
            "The Clean Architecture Blueprint",
            "Chronicles of the Deep Cosmos",
            "World Economic History & Markets",
            "The Art of Strategic Leadership",
            "Cognitive Psychology & Decision Making",
            "Fundamentals of Cloud Infrastructure",
            "The Mindful Living Philosophy",
        ],
    }

    @staticmethod
    def _compute_pareto_alpha(n_customers: int, target_share: float = 0.80) -> float:
        """Finds the exact power-law exponent so top 20% of customers hold target_share mass.

        Args:
            n_customers: Number of customers in the base.
            target_share: Desired proportion of mass for top 20% (default 0.80).

        Returns:
            float: Calibrated Pareto power-law exponent alpha.
        """
        low, high = 0.5, 3.0
        ranks = np.arange(1, n_customers + 1, dtype=np.float64)
        top_k = max(1, int(0.20 * n_customers))
        for _ in range(25):
            mid = (low + high) / 2.0
            w = ranks ** (-mid)
            share = w[:top_k].sum() / w.sum()
            if share < target_share:
                low = mid
            else:
                high = mid
        return mid

    def __init__(
        self,
        num_customers: int = 100_000,
        num_products: int = 10_000,
        num_orders: int = 1_000_000,
        seed: int = 42,
        batch_size: int = 20_000,
    ) -> None:
        """Initializes the SyntheticDataGenerator with volume and seed configurations.

        Args:
            num_customers: Number of customers to generate by default (100K).
            num_products: Number of products to generate by default (10K).
            num_orders: Number of orders to generate by default (1M).
            seed: Seed for random number generators to ensure reproducibility.
            batch_size: Batch size for chunked progress updates.
        """
        self.num_customers = num_customers
        self.num_products = num_products
        self.num_orders = num_orders
        self.seed = seed
        self.batch_size = batch_size

        self.rng = np.random.default_rng(seed)
        self.faker = Faker()
        Faker.seed(seed)

        logger.info(
            "Initialized SyntheticDataGenerator [Customers: %s, Products: %s, Orders: %s, Seed: %d]",
            f"{self.num_customers:,}",
            f"{self.num_products:,}",
            f"{self.num_orders:,}",
            self.seed,
        )

    def generate_customers(
        self,
        count: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generates synthetic customer demographic and account records.

        Generates customers with:
          - customer_id: Formatted alphanumeric unique identifier.
          - name: Realistic customer name generated using Faker.
          - email: Domain-linked email address derived via Faker.
          - age: Integer age drawn from a Normal distribution (mean=35, std=10),
            clipped to realistic working age bounds (18-75).
          - city: City name generated using Faker.
          - country: Country identifier (defaults to USA/international via Faker).
          - registration_date: Account creation date (YYYY-MM-DD) between 2022 and 2025.

        Args:
            count: Number of customer records to generate. Defaults to self.num_customers.

        Returns:
            pd.DataFrame: DataFrame containing generated customer records.

        Example:
            >>> gen = SyntheticDataGenerator(num_customers=50)
            >>> df = gen.generate_customers()
            >>> df.shape[0]
            50
        """
        n = count if count is not None else self.num_customers
        logger.info("Starting generation of %s customers...", f"{n:,}")

        # 1. Vectorized Statistical Distributions via NumPy
        # Normal distribution for age centered around 35 (mean=35, std=10, clipped 18-75)
        raw_ages = self.rng.normal(loc=35.0, scale=10.0, size=n)
        ages = np.clip(np.round(raw_ages).astype(int), 18, 75)

        # Uniform timestamps between 2022-01-01 and 2025-12-31 for registration_date
        start_ts = int(datetime(2022, 1, 1).timestamp())
        end_ts = int(datetime(2025, 12, 31).timestamp())
        reg_timestamps = self.rng.integers(start_ts, end_ts, size=n)
        reg_dates = pd.to_datetime(reg_timestamps, unit="s").strftime("%Y-%m-%d")

        # 2. Formatted unique customer IDs
        customer_ids = [f"CUST_{i:07d}" for i in range(1, n + 1)]

        # 3. PII Generation using Faker with tqdm Progress Tracking
        # Pre-generate diverse Faker name and location pools for high throughput
        logger.debug("Generating PII pools with Faker...")
        pool_size = min(5_000, max(500, n // 10))
        first_names = [self.faker.first_name() for _ in range(pool_size)]
        last_names = [self.faker.last_name() for _ in range(pool_size)]
        cities = [self.faker.city() for _ in range(min(1_000, pool_size))]
        countries = ["United States", "Canada", "United Kingdom", "Germany", "Australia"]

        # Random indices from pools
        fn_indices = self.rng.integers(0, len(first_names), size=n)
        ln_indices = self.rng.integers(0, len(last_names), size=n)
        city_indices = self.rng.integers(0, len(cities), size=n)
        country_indices = self.rng.integers(0, len(countries), size=n)
        domains = ["gmail.com", "yahoo.com", "outlook.com", "icloud.com", "example.com"]
        domain_indices = self.rng.integers(0, len(domains), size=n)

        names: List[str] = []
        emails: List[str] = []
        cust_cities: List[str] = []
        cust_countries: List[str] = []

        batch_size = self.batch_size
        with tqdm(total=n, desc="Generating Customers", unit="cust") as pbar:
            for start_idx in range(0, n, batch_size):
                end_idx = min(start_idx + batch_size, n)
                for i in range(start_idx, end_idx):
                    fn = first_names[fn_indices[i]]
                    ln = last_names[ln_indices[i]]
                    full_name = f"{fn} {ln}"
                    email = f"{fn.lower()}.{ln.lower()}{i+1}@{domains[domain_indices[i]]}"

                    names.append(full_name)
                    emails.append(email)
                    cust_cities.append(cities[city_indices[i]])
                    cust_countries.append(countries[country_indices[i]])

                pbar.update(end_idx - start_idx)

        customers_df = pd.DataFrame(
            {
                "customer_id": customer_ids,
                "name": names,
                "email": emails,
                "age": ages,
                "city": cust_cities,
                "country": cust_countries,
                "registration_date": reg_dates,
            }
        )

        logger.info(
            "Customer generation completed: %s rows created. Mean age: %.1f",
            f"{len(customers_df):,}",
            customers_df["age"].mean(),
        )
        return customers_df

    def generate_products(
        self,
        count: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generates synthetic product catalog records with realistic attributes.

        Generates products with:
          - product_id: Formatted unique catalog identifier (e.g. PROD_00001).
          - name: Descriptive product name mapped to the corresponding category.
          - category: One of ['Electronics', 'Clothing', 'Home', 'Sports', 'Books'].
          - price: Retail price between $10.00 and $500.00 rounded to 2 decimal places.
          - stock: Inventory level (integer 0 to 1,000).
          - rating: Product review rating between 1.0 and 5.0 rounded to 1 decimal place.

        Args:
            count: Number of products to generate. Defaults to self.num_products.

        Returns:
            pd.DataFrame: DataFrame containing generated product catalog records.

        Example:
            >>> gen = SyntheticDataGenerator(num_products=10)
            >>> df = gen.generate_products()
            >>> df.shape[0]
            10
        """
        n = count if count is not None else self.num_products
        logger.info("Starting generation of %s products...", f"{n:,}")

        product_ids = [f"PROD_{i:05d}" for i in range(1, n + 1)]

        # 1. Category Selection
        category_indices = self.rng.integers(0, len(self.CATEGORIES), size=n)
        categories = [self.CATEGORIES[idx] for idx in category_indices]

        # 2. Product Name Construction (Faker descriptor + category template)
        descriptors = [
            "Pro", "Ultra", "Essential", "Eco", "Classic",
            "Deluxe", "Compact", "Ergonomic", "Smart", "Signature"
        ]
        desc_indices = self.rng.integers(0, len(descriptors), size=n)

        names: List[str] = []
        with tqdm(total=n, desc="Generating Products", unit="prod") as pbar:
            for i in range(n):
                cat = categories[i]
                templates = self.PRODUCT_NAME_TEMPLATES[cat]
                template = templates[self.rng.integers(0, len(templates))]
                desc = descriptors[desc_indices[i]]
                names.append(f"{desc} {template}")
                if (i + 1) % self.batch_size == 0 or (i + 1) == n:
                    pbar.update(min(self.batch_size, n - pbar.n))

        # 3. Statistical Distributions for Price, Stock, and Rating
        # Uniform price distribution between $10.00 and $500.00
        prices = np.round(self.rng.uniform(10.0, 500.0, size=n), 2)

        # Stock quantity between 0 (out of stock) and 1,000 units
        stock_quantities = self.rng.integers(0, 1001, size=n)

        # Customer rating between 1.0 and 5.0 (skewed positively around 4.1)
        raw_ratings = self.rng.normal(loc=4.1, scale=0.7, size=n)
        ratings = np.round(np.clip(raw_ratings, 1.0, 5.0), 1)

        products_df = pd.DataFrame(
            {
                "product_id": product_ids,
                "name": names,
                "category": categories,
                "price": prices,
                "stock": stock_quantities,
                "rating": ratings,
            }
        )

        logger.info(
            "Product generation completed: %s rows created. Avg price: $%.2f, Avg rating: %.1f",
            f"{len(products_df):,}",
            products_df["price"].mean(),
            products_df["rating"].mean(),
        )
        return products_df

    def generate_orders(
        self,
        count: Optional[int] = None,
        customers_df: Optional[pd.DataFrame] = None,
        products_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generates synthetic orders using a Pareto distribution for customer activity.

        Implements the 80/20 rule: 20% of customers account for 80% of total orders.
        Also enforces chronological referential integrity: order_date is guaranteed
        to be on or after the associated customer's registration_date.

        Generates orders with:
          - order_id: Formatted unique order identifier (e.g. ORD_0000001).
          - customer_id: Customer ID sampled according to Pareto order propensity.
          - product_id: Product ID sampled from the active product catalog.
          - quantity: Purchase quantity between 1 and 10 items.
          - order_date: Order timestamp (YYYY-MM-DD HH:MM:SS) on/after registration date.

        Args:
            count: Number of order records to generate. Defaults to self.num_orders.
            customers_df: Pre-existing customers DataFrame. Generated automatically if None.
            products_df: Pre-existing products DataFrame. Generated automatically if None.

        Returns:
            pd.DataFrame: DataFrame containing generated order transactions.

        Raises:
            ValueError: If customers_df or products_df is empty.
        """
        n_orders = count if count is not None else self.num_orders

        # Ensure parent datasets exist
        if customers_df is None:
            logger.info("No customers_df provided; generating customers first...")
            customers_df = self.generate_customers()

        if products_df is None:
            logger.info("No products_df provided; generating products first...")
            products_df = self.generate_products()

        num_cust = len(customers_df)
        num_prod = len(products_df)

        if num_cust == 0 or num_prod == 0:
            raise ValueError("customers_df and products_df must contain at least 1 record.")

        logger.info(
            "Starting generation of %s orders across %s customers and %s products...",
            f"{n_orders:,}",
            f"{num_cust:,}",
            f"{num_prod:,}",
        )

        # ------------------------------------------------------------------
        # 1. Pareto 80/20 Customer Sampling:
        # Dynamically calibrate alpha so top 20% of customers hold exactly 80% mass
        # ------------------------------------------------------------------
        alpha = self._compute_pareto_alpha(num_cust, target_share=0.80)
        ranks = np.arange(1, num_cust + 1, dtype=np.float64)
        pareto_weights = ranks ** (-alpha)

        # Permute weights so high-activity customers are randomly scattered
        shuffled_weights = self.rng.permutation(pareto_weights)
        probabilities = shuffled_weights / np.sum(shuffled_weights)

        # Log Pareto mass verification
        top_20_count = int(0.20 * num_cust)
        top_20_mass = np.sum(np.sort(probabilities)[::-1][:top_20_count]) * 100
        logger.info(
            "Pareto distribution configured (alpha=%.4f): Top 20%% of customers hold %.2f%% of order propensity.",
            alpha,
            top_20_mass,
        )

        # ------------------------------------------------------------------
        # 2. Vectorized Sampling with tqdm Progress Tracking
        # ------------------------------------------------------------------
        batch_size = self.batch_size
        sampled_cust_indices = np.empty(n_orders, dtype=np.int64)
        sampled_prod_indices = np.empty(n_orders, dtype=np.int64)
        quantities = np.empty(n_orders, dtype=np.int32)

        with tqdm(total=n_orders, desc="Generating Orders", unit="orders") as pbar:
            for start_idx in range(0, n_orders, batch_size):
                end_idx = min(start_idx + batch_size, n_orders)
                chunk_len = end_idx - start_idx

                sampled_cust_indices[start_idx:end_idx] = self.rng.choice(
                    num_cust,
                    size=chunk_len,
                    replace=True,
                    p=probabilities,
                )
                sampled_prod_indices[start_idx:end_idx] = self.rng.integers(
                    0, num_prod, size=chunk_len
                )
                quantities[start_idx:end_idx] = self.rng.integers(
                    1, 11, size=chunk_len
                )

                pbar.update(chunk_len)

        # ------------------------------------------------------------------
        # 3. Vectorized Order Dates (Referential Integrity Check)
        # ------------------------------------------------------------------
        logger.info("Computing chronologically valid order timestamps...")
        customer_id_series = customers_df["customer_id"].values
        product_id_series = products_df["product_id"].values

        # Extract registration timestamps as integer epoch seconds (pandas cross-version safe)
        reg_timestamps = (
            pd.to_datetime(customers_df["registration_date"])
            .astype("datetime64[s]")
            .astype("int64")
            .values
        )

        sampled_reg_ts = reg_timestamps[sampled_cust_indices]
        end_order_ts = int(datetime(2026, 3, 31, 23, 59, 59).timestamp())

        # Random seconds between customer registration date and end date
        max_offsets = np.maximum(end_order_ts - sampled_reg_ts, 3600)
        random_offsets = (self.rng.random(size=n_orders) * max_offsets).astype(np.int64)
        order_timestamps = sampled_reg_ts + random_offsets
        order_dates = pd.to_datetime(order_timestamps, unit="s").strftime("%Y-%m-%d %H:%M:%S")

        # 4. Assemble Data
        order_ids = [f"ORD_{i:08d}" for i in range(1, n_orders + 1)]
        order_customers = customer_id_series[sampled_cust_indices]
        order_products = product_id_series[sampled_prod_indices]

        orders_df = pd.DataFrame(
            {
                "order_id": order_ids,
                "customer_id": order_customers,
                "product_id": order_products,
                "quantity": quantities,
                "order_date": order_dates,
            }
        )

        # 5. Verify Empirical Pareto 80/20 Distribution
        _, order_counts = np.unique(sampled_cust_indices, return_counts=True)
        sorted_counts = np.sort(order_counts)[::-1]
        active_customers_count = len(sorted_counts)
        top_20_active = max(1, int(0.20 * active_customers_count))
        orders_from_top_20 = sorted_counts[:top_20_active].sum()
        empirical_top_20_pct = (orders_from_top_20 / n_orders) * 100

        logger.info(
            "Order generation completed: %s rows created. "
            "Pareto verification: Top 20%% of active buyers (%s customers) generated %.2f%% of all orders (%s orders).",
            f"{len(orders_df):,}",
            f"{top_20_active:,}",
            empirical_top_20_pct,
            f"{orders_from_top_20:,}",
        )

        return orders_df

    def generate_all(
        self,
        save_to_disk: bool = True,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Runs the complete data generation workflow and exports CSVs.

        Args:
            save_to_disk: Whether to persist generated DataFrames to CSV files.
            output_dir: Target output directory. Defaults to config.paths.raw_dir.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing 'customers', 'products',
            and 'orders' DataFrames.
        """
        logger.info("=" * 80)
        logger.info("STARTING SYNTHETIC E-COMMERCE DATA GENERATION PIPELINE")
        logger.info("=" * 80)

        # 1. Generate Customers
        customers_df = self.generate_customers()

        # 2. Generate Products
        products_df = self.generate_products()

        # 3. Generate Orders
        orders_df = self.generate_orders(
            customers_df=customers_df,
            products_df=products_df,
        )

        datasets = {
            "customers": customers_df,
            "products": products_df,
            "orders": orders_df,
        }

        # Save to disk if requested
        if save_to_disk:
            target_dir = output_dir or config.paths.raw_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Persisting datasets to %s...", target_dir)

            for name, df in datasets.items():
                csv_path = target_dir / f"{name}.csv"
                logger.info("Saving %s (%s rows) to %s...", name, f"{len(df):,}", csv_path.name)
                df.to_csv(csv_path, index=False)

            logger.info("All datasets successfully saved to %s", target_dir)

        logger.info("=" * 80)
        logger.info("SYNTHETIC DATA GENERATION PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        return datasets


# --------------------------------------------------------------------------
# Standalone Convenience Functions for Backward Compatibility & Direct Invocation
# --------------------------------------------------------------------------

def generate_customers(
    num_customers: int = 100_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Convenience function to generate customer records using SyntheticDataGenerator.

    Args:
        num_customers: Total number of customer records.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame: Generated customers DataFrame.
    """
    generator = SyntheticDataGenerator(num_customers=num_customers, seed=seed)
    return generator.generate_customers(count=num_customers)


def generate_products(
    num_products: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Convenience function to generate product catalog records.

    Args:
        num_products: Total number of product records.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame: Generated products DataFrame.
    """
    generator = SyntheticDataGenerator(num_products=num_products, seed=seed)
    return generator.generate_products(count=num_products)


def generate_orders(
    num_orders: int = 1_000_000,
    customers_df: Optional[pd.DataFrame] = None,
    products_df: Optional[pd.DataFrame] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Convenience function to generate order records with Pareto distribution.

    Args:
        num_orders: Total number of order records.
        customers_df: Optional customers DataFrame.
        products_df: Optional products DataFrame.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame: Generated orders DataFrame.
    """
    generator = SyntheticDataGenerator(num_orders=num_orders, seed=seed)
    return generator.generate_orders(
        count=num_orders,
        customers_df=customers_df,
        products_df=products_df,
    )


def run_data_generation(
    cfg: Optional[PipelineConfig] = None,
) -> Dict[str, pd.DataFrame]:
    """Top-level pipeline orchestration function.

    Args:
        cfg: Pipeline configuration instance.

    Returns:
        Dict[str, pd.DataFrame]: Generated datasets dictionary.
    """
    active_cfg = cfg or config
    gen_cfg = active_cfg.generator

    generator = SyntheticDataGenerator(
        num_customers=gen_cfg.num_customers,
        num_products=gen_cfg.num_products,
        num_orders=gen_cfg.num_orders,
        seed=gen_cfg.random_seed,
    )

    return generator.generate_all(
        save_to_disk=True,
        output_dir=active_cfg.paths.raw_dir,
    )


if __name__ == "__main__":
    # Generate datasets using default configuration (100K customers, 10K products, 1M orders)
    run_data_generation()
