"""Common test fixtures for the marvel-characters project."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pyspark.sql import SparkSession

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import project modules after path setup
from marvel_characters.config import ProjectConfig  # noqa: E402


@pytest.fixture(scope="session")
def spark_session() -> SparkSession | MagicMock:
    """Create a SparkSession for testing."""
    try:
        return (
            SparkSession.builder.master("local[1]")
            .appName("marvel-characters-test")
            .config("spark.sql.shuffle.partitions", "1")
            .config("spark.default.parallelism", "1")
            .getOrCreate()
        )
    except Exception:
        # Return a mock if we can't create a real SparkSession
        return MagicMock()


@pytest.fixture
def mock_project_config() -> MagicMock:
    """Create a mock ProjectConfig for testing."""
    config = MagicMock(spec=ProjectConfig)
    config.cat_features = ["Universe", "Origin", "Identity", "Gender", "Marital_Status"]
    config.num_features = ["Height", "Weight", "Teams", "Magic", "Mutant"]
    config.target = "Alive"
    config.catalog_name = "test_catalog"
    config.schema_name = "test_schema"
    return config
