"""Utility class."""

import os

from databricks.sdk import WorkspaceClient


def is_databricks() -> str:
    """Check if the code is running in a Databricks environment."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def get_dbr_host() -> str:
    """Retrieve the Databricks workspace URL.

    This function obtains the workspace URL from Spark configuration.

    :return: The Databricks workspace URL as a string.
    :raises ValueError: If not running in a Databricks environment.
    """
    ws = WorkspaceClient()
    return ws.config.host
