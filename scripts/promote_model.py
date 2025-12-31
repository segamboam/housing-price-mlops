#!/usr/bin/env python3
"""Script to promote a model version to champion in MLflow Model Registry.

Usage:
    python scripts/promote_model.py --version 3
    python scripts/promote_model.py --version 3 --alias champion
    python scripts/promote_model.py --list  # List all versions
"""

import argparse
import os
import sys

import mlflow
from mlflow import MlflowClient


def get_tracking_uri() -> str:
    """Get MLflow tracking URI from environment or default."""
    return os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def list_model_versions(client: MlflowClient, model_name: str) -> None:
    """List all versions of a registered model."""
    print(f"\nRegistered Model: {model_name}")
    print("-" * 70)

    try:
        model = client.get_registered_model(model_name)
        print(f"Description: {model.description or '(none)'}")

        # Get aliases
        aliases = model.aliases or {}
        print(f"Aliases: {aliases if aliases else '(none)'}")
        print()

        # List all versions
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            print("No versions found.")
            return

        print(f"{'Version':<10} {'Stage':<15} {'Run ID':<35} {'Created':<20}")
        print("-" * 80)

        for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
            # Check if this version has any alias
            version_aliases = [alias for alias, ver in aliases.items() if ver == v.version]
            alias_str = f" ({', '.join(version_aliases)})" if version_aliases else ""

            print(
                f"{v.version:<10} {v.current_stage:<15} {v.run_id:<35} {v.creation_timestamp}"
                f"{alias_str}"
            )

    except mlflow.exceptions.MlflowException as e:
        print(f"Error: Model '{model_name}' not found. {e}")
        sys.exit(1)


def promote_model(
    client: MlflowClient, model_name: str, version: str, alias: str
) -> None:
    """Promote a model version to the specified alias."""
    try:
        # Verify version exists
        client.get_model_version(model_name, version)

        # Set alias
        client.set_registered_model_alias(model_name, alias, version)
        print(f"\n✓ Model '{model_name}' version {version} promoted to '{alias}'")

        # Show current state
        model = client.get_registered_model(model_name)
        print(f"\nCurrent aliases: {model.aliases}")

    except mlflow.exceptions.MlflowException as e:
        print(f"Error: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote a model version in MLflow Model Registry"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="housing-price-model",
        help="Name of the registered model",
    )
    parser.add_argument(
        "--version",
        type=str,
        help="Model version to promote",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default="champion",
        help="Alias to assign (default: champion)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all model versions",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (overrides MLFLOW_TRACKING_URI)",
    )

    args = parser.parse_args()

    # Set tracking URI
    tracking_uri = args.tracking_uri or get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")

    client = MlflowClient()

    if args.list:
        list_model_versions(client, args.model_name)
    elif args.version:
        promote_model(client, args.model_name, args.version, args.alias)
    else:
        parser.print_help()
        print("\nError: Either --list or --version must be specified")
        sys.exit(1)


if __name__ == "__main__":
    main()
