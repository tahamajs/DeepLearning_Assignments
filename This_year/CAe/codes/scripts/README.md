# `scripts/` — Helper Scripts for Project Management

This directory contains various helper scripts designed to automate common tasks within the CAe project. These scripts streamline development, manage datasets, and assist with other project-related operations.

## Purpose

The `scripts/` module aims to provide:

-   **Automated Setup**: Scripts for setting up the development environment or specific project components.
-   **Data Management**: Utilities for downloading, preprocessing, or organizing datasets.
-   **Execution Automation**: Scripts to run experiments, tests, or specific workflows.
-   **Utility Functions**: Any other command-line tools that simplify project tasks.

## Structure

Common types of scripts found in this directory include:

-   `download_datasets.py`: A script responsible for downloading external datasets required by the various assignments. It typically includes logic to fetch data from remote URLs, handle different dataset formats, and organize them into a designated `data/` directory.
-   `run_tests.sh`: (Example) A shell script to execute unit tests or smoke tests for the entire project or specific modules.
-   `setup_env.sh`: (Example) A script for setting up the Python environment, installing dependencies, or configuring environment variables.

## Key Guidelines

-   **Command-Line Interface**: Scripts should generally be executable from the command line and support arguments for flexibility (e.g., using `argparse` in Python).
-   **Idempotence**: Scripts should ideally be idempotent, meaning running them multiple times should produce the same result as running them once.
-   **Error Handling**: Scripts should include basic error handling to provide informative messages in case of failures.
-   **Documentation**: Each script should have a clear purpose and, if complex, include comments or internal documentation explaining its functionality and usage.

## Example Usage (`download_datasets.py`)

The `download_datasets.py` script is a crucial utility for ensuring all necessary data is available for the assignments. It can be invoked with specific flags to download different datasets.

```bash
# Example: Download UrbanSound and Flickr datasets
python scripts/download_datasets.py --urbansound --flickr

# Example: Download only the UrbanSound dataset
python scripts/download_datasets.py --urbansound
```

## Example Usage (`run_all.sh`)

The `run_all.sh` script, located in the parent `codes/` directory, orchestrates smoke tests across various assignments. While not directly in `scripts/`, it demonstrates how scripts can be used to manage project workflows.

```bash
# Execute smoke tests from the codes/ directory
bash run_all.sh
```
