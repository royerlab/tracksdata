# TracksData

[![PyPI - License](https://img.shields.io/pypi/l/tracksdata.svg?color=green)](https://github.com/royerlab/tracksdata/raw/main/LICENSE)
[![PyPI - Version](https://img.shields.io/pypi/v/tracksdata.svg)](https://pypi.org/project/tracksdata)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tracksdata.svg)](https://pypi.org/project/tracksdata)
[![CI](https://github.com/royerlab/tracksdata/actions/workflows/ci.yml/badge.svg)](https://github.com/royerlab/tracksdata/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/royerlab/tracksdata/branch/main/graph/badge.svg)](https://codecov.io/gh/royerlab/tracksdata)

A common data structure and basic tools for multi-object tracking.

## Features

- Graph-based representation of tracking problems
- In-memory (RustWorkX) and database-backed (SQL) graph backends
- Nodes and edges can take arbitrary attributes
- Standardize API for node operators (e.g. defining objects and their attributes)
- Standardize API for edge operators (e.g. creating edges between nodes)
- Basic tracking solvers: nearest neighbors and integer linear programming
- Compatible with Cell Tracking Challenge (CTC) format
- Efficient subgraphing based on attributes on any graph backend
- Integration with cell tracking evaluation metrics

## Installation

```console
pip install tracksdata
```

## Why tracksdata?

TracksData provides a common data structure for multi-object tracking problems.
It uses graphs to represent detections (nodes) and their connections (edges), making it easier to work with tracking data across different algorithms.

Key benefits:
- Consistent data representation for tracking problems
- Modular components that can be combined as needed
- Support for both small datasets (in-memory) and large datasets (database)

## Documentation

- [Full Documentation](https://royerlab.github.io/tracksdata/)
- [Installation](https://royerlab.github.io/tracksdata/installation/)
- [Core Concepts](https://royerlab.github.io/tracksdata/concepts/)
- [Getting Started](https://royerlab.github.io/tracksdata/getting_started/)
- [API Reference](https://royerlab.github.io/tracksdata/reference/tracksdata/)
- [FAQ](https://royerlab.github.io/tracksdata/faq/)
