# TracksData

[![PyPI - License](https://img.shields.io/pypi/l/tracksdata.svg?color=green)](https://github.com/royerlab/tracksdata/raw/main/LICENSE)
[![PyPI - Version](https://img.shields.io/pypi/v/tracksdata.svg)](https://pypi.org/project/tracksdata)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tracksdata.svg)](https://pypi.org/project/tracksdata)
[![CI](https://github.com/royerlab/tracksdata/actions/workflows/ci.yaml/badge.svg)](https://github.com/royerlab/tracksdata/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/royerlab/tracksdata/branch/main/graph/badge.svg)](https://codecov.io/gh/royerlab/tracksdata)

A common data structure and basic tools for multi-object tracking.

## Features

- Graph-based representation of tracking problems
- In-memory (RustWorkX) and database-backed (SQL) graph backends
- Node operators for feature extraction from segmentation masks
- Edge operators for connecting detections (distance, IoU, custom metrics)
- Tracking solvers: nearest neighbors and integer linear programming
- Cell Tracking Challenge (CTC) format support
- Attribute filtering and querying system

## Installation

Until rustworkx 0.17.0 is released, you need to have rust installed to compile the latest rustworkx.

```console
conda install -c conda-forge rust
```

Then install tracksdata with the following command:

```console
pip install .
```

## Why tracksdata?

TracksData provides a common data structure for multi-object tracking problems. It uses graphs to represent detections (nodes) and their connections (edges), making it easier to work with tracking data across different algorithms and formats.

Key benefits:
- Consistent data representation for tracking problems
- Modular components that can be combined as needed
- Support for both small datasets (in-memory) and large datasets (database)
- Integration with Cell Tracking Challenge benchmarks

## Documentation

- [Full Documentation](https://royerlab.github.io/tracksdata/)
- [Getting Started](https://royerlab.github.io/tracksdata/getting_started/)
- [Core Concepts](https://royerlab.github.io/tracksdata/concepts/)
- [API Reference](https://royerlab.github.io/tracksdata/reference/)
- [FAQ](https://royerlab.github.io/tracksdata/faq/)
