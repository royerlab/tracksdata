# Installation

## Requirements

TracksData requires Python 3.10 or later.

## Installing Rust (Required)

Until rustworkx 0.17.0 is released, you need to have Rust installed to compile the latest rustworkx:

```bash
conda install -c conda-forge rust
```

## Install TracksData

### From Source (Development)

To install the latest development version:

```bash
git clone https://github.com/royerlab/tracksdata.git
cd tracksdata
pip install .
```

### With Optional Dependencies

For testing:
```bash
pip install .[test]
```

For documentation:
```bash
pip install .[docs]
```

## Verify Installation

You can verify the installation by importing the library:

```python
import tracksdata
print(tracksdata.__version__)
```
