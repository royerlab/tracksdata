# Installation

## Requirements

TracksData requires Python 3.10 or later.

## Install TracksData

### From pypi

```bash
pip install tracksdata
```

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
import tracksdata as td
print(td.__version__)
```
