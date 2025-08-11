import os
import zipfile
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import pytest
import requests

from tracksdata.graph import RustWorkXGraph, SQLGraph
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.graph._rustworkx_graph import IndexedRXGraph


@pytest.fixture(params=[RustWorkXGraph, SQLGraph, IndexedRXGraph])
def graph_backend(request) -> BaseGraph:
    """Fixture that provides all implementations of BaseGraph."""
    graph_class: BaseGraph = request.param

    if graph_class == SQLGraph:
        return graph_class(
            drivername="sqlite",
            database=":memory:",
            engine_kwargs={"connect_args": {"check_same_thread": False}},
        )
    elif graph_class == IndexedRXGraph:
        #  we hack the IndexedRXGraph to have a non-trivial index map
        rng = np.random.default_rng(42)
        max_index = 2**32
        orig_add_node = graph_class.add_node
        orig_bulk_add_nodes = graph_class.bulk_add_nodes
        obj = graph_class()

        def _add_node_with_index(
            self,
            attrs: dict[str, Any],
            validate_keys: bool = True,
            index: int | None = None,
        ) -> int:
            if index is not None:
                # Use the provided index
                new_index = index
            else:
                # Generate a random index
                while True:
                    new_index = rng.integers(0, max_index).item()
                    if new_index not in self._external_to_local:
                        break
            return orig_add_node(self, attrs, validate_keys, new_index)

        def _bulk_add_nodes_with_index(
            self,
            nodes: list[dict[str, Any]],
            indices: list[int] | None = None,
        ) -> list[int]:
            if indices is None:
                # Generate random indices
                current_max = max(self.node_ids()) + 1 if self.num_nodes > 0 else 0
                indices = (rng.integers(current_max, max_index) + np.arange(len(nodes))).tolist()
            return orig_bulk_add_nodes(self, nodes, indices)

        obj.add_node = MethodType(_add_node_with_index, obj)
        obj.bulk_add_nodes = MethodType(_bulk_add_nodes_with_index, obj)
        return obj
    else:
        return graph_class()


def _download_to(url: str, output_path: Path) -> None:
    """Download CTC data."""
    with open(output_path, "wb") as f:
        response = requests.get(url)
        response.raise_for_status()
        f.write(response.content)


@pytest.fixture(scope="session")
def ctc_data_dir(pytestconfig: pytest.Config) -> Path:
    """Fixture to download CTC data."""

    temp_dir = Path(pytestconfig.cache._cachedir) / "downloads"
    temp_dir.mkdir(parents=True, exist_ok=True)

    zip_dir = temp_dir / "DIC-C2DH-HeLa.zip"
    out_dir = temp_dir / "DIC-C2DH-HeLa"
    url = "https://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip"

    if not zip_dir.exists():
        _download_to(url, zip_dir)

    if not out_dir.exists():
        try:
            with zipfile.ZipFile(zip_dir) as zip_file:
                zip_file.extractall(temp_dir)

        except zipfile.BadZipFile:
            os.remove(zip_dir)
            _download_to(url, zip_dir)
            # retry
            with zipfile.ZipFile(zip_dir) as zip_file:
                zip_file.extractall(temp_dir)

    return out_dir
