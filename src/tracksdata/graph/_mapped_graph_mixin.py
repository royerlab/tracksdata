"""
Mixin class for graphs that need ID mapping between local and external coordinates.

This module provides common functionality for graphs that maintain a mapping between
internal/local node IDs and external/world node IDs, such as IndexedRXGraph and GraphView.
"""

from collections.abc import Sequence
from typing import overload

import bidict
import numpy as np
import polars as pl


class MappedGraphMixin:
    """
    Mixin for graphs that need ID mapping between local and external coordinates.

    This mixin provides common functionality for graphs that maintain bidirectional
    mapping between internal node IDs (used by the underlying graph structure) and
    external node IDs (exposed to users or parent graphs).

    Attributes
    ----------
    _local_to_external : bidict.bidict[int, int]
        Bidirectional mapping from local IDs to external IDs
    _external_to_local : bidict.bidict[int, int]
        Inverse mapping from external IDs to local IDs
    """

    def __init__(self, id_map: dict[int, int] | bidict.bidict[int, int] | None = None):
        """
        Initialize the ID mapping.

        Parameters
        ----------
        id_map : dict[int, int] | None
            Mapping from local IDs to external IDs. If None, creates empty mapping.
        """
        if id_map is None:
            self._local_to_external = bidict.bidict()
        elif isinstance(id_map, bidict.bidict):
            self._local_to_external = id_map
        elif isinstance(id_map, dict):
            self._local_to_external = bidict.bidict(id_map)
        else:
            raise ValueError(f"Invalid type for id_map: {type(id_map)}")
        self._external_to_local = self._local_to_external.inverse

    @overload
    def _map_to_external(self, local_ids: None) -> None: ...

    @overload
    def _map_to_external(self, local_ids: int) -> int: ...

    @overload
    def _map_to_external(self, local_ids: Sequence[int]) -> list[int]: ...

    def _map_to_external(self, local_ids: int | Sequence[int] | None) -> int | list[int] | None:
        """
        Transform local IDs to external coordinates.

        Parameters
        ----------
        local_ids : int | Sequence[int] | None
            Local IDs to transform

        Returns
        -------
        int | list[int] | None
            External IDs corresponding to the local IDs
        """
        if local_ids is None:
            return None
        if isinstance(local_ids, int):
            return self._local_to_external[local_ids]
        return [self._local_to_external[lid] for lid in local_ids]

    @overload
    def _map_to_local(self, external_ids: None) -> None: ...

    @overload
    def _map_to_local(self, external_ids: int) -> int: ...

    @overload
    def _map_to_local(self, external_ids: Sequence[int]) -> list[int]: ...

    def _map_to_local(self, external_ids: int | Sequence[int] | None) -> int | list[int] | None:
        """
        Transform external IDs to local coordinates.

        Parameters
        ----------
        external_ids : int | Sequence[int] | None
            External IDs to transform

        Returns
        -------
        int | list[int] | None
            Local IDs corresponding to the external IDs
        """
        if external_ids is None:
            return None
        if isinstance(external_ids, int):
            return self._external_to_local[external_ids]
        return [self._external_to_local[eid] for eid in external_ids]

    def _map_df_to_external(self, df: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
        """
        Transform node IDs in DataFrame columns from local to external coordinates.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame containing node IDs to transform
        columns : Sequence[str]
            Column names containing node IDs to transform

        Returns
        -------
        pl.DataFrame
            DataFrame with transformed node IDs
        """
        for col in columns:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).map_elements(self._local_to_external.__getitem__, return_dtype=pl.Int64).alias(col)
                )
        return df

    def _vectorized_map_to_external(self, local_ids: np.ndarray | Sequence[int]) -> np.ndarray:
        """
        Vectorized transformation of local IDs to external coordinates.

        Parameters
        ----------
        local_ids : np.ndarray | Sequence[int]
            Array or sequence of local IDs

        Returns
        -------
        np.ndarray
            Array of external IDs
        """
        vec_map = np.vectorize(self._local_to_external.__getitem__, otypes=[int])
        return vec_map(np.asarray(local_ids, dtype=int))

    def _vectorized_map_to_local(self, external_ids: np.ndarray | Sequence[int]) -> np.ndarray:
        """
        Vectorized transformation of external IDs to local coordinates.

        Parameters
        ----------
        external_ids : np.ndarray | Sequence[int]
            Array or sequence of external IDs

        Returns
        -------
        np.ndarray
            Array of local IDs
        """
        vec_map = np.vectorize(self._external_to_local.__getitem__, otypes=[int])
        return vec_map(np.asarray(external_ids, dtype=int))

    def _add_id_mapping(self, local_id: int, external_id: int) -> None:
        """
        Add a new ID mapping.

        Parameters
        ----------
        local_id : int
            Local node ID
        external_id : int
            External node ID
        """
        try:
            self._local_to_external.put(local_id, external_id)
        except bidict.ValueDuplicationError as e:
            # Convert ValueDuplicationError to KeyDuplicationError since from user perspective
            # the external_id (their "key"/index) is what's being duplicated
            raise bidict.KeyDuplicationError(e.args[0]) from e

    def _add_id_mappings(self, mappings: Sequence[tuple[int, int]]) -> None:
        """
        Add multiple ID mappings at once.

        Parameters
        ----------
        mappings : Sequence[tuple[int, int]]
            Sequence of (local_id, external_id) pairs
        """
        self._local_to_external.putall(mappings)

    def _remove_id_mapping(
        self,
        *,
        local_id: int | None = None,
        external_id: int | None = None,
    ) -> None:
        """
        Remove an ID mapping.

        Parameters
        ----------
        local_id : int
            Local node ID to remove from mapping
        external_id : int
            External node ID to remove from mapping
        """
        if local_id is not None:
            del self._local_to_external[local_id]
        elif external_id is not None:
            del self._external_to_local[external_id]
        else:
            raise ValueError("Either local_id or external_id must be provided")

    def _get_external_ids(self) -> list[int]:
        """
        Get all external IDs in the mapping.

        Returns
        -------
        list[int]
            List of all external node IDs
        """
        return list(self._local_to_external.values())

    def _get_local_ids(self) -> list[int]:
        """
        Get all local IDs in the mapping.

        Returns
        -------
        list[int]
            List of all local node IDs
        """
        return list(self._local_to_external.keys())
