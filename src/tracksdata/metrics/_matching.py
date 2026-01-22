"""
Matching strategies for graph node comparison.

This module provides different matching strategies for comparing nodes between graphs,
used for metrics computation and graph comparison operations.
"""

import abc

import numpy as np
import polars as pl
from scipy.spatial.distance import cdist

from tracksdata.constants import DEFAULT_ATTR_KEYS


class Matching(abc.ABC):
    """
    Base class for matching strategies between graph nodes.

    Attributes
    ----------
    optimal : bool
        Whether to perform optimal matching using bipartite matching algorithms.
        When True, ensures one-to-one correspondence between matched nodes.
        When False, allows multiple matches per node.
    """

    def __init__(self, optimal: bool):
        """
        Initialize the matching strategy.

        Parameters
        ----------
        optimal : bool
            Whether to perform optimal matching.
        """
        self.optimal = optimal

    @abc.abstractmethod
    def compute_weights(
        self,
        ref_group: pl.DataFrame,
        comp_group: pl.DataFrame,
        reference_graph_key: str,
        input_graph_key: str,
    ) -> tuple[list[int], list[int], list[int], list[int], list[float]]:
        """
        Compute matching weights between reference and comparison groups.

        Parameters
        ----------
        ref_group : pl.DataFrame
            Reference group nodes for a single time point.
        comp_group : pl.DataFrame
            Comparison group nodes for the same time point.
        reference_graph_key : str
            Key to identify nodes in the reference graph.
        input_graph_key : str
            Key to identify nodes in the input graph.

        Returns
        -------
        tuple[list[int], list[int], list[int], list[int], list[float]]
            - mapped_ref: List of reference node IDs
            - mapped_comp: List of comparison node IDs
            - rows: Row indices for sparse matrix
            - cols: Column indices for sparse matrix
            - scores: Matching scores (higher is better)
        """

    @abc.abstractmethod
    def get_required_attrs(self, attr_keys: list[str]) -> list[str]:
        """
        Get the list of required node attributes for this matching strategy.

        Parameters
        ----------
        attr_keys : list[str]
            List of attribute keys to consider.

        Returns
        -------
        list[str]
            List of attribute keys required by this matching strategy.
        """


class MaskMatching(Matching):
    """
    Mask-based matching using Intersection over Union (IoU).

    This strategy matches nodes based on their spatial masks, computing
    IoU and intersection over reference mask area.

    Attributes
    ----------
    optimal : bool
        Whether to perform optimal matching.
    min_reference_intersection : float
        Minimum intersection over reference mask area to consider a match.
        Range: [0, 1]. Default is 0.5.
    """

    def __init__(self, min_reference_intersection: float = 0.5, optimal: bool = True):
        """
        Initialize mask-based matching.

        Parameters
        ----------
        optimal : bool
            Whether to perform optimal matching.
        min_reference_intersection : float, optional
            Minimum intersection over reference mask area. Default is 0.5.
        """
        super().__init__(optimal=optimal)
        if not 0 <= min_reference_intersection <= 1:
            raise ValueError("min_reference_intersection must be between 0 and 1")
        self.min_reference_intersection = min_reference_intersection

    def compute_weights(
        self,
        ref_group: pl.DataFrame,
        comp_group: pl.DataFrame,
        reference_graph_key: str,
        input_graph_key: str,
    ) -> tuple[list[int], list[int], list[int], list[int], list[float]]:
        """
        Compute IoU-based matching weights between masks.

        Parameters
        ----------
        ref_group : pl.DataFrame
            Reference group nodes with mask attributes.
        comp_group : pl.DataFrame
            Comparison group nodes with mask attributes.
        reference_graph_key : str
            Key to identify nodes in the reference graph.
        input_graph_key : str
            Key to identify nodes in the input graph.

        Returns
        -------
        tuple[list[int], list[int], list[int], list[int], list[float]]
            Matching data: mapped_ref, mapped_comp, rows, cols, weights (IoU values).
        """
        from tracksdata.utils._dtypes import column_from_bytes

        # Handle serialized masks if needed
        if ref_group[DEFAULT_ATTR_KEYS.MASK].dtype == pl.Binary:
            ref_group = column_from_bytes(ref_group, DEFAULT_ATTR_KEYS.MASK)
            comp_group = column_from_bytes(comp_group, DEFAULT_ATTR_KEYS.MASK)

        mapped_ref = []
        mapped_comp = []
        rows = []
        cols = []
        weights = []

        for i, (ref_id, ref_mask) in enumerate(
            zip(ref_group[reference_graph_key], ref_group[DEFAULT_ATTR_KEYS.MASK], strict=True)
        ):
            for j, (comp_id, comp_mask) in enumerate(
                zip(comp_group[input_graph_key], comp_group[DEFAULT_ATTR_KEYS.MASK], strict=True)
            ):
                # Intersection over reference is used to select the matches
                inter = ref_mask.intersection(comp_mask)
                ctc_score = inter / ref_mask.size
                if ctc_score > self.min_reference_intersection:
                    mapped_ref.append(ref_id)
                    mapped_comp.append(comp_id)
                    rows.append(i)
                    cols.append(j)

                    # IoU as the matching weight
                    iou = inter / (ref_mask.size + comp_mask.size - inter)
                    weights.append(iou.item())

        return mapped_ref, mapped_comp, rows, cols, weights

    def get_required_attrs(self, attr_keys: list[str]) -> list[str]:
        """
        Get required attributes for mask matching.

        Returns
        -------
        list[str]
            List containing the mask attribute key.
        """
        if DEFAULT_ATTR_KEYS.MASK not in attr_keys:
            raise ValueError(f"Mask attribute key '{DEFAULT_ATTR_KEYS.MASK}' is required for mask matching")
        return [DEFAULT_ATTR_KEYS.MASK]


class DistanceMatching(Matching):
    """
    Distance-based matching using centroid coordinates.

    This strategy matches nodes based on the Euclidean distance between
    their centroids. Supports anisotropic data through scaling factors.
    """

    def __init__(
        self,
        max_distance: float,
        optimal: bool = True,
        attr_keys: tuple[str, ...] | None = None,
        scale: tuple[float, ...] | None = None,
    ):
        """
        Initialize distance-based matching.

        Parameters
        ----------
        optimal : bool
            Whether to perform optimal matching.
        max_distance : float
            Maximum distance for a match.
        attr_keys : tuple[str, ...], optional
            Coordinate keys for centroids. Default is (DEFAULT_ATTR_KEYS.Z, DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X)
            if DEFAULT_ATTR_KEYS.Z exists, otherwise (DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X).
        scale : tuple[float, ...] | None, optional
            Physical scale per dimension. Default is None (isotropic).
        """
        super().__init__(optimal=optimal)
        self.max_distance = max_distance
        self.attr_keys = attr_keys
        self.scale = scale

    def compute_weights(
        self,
        ref_group: pl.DataFrame,
        comp_group: pl.DataFrame,
        reference_graph_key: str,
        input_graph_key: str,
    ) -> tuple[list[int], list[int], list[int], list[int], list[float]]:
        """
        Compute distance-based matching weights between centroids.

        Parameters
        ----------
        ref_group : pl.DataFrame
            Reference group nodes with centroid attributes.
        comp_group : pl.DataFrame
            Comparison group nodes with centroid attributes.
        reference_graph_key : str
            Key to identify nodes in the reference graph.
        input_graph_key : str
            Key to identify nodes in the input graph.

        Returns
        -------
        tuple[list[int], list[int], list[int], list[int], list[float]]
            Matching data: mapped_ref, mapped_comp, rows, cols, weights (1/(1+distance)).
        """

        if self.attr_keys is None:
            if DEFAULT_ATTR_KEYS.Z in ref_group.columns:
                attr_keys = [DEFAULT_ATTR_KEYS.Z, DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X]
            else:
                attr_keys = [DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X]
        else:
            attr_keys = list(self.attr_keys)

        # Extract centroids as numpy arrays (N x D and M x D)
        ref_centroids = ref_group.select(attr_keys).to_numpy()
        comp_centroids = comp_group.select(attr_keys).to_numpy()

        # Apply scale for anisotropic data
        if self.scale is not None:
            if len(self.scale) != len(attr_keys):
                raise ValueError(f"Scale length ({len(self.scale)}) must match attr_keys length ({len(attr_keys)})")

            scale_arr = np.array(self.scale)
            ref_centroids = ref_centroids * scale_arr
            comp_centroids = comp_centroids * scale_arr

        # Compute distance matrix: shape (M, N) where M=comp, N=ref
        distance_matrix = cdist(comp_centroids, ref_centroids)

        # Mask out distances greater than threshold
        comp_indices, ref_indices = np.where(distance_matrix <= self.max_distance)
        distances = distance_matrix[comp_indices, ref_indices]

        # Get node IDs
        ref_ids = ref_group[reference_graph_key].to_numpy()
        comp_ids = comp_group[input_graph_key].to_numpy()

        # Map indices to node IDs
        mapped_ref = ref_ids[ref_indices].tolist()
        mapped_comp = comp_ids[comp_indices].tolist()

        # Return sparse matrix coordinates and weights
        rows = ref_indices.tolist()
        cols = comp_indices.tolist()
        weights = (1.0 / (1.0 + distances)).tolist()

        return mapped_ref, mapped_comp, rows, cols, weights

    def get_required_attrs(self, attr_keys: list[str]) -> list[str]:
        """
        Get required attributes for distance matching.

        Returns
        -------
        list[str]
            List of centroid coordinate keys.
        """
        if self.attr_keys is None:
            if DEFAULT_ATTR_KEYS.Z in attr_keys:
                return [DEFAULT_ATTR_KEYS.Z, DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X]
            else:
                return [DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X]
        else:
            return list(self.attr_keys)
