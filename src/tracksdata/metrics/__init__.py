"""
Evaluation metrics for tracking performance, including the [CTC](https://celltrackingchallenge.net/) benchmark metrics.

If you use this module, please cite the respective papers of each metric, as described in
[here](https://github.com/CellTrackingChallenge/py-ctcmetrics?tab=readme-ov-file#acknowledgement-and-citations).
"""

from tracksdata.metrics._ctc_metrics import evaluate_ctc_metrics

__all__ = ["evaluate_ctc_metrics"]
