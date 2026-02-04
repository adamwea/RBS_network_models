"""RBS_network_models package.

Keep top-level imports lightweight so utility modules and CLIs can be used
without requiring the full optional dependency stack (e.g. MEA_Analysis,
NetPyNE/NEURON, model packages).

Set `RBS_NETWORK_MODELS_IMPORT_ALL=1` to restore legacy behavior of importing
convenience symbols at package import time.
"""

from __future__ import annotations

import os


# Optional convenience imports (legacy)
if os.environ.get("RBS_NETWORK_MODELS_IMPORT_ALL", "0") == "1":
	try:
		from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis import network_analysis  # type: ignore
	except Exception:
		network_analysis = None  # type: ignore

	try:
		from . import extract_features  # noqa: F401
	except Exception:
		extract_features = None  # type: ignore

	try:
		from .utils import *  # type: ignore  # noqa: F403
	except Exception:
		pass

	try:
		from .models.CDKL5_E6D_T2_C1_05212024 import *  # type: ignore  # noqa: F403
	except Exception:
		pass
