"""Run inside the optional Linux image to validate its bundled PyG models."""

import importlib.util
import math
from pathlib import Path

import matgl
import torch
from pymatgen.core import Lattice, Structure

from matsci_agent.api.main import _matgl_missing
from matsci_agent.tools.property_predictor import _calculate_matgl


assert not _matgl_missing()
assert importlib.util.find_spec("dgl") is None
assert torch.version.cuda is None

for path in (
    Path("models/pretrained/MEGNet-BandGap-mfi-MP-2019.4.1"),
    Path("models/pretrained/TensorNet-PES-MatPES-PBE-2025.2"),
):
    model = matgl.load_model(str(path))
    saved = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    current = model.state_dict()
    assert not set(current) - set(saved), f"Missing weights in {path}"
    assert set(saved) - set(current) <= {"element_refs.onehot"}, f"Unexpected weights in {path}"
    assert all(current[key].shape == saved[key].shape for key in set(current) & set(saved))

si = Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
features = {"structure": si.as_dict()}
gap, error = _calculate_matgl(features, relax=False, max_steps=1)
assert error is None and gap is not None and math.isclose(gap, 0.33, abs_tol=0.03), (gap, error)
relaxed_gap, error = _calculate_matgl(features, relax=True, max_steps=1)
assert error is None and relaxed_gap is not None and math.isfinite(relaxed_gap), (relaxed_gap, error)
print("PyG model loads, weights, band-gap prediction, and one-step relaxation passed.")
