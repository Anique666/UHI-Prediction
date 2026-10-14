"""UHI deployment layer.

Isolated, additive package: exports the retained ConvLSTM to ONNX, validates
numerical parity against CPU PyTorch, and provides baseline + mitigation
inference. It reuses the authoritative preprocessing / reconstruction / UHI /
strategy implementations and never modifies the frozen model or dataset.

Nothing here imports or writes training artifacts beyond the checkpoint reader.
"""