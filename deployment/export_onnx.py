"""Export the retained ConvLSTM to ONNX (opset 17, dynamic batch).

Reads the production checkpoint read-only via ``training.engine.load_checkpoint``
(strict state-dict load), exports to ``deployment/artifacts/convlstm_lst.onnx``,
validates the graph with the ONNX checker and an onnxruntime CPU session, and
records all hashes. Never writes the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment.onnx_contract import (  # noqa: E402
    ContractError,
    load_contract,
    sha256_file,
)

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
DEFAULT_ONNX = ARTIFACTS / "convlstm_lst.onnx"


def build_reference_model(contract, checkpoint_path=None):
    """CPU model in eval mode with strict state-dict loading (read-only)."""
    import torch

    from models.convlstm import ConvLSTM
    from training.engine import load_checkpoint

    m = contract.model
    model = ConvLSTM(
        input_channels=len(contract.channel_names),
        hidden_channels=int(m["hidden_channels"]),
        kernel_size=int(m["kernel_size"]),
        output_channels=int(m["output_channels"]),
    )
    ckpt = load_checkpoint(
        checkpoint_path or contract.checkpoint_path, model, map_location="cpu"
    )
    # Fail loudly if the checkpoint exposes a different architecture.
    if "model_state_dict" not in ckpt:
        raise ContractError("Checkpoint has no 'model_state_dict' key")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def export(contract, out_path: Path = DEFAULT_ONNX, checkpoint_path=None) -> dict:
    import onnx
    import torch

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sha_before = sha256_file(checkpoint_path or contract.checkpoint_path)

    model, ckpt = build_reference_model(contract, checkpoint_path)

    dummy = torch.zeros(
        (1, *contract.input_shape_tail), dtype=torch.float32
    )
    with torch.no_grad():
        torch_out = model(dummy)
    if tuple(torch_out.shape) != (1, *contract.output_shape_tail):
        raise ContractError(
            f"Unexpected torch output shape {tuple(torch_out.shape)}"
        )

    dynamic_axes = {
        contract.input_name: {0: "batch"},
        contract.output_name: {0: "batch"},
    }
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        opset_version=contract.onnx_opset,
        input_names=[contract.input_name],
        output_names=[contract.output_name],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        export_params=True,
        # torch 2.13 defaults to the dynamo exporter, which requires the
        # optional onnxscript package.  The stable legacy exporter fully
        # supports this model and opset 17 without changing the environment.
        dynamo=False,
    )

    # --- validate the graph
    onnx_model = onnx.load(str(out_path))
    # The model architecture fixes all dimensions apart from batch.  The legacy
    # exporter leaves output H/W symbolic because they flow through shape ops;
    # make the public ONNX contract explicit without changing the graph.
    out_shape = onnx_model.graph.output[0].type.tensor_type.shape.dim
    for dim, value in zip(out_shape[1:], contract.output_shape_tail):
        dim.ClearField("dim_param")
        dim.dim_value = int(value)
    onnx.save(onnx_model, str(out_path))
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    graph_ops = sorted({n.op_type for n in onnx_model.graph.node})

    # --- validate with onnxruntime CPU on a batch-2 input
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    in_meta = sess.get_inputs()[0]
    out_meta = sess.get_outputs()[0]
    if in_meta.name != contract.input_name or out_meta.name != contract.output_name:
        raise ContractError("ONNX session I/O names do not match the contract")
    if in_meta.type != "tensor(float)" or out_meta.type != "tensor(float)":
        raise ContractError("ONNX I/O must be float32")
    sample = np.zeros((2, *contract.input_shape_tail), dtype=np.float32)
    got = sess.run([contract.output_name], {contract.input_name: sample})[0]
    if tuple(got.shape) != (2, *contract.output_shape_tail):
        raise ContractError(f"ONNX batch-2 output shape {tuple(got.shape)} unexpected")

    sha_after = sha256_file(checkpoint_path or contract.checkpoint_path)
    if sha_before != sha_after:
        raise ContractError("Checkpoint changed during export — aborting")

    report = {
        "onnx_path": str(out_path),
        "onnx_sha256": sha256_file(out_path),
        "checkpoint_path": str(checkpoint_path or contract.checkpoint_path),
        "checkpoint_sha256_before": sha_before,
        "checkpoint_sha256_after": sha_after,
        "checkpoint_unchanged": sha_before == sha_after,
        "opset": contract.onnx_opset,
        "ir_version": int(onnx_model.ir_version),
        "producer": f"{onnx_model.producer_name} {onnx_model.producer_version}".strip(),
        "input": {"name": in_meta.name, "type": in_meta.type, "shape": in_meta.shape},
        "output": {"name": out_meta.name, "type": out_meta.type, "shape": out_meta.shape},
        "graph_ops": graph_ops,
        "dynamic_axes": dynamic_axes,
        "dependency_versions": contract.dependency_versions,
    }
    (out_path.parent / "export_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Export the retained ConvLSTM to ONNX")
    ap.add_argument("--out", default=str(DEFAULT_ONNX))
    ap.add_argument("--checkpoint", default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    contract = load_contract(checkpoint_path=args.checkpoint)
    report = export(contract, Path(args.out), args.checkpoint)
    print(f"ONNX written       : {report['onnx_path']}")
    print(f"ONNX SHA-256       : {report['onnx_sha256']}")
    print(f"Checkpoint SHA-256 : {report['checkpoint_sha256_before']} (unchanged={report['checkpoint_unchanged']})")
    print(f"opset/ir_version   : {report['opset']} / {report['ir_version']}")
    print(f"input/output       : {report['input']['name']} {report['input']['shape']} -> "
          f"{report['output']['name']} {report['output']['shape']}")
    print(f"graph ops          : {', '.join(report['graph_ops'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
