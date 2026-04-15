import torch

_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from ultralytics import YOLO

for name in ["yolov8n", "yolov8s", "yolov8l"]:
    print(f"Exporting {name}.pt → {name}.onnx ...")
    YOLO(f"{name}.pt").export(format="onnx", dynamic=True, simplify=False, opset=12)
    print(f"Done: {name}.onnx")
