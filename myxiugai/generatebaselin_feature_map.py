import sys
sys.path.append("/home/user/zq/python/detection/ultralytics2")
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def save_feature_by_module(model, img_path, module, save_path):
    features = {}

    def hook_fn(module, input, output):
        if isinstance(output, (list, tuple)):
            output = output[0]
        features["feat"] = output.detach().cpu()

    handle = module.register_forward_hook(hook_fn)

    model(img_path)

    handle.remove()

    feat = features["feat"]  # [1, C, H, W]

    feat_map = feat[0].max(dim=0)[0].numpy()
    feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-6)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    feat_map = cv2.resize(feat_map, (img.shape[1], img.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * feat_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 0.55, heatmap, 0.45, 0)

    plt.figure(figsize=(4, 4))
    plt.imshow(overlay)
    plt.axis("off")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

    print("saved:", save_path)


if __name__ == "__main__":
    baseline_weight = "/home/user/storage/python/fuxian/yolo/runs/detect/weights/best.pt"
    img_path = "/home/user/storage/test/tile_n30741.jpg"

    save_dir = "/home/user/storage/yoloeffect/MRAF"
    os.makedirs(save_dir, exist_ok=True)

    baseline_model = YOLO(baseline_weight)

    # baseline 对应 MSFA 模型中第 9 层输出的位置
    # baseline 的第 8 层是 C2f，最适合作为对应层特征
    save_feature_by_module(
        baseline_model,
        img_path,
        baseline_model.model.model[8],
        f"{save_dir}/baseline_layer8_C2f.png"
    )