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

    # 用 max 比 mean 更容易突出强响应
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
    plt.savefig(save_path, dpi=150,bbox_inches="tight", pad_inches=0)
    plt.close()

    print("saved:", save_path)


if __name__ == "__main__":
    weight_path = "/home/user/runs/detect/yolov8MSFA/weights/best.pt"
    img_path = "/home/user/images/test/tile_n30741.jpg"

    save_dir = "/home/user/storage/yoloeffect/MRAF"
    os.makedirs(save_dir, exist_ok=True)

    model = YOLO(weight_path)

    # 第三个 MSFA，也就是 layer 9
    msfa = model.model.model[9]

    # 1. MSFA 输入前：layer 8 输出
    save_feature_by_module(
        model,
        img_path,
        model.model.model[8],
        f"{save_dir}/before_MSFA_layer8.png"
    )

    # 2. 标准卷积分支：3×3
    save_feature_by_module(
        model,
        img_path,
        msfa.branch1,
        f"{save_dir}/branch1_standard_3x3.png"
    )

    # 3. 非对称卷积分支：1×3 → 3×1
    save_feature_by_module(
        model,
        img_path,
        msfa.branch2,
        f"{save_dir}/branch2_asym_1x3_3x1.png"
    )

    # 4. 非对称卷积分支：3×1 → 1×3
    save_feature_by_module(
        model,
        img_path,
        msfa.branch3,
        f"{save_dir}/branch3_asym_3x1_1x3.png"
    )

    # 5. 空洞卷积分支
    save_feature_by_module(
        model,
        img_path,
        msfa.branch4,
        f"{save_dir}/branch4_dilation.png"
    )

    # 6. MSFA 输出后：layer 9 输出
    save_feature_by_module(
        model,
        img_path,
        model.model.model[11],
        f"{save_dir}/after_MSFA_layer11.png"
    )