import os

import cv2
import numpy as np
import torch
from PIL import Image

from model import SegFormerB0


INPUT_SIZE = (256, 256)


def load_model(weight_path, device):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"模型权重文件不存在: {weight_path}\n"
            "请将 segformer_wbc_pseudo_mask.pth 放到 weights 目录下。"
        )

    model = SegFormerB0(in_ch=3, out_ch=1, decoder_dim=128).to(device)
    checkpoint = torch.load(weight_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    if isinstance(checkpoint, dict):
        checkpoint = {
            key.replace("module.", "", 1): value
            for key, value in checkpoint.items()
        }

    model.load_state_dict(checkpoint)
    model.eval()
    return model


def preprocess_image(image_path):
    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError("无法读取上传图像，请确认文件是有效的 jpg/jpeg/png 图片。") from exc

    original_img = np.array(pil_img, dtype=np.uint8)
    resized_img = pil_img.resize(INPUT_SIZE, Image.BILINEAR)
    resized = np.array(resized_img, dtype=np.float32)

    normalized = resized / 255.0
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    return tensor.float(), original_img


def predict_mask(model, tensor, device, threshold=0.5):
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)
        mask = (prob > threshold).float()

    mask_np = mask.squeeze().detach().cpu().numpy()
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    return mask_uint8


def create_overlay(original_img, mask):
    h, w = original_img.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = original_img.copy()
    red_layer = np.zeros_like(original_img, dtype=np.uint8)
    red_layer[:, :, 0] = 255

    mask_bool = mask_resized > 0
    alpha = 0.35
    overlay[mask_bool] = cv2.addWeighted(
        original_img[mask_bool],
        1.0 - alpha,
        red_layer[mask_bool],
        alpha,
        0,
    )

    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_bgr, contours, -1, (0, 0, 255), 2)
    overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return overlay


def calculate_mask_stats(mask):
    foreground_pixels = int(np.count_nonzero(mask > 0))
    total_pixels = int(mask.size)
    area_ratio = (foreground_pixels / total_pixels * 100.0) if total_pixels else 0.0
    return foreground_pixels, area_ratio


def save_result_images(mask, overlay, results_dir, file_id):
    os.makedirs(results_dir, exist_ok=True)

    mask_path = os.path.join(results_dir, f"{file_id}_mask.png")
    overlay_path = os.path.join(results_dir, f"{file_id}_overlay.png")

    Image.fromarray(mask.astype(np.uint8), mode="L").save(mask_path)
    Image.fromarray(overlay.astype(np.uint8), mode="RGB").save(overlay_path)
    return mask_path, overlay_path
