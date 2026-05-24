import os
import time
import uuid

import torch
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from utils import (
    calculate_mask_stats,
    create_overlay,
    load_model,
    predict_mask,
    preprocess_image,
    save_result_images,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "static", "results")
WEIGHT_PATH = os.path.join(BASE_DIR, "weights", "segformer_wbc_pseudo_mask.pth")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["RESULT_FOLDER"] = RESULT_DIR
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = load_model(WEIGHT_PATH, device)
    startup_error = None
except Exception as exc:
    model = None
    startup_error = str(exc)
    print(f"[启动错误] {startup_error}")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def static_url_from_path(path):
    rel_path = os.path.relpath(path, os.path.join(BASE_DIR, "static"))
    return "/static/" + rel_path.replace(os.sep, "/")


def normalize_threshold(raw_value):
    try:
        threshold = float(raw_value)
    except (TypeError, ValueError):
        threshold = 0.5
    return min(max(threshold, 0.1), 0.9)


def device_label():
    return "CUDA" if device.type == "cuda" else "CPU"


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        error=startup_error,
        device=device_label(),
        threshold=0.5,
    )


@app.route("/predict", methods=["POST"])
def predict():
    threshold = normalize_threshold(request.form.get("threshold", 0.5))

    if startup_error or model is None:
        return render_template(
            "index.html",
            error=startup_error,
            device=device_label(),
            threshold=threshold,
        )

    if "file" not in request.files:
        return render_template(
            "index.html",
            error="请先选择一张白细胞显微图像。",
            device=device_label(),
            threshold=threshold,
        )

    file = request.files["file"]
    if file.filename == "":
        return render_template(
            "index.html",
            error="请先选择一张白细胞显微图像。",
            device=device_label(),
            threshold=threshold,
        )

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error="文件格式不支持，请上传 jpg、jpeg 或 png 图片。",
            device=device_label(),
            threshold=threshold,
        )

    try:
        ext = file.filename.rsplit(".", 1)[1].lower()
        safe_name = secure_filename(file.filename)
        if not safe_name:
            safe_name = f"upload.{ext}"

        file_id = uuid.uuid4().hex
        upload_name = f"{file_id}_{safe_name}"
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], upload_name)
        file.save(upload_path)

        start_time = time.perf_counter()
        tensor, original_img = preprocess_image(upload_path)
        mask = predict_mask(model, tensor, device, threshold=threshold)
        overlay = create_overlay(original_img, mask)
        inference_time = (time.perf_counter() - start_time) * 1000

        foreground_pixels, area_ratio = calculate_mask_stats(mask)
        original_h, original_w = original_img.shape[:2]
        original_size = f"{original_w} × {original_h}"

        mask_path, overlay_path = save_result_images(
            mask,
            overlay,
            app.config["RESULT_FOLDER"],
            file_id,
        )

        return render_template(
            "index.html",
            original_url=static_url_from_path(upload_path),
            mask_url=static_url_from_path(mask_path),
            overlay_url=static_url_from_path(overlay_path),
            inference_time=f"{inference_time:.2f}",
            device=device_label(),
            threshold=f"{threshold:.2f}",
            original_size=original_size,
            foreground_pixels=foreground_pixels,
            area_ratio=f"{area_ratio:.2f}",
        )
    except Exception as exc:
        return render_template(
            "index.html",
            error=f"推理失败：{exc}",
            device=device_label(),
            threshold=threshold,
        )


if __name__ == "__main__":
    if startup_error:
        raise RuntimeError(startup_error)
    app.run(host="127.0.0.1", port=5000, debug=False)
