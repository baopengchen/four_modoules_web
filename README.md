# 白细胞细胞核分割 Web 应用

本项目是一个可本地运行的白细胞细胞核分割系统，使用 Flask 搭建 Web 服务，使用 PyTorch 加载训练好的 SegFormer-B0 轻量模型，对用户上传的白细胞显微图像进行细胞核分割。

页面会展示：

- 原始图像
- 预测二值掩膜
- 掩膜叠加到原图上的可视化结果
- Mask PNG 下载链接
- Overlay PNG 下载链接
- 单次推理耗时

## 项目结构

```text
wbc-nucleus-seg-web/
├── app.py
├── model.py
├── utils.py
├── requirements.txt
├── README.md
├── weights/
│   └── segformer_wbc_pseudo_mask.pth
├── static/
│   ├── css/
│   │   └── style.css
│   ├── uploads/
│   ├── results/
│   └── js/
│       └── main.js
└── templates/
    └── index.html
```

## 环境安装

建议使用 Python 3.9 或以上版本。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux 或 macOS 激活虚拟环境：

```bash
source .venv/bin/activate
```

如果需要使用 CUDA，请根据本机 CUDA 版本安装匹配的 PyTorch 版本。

## 权重文件放置位置

请将训练好的模型权重放在以下路径：

```text
weights/segformer_wbc_pseudo_mask.pth
```

如果该文件不存在，程序启动时会给出清晰报错。

## 启动命令

```bash
python app.py
```

默认访问地址：

```text
http://127.0.0.1:5000
```

程序会自动选择推理设备：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## 使用步骤

1. 打开浏览器访问 `http://127.0.0.1:5000`。
2. 点击上传区域，选择一张 `jpg`、`jpeg` 或 `png` 白细胞显微图像。
3. 点击“开始预测”。
4. 页面会显示原始图像、预测掩膜、叠加结果和推理耗时。
5. 点击“下载 Mask”或“下载 Overlay”保存结果图像。

## 注意事项

- 本系统使用伪掩膜训练模型。
- 输出结果仅用于实验展示和辅助分析。
- 不能作为临床诊断依据。
- 上传图片会保存到 `static/uploads`。
- 预测结果会保存到 `static/results`。
- 每次预测的结果文件名都会加入 uuid，避免覆盖历史结果。
