import json
import matplotlib
import numpy as np

from torchvision import transforms
from PIL import Image

import torch
from matplotlib import pyplot as plt
import os
from timm.models import create_model

import urllib.request


device = 'cuda'


def download_from_url(url, path=None, root="./"):
    if path is None:
        _, filename = os.path.split(url)
        root = os.path.abspath(root)
        path = os.path.join(root, filename)
    urllib.request.urlretrieve(url, path)
    print(f"Downloaded file to {path}")


def load_class_names(json_path):
    with open(json_path, "r") as f:
        return list(json.load(f).values())

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalized
    ])
    return transform(image)


@torch.inference_mode()
def predict_probs_for_image(model, image_path):
    image = preprocess_image(image_path).unsqueeze(0) # add batch dim
    model.eval()
    outputs = model(image.to(device))
    probs = torch.nn.functional.softmax(outputs, dim=1).cpu()
    return (probs[0] * 100).tolist()


def plot_probs(texts, probs, fig_ax, lang_type=None, save_path=None):
    # reverse the order to plot from top to bottom
    sorted_indices = np.argsort(probs)
    texts = np.array(texts)[sorted_indices]
    probs = np.array(probs)[sorted_indices]
    if fig_ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig, ax = fig_ax

    font_prop = matplotlib.font_manager.FontProperties(
        fname=lang_type_to_font_path(lang_type)
    )
    ax.barh(texts, probs, color="darkslateblue", height=0.3)
    ax.barh(texts, 100 - probs, color="silver", height=0.3, left=probs)
    for bar, label, val in zip(ax.patches, texts, probs):
        ax.text(
            0,
            bar.get_y() - bar.get_height(),
            label,
            color="black",
            ha="left",
            va="center",
            fontproperties=font_prop,
        )
        ax.text(
            bar.get_x() + bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f} %",
            fontweight="bold",
            ha="left",
            va="center",
        )

    ax.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")  # 保存图片并移除多余空白
        print(f"Figure saved to {save_path}")


def predict_probs_and_plot(
    model, image_path, texts, plot_image=True, fig_ax=None, lang_type=None
):
    if plot_image:
        fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(12, 6))
        image = Image.open(image_path).convert('RGB')
        ax_1.imshow(image)
        ax_1.axis("off")
    probs = predict_probs_for_image(model, image_path)
    plot_probs(texts, probs, (fig, ax_2), lang_type=lang_type, save_path='./prediction_probs.png')


def lang_type_to_font_path(lang_type):
    mapping = {
        None: "https://cdn.jsdelivr.net/gh/notofonts/notofonts.github.io/fonts/NotoSans/hinted/ttf/NotoSans-Regular.ttf",
        "cjk": "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
        "devanagari": "https://cdn.jsdelivr.net/gh/notofonts/notofonts.github.io/fonts/NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Regular.ttf",
        "emoji": "https://github.com/MorbZ/OpenSansEmoji/raw/master/OpenSansEmoji.ttf",
    }
    return download_from_url(mapping[lang_type])

if __name__ == '__main__':
    model = create_model(
        'mobilenetv4_conv_large'
    )
    model.reset_classifier(num_classes=5)
    model.load_state_dict(torch.load('./output/mobilenetv4_conv_large_best_checkpoint.pth')['model'])
    model.to(device)

    texts = load_class_names('./classes_indices.json')
    # image_path = r'D:/flower_data/roses/1666341535_99c6f7509f_n.jpg'
    image_path = r'D:/flower_data/sunflowers/44079668_34dfee3da1_n.jpg'
    predict_probs_and_plot(model, image_path, texts)