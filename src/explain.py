import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import urllib
from typing import List, Tuple
import os
import hydra
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from captum.attr import (
    GradientShap,
    IntegratedGradients,
    NoiseTunnel,
    Occlusion,
    Saliency,
)
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
from omegaconf import DictConfig
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_lightning import LightningModule

from utils import get_pylogger

log = get_pylogger(__name__)


def get_integrated_gradients(
    model: timm,
    image_tensor: torch.Tensor,
    default_cmap: LinearSegmentedColormap,
    transformed_img: torch.Tensor,
    pred_label_idx: torch.Tensor,
    image_name: str
) -> None:
    """To explain the model using IntegratedGradients."""

    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(
        image_tensor, target=pred_label_idx, n_steps=200
    )

    plt_outputs = viz.visualize_image_attr(
        np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        method="heat_map",
        cmap=default_cmap,
        show_colorbar=True,
        sign="positive",
        outlier_perc=1,
    )
    lst = image_name.split('.')
    image_name_saved = lst[1] + '_ig.' + lst[-1]
    image_name_save_path = '.' + image_name_saved
    plt_outputs[0].savefig(image_name_save_path)

def get_noise_tunnel(
    model: timm,
    image_tensor: torch.Tensor,
    default_cmap: LinearSegmentedColormap,
    transformed_img: torch.Tensor,
    pred_label_idx: torch.Tensor,
    image_name: str
) -> None:
    """To explain the model using Noise tunnel with IntegratedGradients."""

    integrated_gradients = IntegratedGradients(model)
    noise_tunnel = NoiseTunnel(integrated_gradients)

    attributions_ig_nt = noise_tunnel.attribute(
        image_tensor, nt_samples=10, nt_type="smoothgrad_sq", target=pred_label_idx
    )

    plt_outputs = viz.visualize_image_attr_multiple(
        np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "positive"],
        cmap=default_cmap,
        show_colorbar=True,
    )
    lst = image_name.split('.')
    image_name_saved = lst[1] + '_noise.' + lst[-1]
    image_name_save_path = '.' + image_name_saved
    plt_outputs[0].savefig(image_name_save_path)


def get_shap(
    model: timm,
    image_tensor: torch.Tensor,
    default_cmap: LinearSegmentedColormap,
    transformed_img: torch.Tensor,
    pred_label_idx: torch.Tensor,
    image_name: str
) -> None:
    """To explain the model using SHAP."""

    gradient_shap = GradientShap(model)

    # Defining baseline distribution of images
    rand_img_dist = torch.cat([image_tensor * 0, image_tensor * 1])

    attributions_gs = gradient_shap.attribute(
        image_tensor, n_samples=50, stdevs=0.0001, baselines=rand_img_dist, target=pred_label_idx
    )

    plt_outputs = viz.visualize_image_attr_multiple(
        np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "absolute_value"],
        cmap=default_cmap,
        show_colorbar=True,
    )
    lst = image_name.split('.')
    image_name_saved = lst[1] + '_shap.' + lst[-1]
    image_name_save_path = '.' + image_name_saved
    plt_outputs[0].savefig(image_name_save_path)    


def get_occlusion(
    model: timm,
    image_tensor: torch.Tensor,
    transformed_img: torch.Tensor,
    pred_label_idx: torch.Tensor,
    image_name: str
) -> None:
    """To explain the model using Occlusion."""

    occlusion = Occlusion(model)

    attributions_occ = occlusion.attribute(
        image_tensor,
        strides=(3, 8, 8),
        target=pred_label_idx,
        sliding_window_shapes=(3, 15, 15),
        baselines=0,
    )

    plt_outputs = viz.visualize_image_attr_multiple(
        np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "positive"],
        show_colorbar=True,
        outlier_perc=2,
    )
    lst = image_name.split('.')
    image_name_saved = lst[1] + '_occlusion.' + lst[-1]
    image_name_save_path = '.' + image_name_saved
    plt_outputs[0].savefig(image_name_save_path)


def get_saliency(model: timm, image_tensor: torch.Tensor, pred_label_idx: torch.Tensor, image_name: str) -> None:
    """To explain the model using Saliency."""

    saliency = Saliency(model)
    grads = saliency.attribute(image_tensor, target=pred_label_idx)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_transform = T.Compose(
        [
            T.Normalize(
                mean=(-1 * np.array(mean) / np.array(std)).tolist(),
                std=(1 / np.array(std)).tolist(),
            )
        ]
    )

    original_image = np.transpose(
        inv_transform(image_tensor).squeeze(0).cpu().detach().numpy(), (1, 2, 0)
    )
    # print(f"{original_image.shape}")

    _ = viz.visualize_image_attr(
        None, original_image, method="original_image", title="Original Image"
    )
    plt_outputs = viz.visualize_image_attr(
        grads,
        original_image,
        method="blended_heat_map",
        sign="absolute_value",
        show_colorbar=True,
        title="Overlaid Gradient Magnitudes",
    )
    lst = image_name.split('.')
    image_name_saved = lst[1] + '_saliency.' + lst[-1]
    image_name_save_path = '.' + image_name_saved
    plt_outputs[0].savefig(image_name_save_path)    


def get_gradcam(model, image_tensor: torch.Tensor, pred_label_idx: torch.Tensor, image_name: str) -> None:
    """To explain the model using GradCAM."""

    target_layers = [model.net.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)  # ,use_cuda=True)
    targets = [ClassifierOutputTarget(pred_label_idx)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_transform = T.Compose(
        [
            T.Normalize(
                mean=(-1 * np.array(mean) / np.array(std)).tolist(),
                std=(1 / np.array(std)).tolist(),
            )
        ]
    )

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    rgb_img = inv_transform(image_tensor).cpu().squeeze().permute(1, 2, 0).detach().numpy()
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # plt.imshow(visualization)
    # plt.show()
    lst = image_name.split('.')
    image_name_saved = lst[1] + '_gradcam.' + lst[-1]
    image_name_save_path = '.' + image_name_saved
    pil_image = Image.fromarray(visualization)
    pil_image.save(image_name_save_path)


def get_gradcamplusplus(model, image_tensor: torch.Tensor, pred_label_idx: torch.Tensor, image_name: str) -> None:
    """To explain the model using GradCAM++"""

    target_layers = [model.net.layer4[-1]]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)  # , use_cuda=True)
    targets = [ClassifierOutputTarget(pred_label_idx)]

    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_transform = T.Compose(
        [
            T.Normalize(
                mean=(-1 * np.array(mean) / np.array(std)).tolist(),
                std=(1 / np.array(std)).tolist(),
            )
        ]
    )

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    rgb_img = inv_transform(image_tensor).cpu().squeeze().permute(1, 2, 0).detach().numpy()
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # plt.imshow(visualization)
    # plt.show()
    lst = image_name.split('.')
    image_name_saved = lst[1] + '_gradcamplusplus.' + lst[-1]
    image_name_save_path = '.' + image_name_saved
    pil_image = Image.fromarray(visualization)
    pil_image.save(image_name_save_path)   


def explain_model(cfg: DictConfig) -> None:
    """Function to implement various model explanation technique
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        None
    """

    log.info("Running Model explanation")

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model.eval()
    log.info(f"Loaded Model")

    # Download human-readable labels for ImageNet and get the class names
    # url, filename = (
    #     "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
    #     "imagenet_classes.txt",
    # )
    # urllib.request.urlretrieve(url, filename)

    with open(cfg.imagenet_classes) as f:
        categories = [s.strip() for s in f.readlines()]

    transforms = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
        ]
    )
    transform_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = Image.open(cfg.input_image)
    transformed_img = transforms(image)
    image_tensor = transform_normalize(transformed_img)
    image_tensor = image_tensor.unsqueeze(0)

    # img_tensor = img_tensor.to(device)
    output = model(image_tensor)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    predicted_label = categories[pred_label_idx.item()]
    log.info(
        f"Predicted: {predicted_label} (confidence = {prediction_score.squeeze().item()}, index = {pred_label_idx.item()})"
    )

    default_cmap = LinearSegmentedColormap.from_list(
        "white to red", [(0, "#ffffff"), (0.25, "#ff0000"), (1, "#ff0000")], N=256
    )
    image_name = cfg.input_image

    if 1 in cfg.model_explain:
        log.info("Integrated Gradients :")
        get_integrated_gradients(
            model, image_tensor, default_cmap, transformed_img, pred_label_idx, image_name
        )

    if 2 in cfg.model_explain:
        log.info("Noise Tunnel :")
        get_noise_tunnel(model, image_tensor, default_cmap, transformed_img, pred_label_idx, image_name)

    if 3 in cfg.model_explain:
        log.info("SHAP :")
        get_shap(model, image_tensor, default_cmap, transformed_img, pred_label_idx, image_name)

    if 4 in cfg.model_explain:
        log.info("Occlusion :")
        get_occlusion(model, image_tensor, transformed_img, pred_label_idx, image_name)

    if 5 in cfg.model_explain:
        log.info("Saliency :")
        image_tensor_grad = image_tensor
        image_tensor_grad.requires_grad = True
        get_saliency(model, image_tensor_grad, pred_label_idx, image_name)

    if 6 in cfg.model_explain:
        log.info("GradCAM :")
        image_tensor_grad = image_tensor
        image_tensor_grad.requires_grad = True
        get_gradcam(model, image_tensor_grad, pred_label_idx, image_name)

    if 7 in cfg.model_explain:
        log.info("GradCAMPlusPlus :")
        image_tensor_grad = image_tensor
        image_tensor_grad.requires_grad = True
        get_gradcamplusplus(model, image_tensor_grad, pred_label_idx, image_name)

@hydra.main(version_base="1.2", config_path="../configs", config_name="explain.yaml")
def main(cfg: DictConfig) -> None:
    image_lst = os.listdir(cfg.input_image_folder)
    for image_file_name in image_lst:
        if ('palace' in image_file_name) or \
           ('goldfish' in image_file_name):
            cfg.input_image = cfg.input_image_folder + image_file_name
            log.info(f'Explaining {cfg.input_image}.......')
            explain_model(cfg)
        else:
            log.info(f'Bypassing {image_file_name} as it is already explained before.......')

if __name__ == "__main__":
    main()