import numpy as np
import torch
import torch.nn.functional as F
from sybil.serie import Serie
from typing import Dict, List, Union
import cv2
import os


def visualize_attentions(
    series: Serie,
    attentions: List[Dict[str, torch.Tensor]],
    save_directory: str = None,
    gain: int = 3,
) -> List[List[np.ndarray]]:
    """
    Args:
        series (Serie): series object
        attention_dict (Dict[str, torch.Tensor]): attention dictionary output from model
        save_directory (str, optional): where to save the images. Defaults to None.
        gain (int, optional): how much to scale attention values by for visualization. Defaults to 3.

    Returns:
        List[List[np.ndarray]]: list of list of overlayed images
    """

    if isinstance(series, Serie):
        series = [series]

    series_overlays = []
    for serie_idx, serie in enumerate(series):
        a1 = attentions[serie_idx]["image_attention_1"]
        v1 = attentions[serie_idx]["volume_attention_1"]

        # TODO:
        if len(a1) > 1:
            a1 = a1.mean(0)
            v1 = v1.mean(0)

        attention = torch.exp(a1) * torch.exp(v1).unsqueeze(-1)
        attention = attention.view(1, 25, 16, 16)

        N = len(serie)
        attention_up = F.interpolate(
            attention.unsqueeze(0), (N, 512, 512), mode="trilinear"
        )

        # get original image
        images = serie.get_raw_images()

        overlayed_images = []
        for i in range(N):
            overlayed = np.zeros((512, 512, 3))
            overlayed[..., 0] = images[i]
            overlayed[..., 1] = images[i]
            overlayed[..., 2] = np.int16(
                np.clip(
                    (attention_up[0, 0, i] * gain * 256) + images[i],
                    a_min=0,
                    a_max=256,
                )
            )
            overlayed_images.append(overlayed)
        if save_directory is not None:
            save_path = os.path.join(save_directory, f"serie_{serie_idx}")
            save_images(overlayed_images, save_path, f"serie_{serie_idx}")

        series_overlays.append(overlayed_images)
    return series_overlays


def save_images(img_list, directory, name):
    os.makedirs(directory, exist_ok=True)
    N = len(str(len(img_list)))
    for i, im in enumerate(img_list):
        cv2.imwrite(f"{directory}/{name}_{'0'*(N - len(str(i))) }{i}.png", im)
