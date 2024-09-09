#!/usr/bin/env python

__doc__ = """
This example shows how to use a client to access a 
remote Sybil server (running Ark) to predict risk scores for a set of DICOM files.

The server must be started separately.

https://github.com/reginabarzilaygroup/Sybil/wiki
https://github.com/reginabarzilaygroup/ark/wiki
"""
import json
import os

import numpy as np
import requests

import sybil.utils.visualization

from utils import get_demo_data

if __name__ == "__main__":

    dicom_files = get_demo_data()
    serie = sybil.Serie(dicom_files)

    # Set the URL of the remote Sybil server
    ark_hostname = "localhost"
    ark_port = 5000

    # Set the URL of the remote Sybil server
    ark_host = f"http://{ark_hostname}:{ark_port}"

    data_dict = {"return_attentions": True}
    payload = {"data": json.dumps(data_dict)}

    # Check if the server is running and reachable
    resp = requests.get(f"{ark_host}/info")
    if resp.status_code != 200:
        raise ValueError(f"Failed to connect to ARK server. Status code: {resp.status_code}")

    info_data = resp.json()["data"]
    assert info_data["modelName"].lower() == "sybil", "The ARK server is not running Sybil"
    print(f"ARK server info: {info_data}")

    # Submit prediction to ARK server.
    files = [('dicom', open(file_path, 'rb')) for file_path in dicom_files]
    r = requests.post(f"{ark_host}/dicom/files", files=files, data=payload)
    _ = [f[1].close() for f in files]
    if r.status_code != 200:
        raise ValueError(f"Error occurred processing DICOM files. Status code: {r.status_code}.\n{r.text}")

    r_json = r.json()
    predictions = r_json["data"]["predictions"]

    scores = predictions[0]
    print(f"Risk scores: {scores}")

    attentions = predictions[1]
    attentions = np.array(attentions)
    print(f"Ark received attention shape: {attentions.shape}")

    # Visualize attention maps
    save_directory = "remote_ark_sybil_attention_output"

    print(f"Writing attention images to {save_directory}")

    images = serie.get_raw_images()
    overlayed_images = sybil.utils.visualization.build_overlayed_images(images, attentions, gain=3)

    if save_directory is not None:
        serie_idx = 0
        save_path = os.path.join(save_directory, f"serie_{serie_idx}")
        sybil.utils.visualization.save_images(overlayed_images, save_path, f"serie_{serie_idx}")

    print(f"Finished writing attention images to {save_directory}")

