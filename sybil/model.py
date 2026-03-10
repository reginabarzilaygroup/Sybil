from argparse import Namespace
from io import BytesIO
import os
from typing import NamedTuple, Union, Dict, List, Optional, Tuple
from urllib.request import urlopen
from zipfile import ZipFile
import ants
import cc3d
import torch
import pydicom
import numpy as np

import torch.nn.functional as F
from sybil.serie import Serie
from sybil.models.sybil import SybilNet
from sybil.models.sybil2 import Sybil17
from sybil.models.nnunet import nnUNet, nnUNetConfidence
from sybil.models.calibrator import SimpleClassifierGroup
from loguru import logger
from sybil.utils.device_utils import get_default_device, get_most_free_gpu, get_device_mem_info
from sybil.utils.registration import (
    transform_index_to_physicalpoint,
    transform_physicalpoint_to_index,
    resize_xy_fast,
)
from sybil.utils.nodule_tracking import link_nodules_by_center_distance
from lungmask import LMInferer

# Leaving this here for a bit; these are IDs to download the models from Google Drive
NAME_TO_FILE = {
    "sybil_base": {
        "checkpoint": ["28a7cd44f5bcd3e6cc760b65c7e0d54d"],
        "google_checkpoint_id": ["1ftYbav_BbUBkyR3HFCGnsp-h4uH1yhoz"],
        "google_calibrator_id": "1F5TOtzueR-ZUvwl8Yv9Svs2NPP5El3HY",
    },
    "sybil_1": {
        "checkpoint": ["28a7cd44f5bcd3e6cc760b65c7e0d54d"],
        "google_checkpoint_id": ["1ftYbav_BbUBkyR3HFCGnsp-h4uH1yhoz"],
        "google_calibrator_id": "1F5TOtzueR-ZUvwl8Yv9Svs2NPP5El3HY",
    },
    "sybil_2": {
        "checkpoint": ["56ce1a7d241dc342982f5466c4a9d7ef"],
        "google_checkpoint_id": ["1rscGi1grSxaVGzn-tqKtuAR3ipo0DWgA"],
        "google_calibrator_id": "1zKLVYBaiuMOx7p--e2zabs1LbQ-XXxcZ",
    },
    "sybil_3": {
        "checkpoint": ["624407ef8e3a2a009f9fa51f9846fe9a"],
        "google_checkpoint_id": ["1DV0Ge7n9r8WAvBXyoNRPwyA7VL43csAr"],
        "google_calibrator_id": "1qh4nawgE2Kjf_H97XuuTpL7XUIX7JOJn",
    },
    "sybil_4": {
        "checkpoint": ["64a91b25f84141d32852e75a3aec7305"],
        "google_checkpoint_id": ["1Acz_yzdJMpkz3PRrjXy526CjAboMEIHX"],
        "google_calibrator_id": "1QIvvCYLaesPGMEiE2Up77pKL3ygDdGU2",
    },
    "sybil_5": {
        "checkpoint": ["65fd1f04cb4c5847d86a9ed8ba31ac1a"],
        "google_checkpoint_id": ["1uV58SD-Qtb6xElTzWPDWWnloH1KB_zrP"],
        "google_calibrator_id": "1yDq1_A5w-fSdxzq4K2YSBRNcQQkDnH0K",
    },
    "sybil_ensemble": {
        "checkpoint": [
            "28a7cd44f5bcd3e6cc760b65c7e0d54d",
            "56ce1a7d241dc342982f5466c4a9d7ef",
            "624407ef8e3a2a009f9fa51f9846fe9a",
            "64a91b25f84141d32852e75a3aec7305",
            "65fd1f04cb4c5847d86a9ed8ba31ac1a",
        ],
        "google_checkpoint_id": [
            "1ftYbav_BbUBkyR3HFCGnsp-h4uH1yhoz",
            "1rscGi1grSxaVGzn-tqKtuAR3ipo0DWgA",
            "1DV0Ge7n9r8WAvBXyoNRPwyA7VL43csAr",
            "1Acz_yzdJMpkz3PRrjXy526CjAboMEIHX",
            "1uV58SD-Qtb6xElTzWPDWWnloH1KB_zrP",
        ],
        "google_calibrator_id": "1FxHNo0HqXYyiUKE_k2bjatVt9e64J9Li",
    },
}

CHECKPOINT_URL = os.getenv("SYBIL_CHECKPOINT_URL", "https://github.com/reginabarzilaygroup/Sybil/releases/download/v1.5.0/sybil_checkpoints.zip")


class Prediction(NamedTuple):
    scores: List[List[float]]
    attentions: List[Dict[str, np.ndarray]] = None


class Evaluation(NamedTuple):
    auc: List[float]
    c_index: float
    scores: List[List[float]]
    attentions: List[Dict[str, np.ndarray]] = None


def download_sybil(name, cache) -> Tuple[List[str], str]:
    """Download trained models and calibrator"""
    # Create cache folder if not exists
    cache = os.path.expanduser(cache)
    os.makedirs(cache, exist_ok=True)

    # Download models
    model_files = NAME_TO_FILE[name]
    checkpoints = model_files["checkpoint"]
    download_calib_path = os.path.join(cache, f"{name}_simple_calibrator.json")
    have_all_files = os.path.exists(download_calib_path)

    download_model_paths = []
    for checkpoint in checkpoints:
        cur_checkpoint_path = os.path.join(cache, f"{checkpoint}.ckpt")
        have_all_files &= os.path.exists(cur_checkpoint_path)
        download_model_paths.append(cur_checkpoint_path)

    if not have_all_files:
        print(f"Downloading models to {cache}")
        download_and_extract(CHECKPOINT_URL, cache)

    return download_model_paths, download_calib_path


def download_and_extract(remote_url: str, local_dir: str) -> List[str]:
    os.makedirs(local_dir, exist_ok=True)
    resp = urlopen(remote_url)
    with ZipFile(BytesIO(resp.read())) as zip_file:
        all_files_and_dirs = zip_file.namelist()
        zip_file.extractall(local_dir)
    return all_files_and_dirs


def _torch_set_num_threads(threads) -> int:
    """
    Set the number of CPU threads for torch to use.
    Set to a negative number for no-op.
    Set to 0 for the number of CPUs.
    """
    if threads < 0:
        return torch.get_num_threads()
    if threads is None or threads == 0:
        # I've never seen a benefit to going higher than 8 and sometimes there is a big slowdown
        threads = min(8, os.cpu_count())

    torch.set_num_threads(threads)
    return torch.get_num_threads()


class Sybil:
    def __init__(
        self,
        name_or_path: Union[List[str], str] = "sybil_ensemble",
        cache: str = "~/.sybil/",
        calibrator_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize a trained Sybil model for inference.

        Parameters
        ----------
        name_or_path: list or str
            Alias to a provided pretrained Sybil model or path
            to a sybil checkpoint.
        cache: str
            Directory to download model checkpoints to
        calibrator_path: str
            Path to calibrator pickle file corresponding with model
        device: str
            If provided, will run inference using this device.
            By default, uses GPU with the most free memory, if available.

        """

        # Download if needed
        if isinstance(name_or_path, str) and (name_or_path in NAME_TO_FILE):
            name_or_path, calibrator_path = download_sybil(name_or_path, cache)

        elif not all(os.path.exists(p) for p in name_or_path):
            raise ValueError(
                "No saved model or local path: {}".format(
                    [p for p in name_or_path if not os.path.exists(p)]
                )
            )

        # Check calibrator path before continuing
        if (calibrator_path is not None) and (not os.path.exists(calibrator_path)):
            raise ValueError(f"Path not found for calibrator {calibrator_path}")

        # Set device.
        # If set manually, use it and stay there.
        # Otherwise, pick the most free GPU now and at predict time.
        self._device_flexible = True
        if device is not None:
            self.device = device
            self._device_flexible = False
        else:
            self.device = get_default_device()

        self.ensemble = torch.nn.ModuleList()
        for path in name_or_path:
            self.ensemble.append(self.load_model(path))
        self.to(self.device)

        if calibrator_path is not None:
            self.calibrator = SimpleClassifierGroup.from_json_grouped(calibrator_path)
        else:
            self.calibrator = None

    def load_model(self, path):
        """Load model from path.

        Parameters
        ----------
        path : str
            Path to a sybil checkpoint.

        Returns
        -------
        model
            Pretrained Sybil model
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location="cpu")
        args = checkpoint["args"]
        self._max_followup = args.max_followup
        self._censoring_dist = args.censoring_distribution
        model = SybilNet(args)

        # Remove model from param names
        state_dict = {k[6:]: v for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict)  # type: ignore
        if self.device is not None:
            model.to(self.device)

        # Set eval
        model.eval()
        logger.info(f"Loaded model from {path}")
        return model

    def _calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Calibrate raw predictions

        Parameters
        ----------
        scores: np.ndarray
            risk scores as numpy array

        Returns
        -------
            np.ndarray: calibrated risk scores as numpy array
        """
        if self.calibrator is None:
            return scores

        calibrated_scores = []
        for YEAR in range(scores.shape[1]):
            probs = scores[:, YEAR].reshape(-1, 1)
            probs = self.calibrator["Year{}".format(YEAR + 1)].predict_proba(probs)[:, -1]
            calibrated_scores.append(probs)

        return np.stack(calibrated_scores, axis=1)

    def _predict(
        self,
        model: SybilNet,
        series: Union[Serie, List[Serie]],
        return_attentions: bool = False,
    ) -> Prediction:
        """Run predictions over the given serie(s).

        Parameters
        ----------
        model: SybilNet
            Instance of SybilNet
        series : Union[Serie, Iterable[Serie]]
            One or multiple series to run predictions for.
        return_attentions : bool
            If True, returns attention scores for each serie. See README for details.

        Returns
        -------
        Prediction
            Output prediction as risk scores.

        """
        if isinstance(series, Serie):
            series = [series]
        elif not isinstance(series, list):
            raise ValueError("Expected either a Serie object or list of Serie objects.")

        scores: List[List[float]] = []
        attentions: List[Dict[str, np.ndarray]] = [] if return_attentions else None
        for serie in series:
            if not isinstance(serie, Serie):
                raise ValueError("Expected a list of Serie objects.")

            volume = serie.get_volume()
            if self.device is not None:
                volume = volume.to(self.device)

            with torch.no_grad():
                out = model(volume)
                score = out["logit"].sigmoid().squeeze(0).cpu().numpy()
                scores.append(score.tolist())
                if return_attentions:
                    attentions.append(
                        {
                            "image_attention_1": out["image_attention_1"]
                            .detach()
                            .cpu(),
                            "volume_attention_1": out["volume_attention_1"]
                            .detach()
                            .cpu(),
                            "hidden": out["hidden"]
                            .detach()
                            .cpu(),
                        }
                    )

        return Prediction(scores=scores, attentions=attentions)

    def predict(
        self, series: Union[Serie, List[Serie]], return_attentions: bool = False, threads=0,
    ) -> Prediction:
        """Run predictions over the given serie(s) and ensemble

        Parameters
        ----------
        series : Union[Serie, Iterable[Serie]]
            One or multiple series to run predictions for.
        return_attentions : bool
            If True, returns attention scores for each serie. See README for details.
        threads : int
            Number of CPU threads to use for PyTorch inference.

        Returns
        -------
        Prediction
            Output prediction. See details for :class:`~sybil.model.Prediction`".

        """

        # Set CPU threads available to torch
        num_threads = _torch_set_num_threads(threads)
        logger.debug(f"Using {num_threads} threads for PyTorch inference")

        if self._device_flexible:
            self.device = self._pick_device()
            self.to(self.device)
        logger.debug(f"Beginning prediction on device: {self.device}")

        scores = []
        attentions_ = [] if return_attentions else None
        attention_keys = None
        for sybil in self.ensemble:
            pred = self._predict(sybil, series, return_attentions)
            scores.append(pred.scores)
            if return_attentions:
                attentions_.append(pred.attentions)
                if attention_keys is None:
                    attention_keys = pred.attentions[0].keys()

        scores = np.mean(np.array(scores), axis=0)
        calib_scores = self._calibrate(scores).tolist()

        attentions = None
        if return_attentions:
            attentions = []
            for i in range(len(series)):
                att = {}
                for key in attention_keys:
                    att[key] = np.stack([
                        attentions_[j][i][key] for j in range(len(self.ensemble))
                    ])
                attentions.append(att)

        return Prediction(scores=calib_scores, attentions=attentions)

    def evaluate(
        self, series: Union[Serie, List[Serie]], return_attentions: bool = False
    ) -> Evaluation:
        """Run evaluation over the given serie(s).

        Parameters
        ----------
        series : Union[Serie, List[Serie]]
            One or multiple series to run evaluation for.
        return_attentions : bool
            If True, returns attention scores for each serie. See README for details.

        Returns
        -------
        Evaluation
            Output evaluation. See details for :class:`~sybil.model.Evaluation`.

        """
        from sybil.utils.metrics import get_survival_metrics
        if isinstance(series, Serie):
            series = [series]
        elif not isinstance(series, list):
            raise ValueError(
                "Expected either a Serie object or an iterable over Serie objects."
            )

        # Check all have labels
        if not all(serie.has_label() for serie in series):
            raise ValueError("All series must have a label for evaluation")

        # Get scores and labels
        predictions = self.predict(series, return_attentions)
        scores = predictions.scores
        labels = [serie.get_label(self._max_followup) for serie in series]

        # Convert to format for survival metrics
        input_dict = {
            "probs": torch.tensor(scores),
            "censors": torch.tensor([label.censor_time for label in labels]),
            "golds": torch.tensor([label.y for label in labels]),
        }
        args = Namespace(
            max_followup=self._max_followup, censoring_distribution=self._censoring_dist
        )
        out = get_survival_metrics(input_dict, args)
        auc = [float(out[f"{i + 1}_year_auc"]) for i in range(self._max_followup)]
        c_index = float(out["c_index"])

        return Evaluation(auc=auc, c_index=c_index, scores=scores, attentions=predictions.attentions)

    def to(self, device: str):
        """Move model to device.

        Parameters
        ----------
        device : str
            Device to move model to.
        """
        self.device = device
        self.ensemble.to(device)

    def _pick_device(self):
        """
        Pick the device to run inference on.
        This is based on the device with the most free memory, with a preference for remaining
        on the current device.

        Motivation is to enable multiprocessing without the processes needed to communicate.
        """
        if not torch.cuda.is_available():
            return get_default_device()

        # Get size of the model in memory (approximate)
        model_mem = 9*sum(p.numel() * p.element_size() for p in self.ensemble.parameters())

        # Check memory available on current device.
        # If it seems like we're the only thing on this GPU, stay.
        free_mem, total_mem = get_device_mem_info(self.device)
        cur_allocated = total_mem - free_mem
        min_to_move = int(1.01 * model_mem)
        if cur_allocated < min_to_move:
            return self.device
        else:
            # Otherwise, get the most free GPU
            return get_most_free_gpu()


class Sybil2: 
    def __init__(
        self,
        name_or_path: Union[List[str], str] = "sybil_ensemble",
        cache: str = "~/.sybil/",
        calibrator_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize a trained Sybil model for inference.

        Parameters
        ----------
        name_or_path: list or str
            Alias to a provided pretrained Sybil model or path
            to a sybil checkpoint.
        cache: str
            Directory to download model checkpoints to
        calibrator_path: str
            Path to calibrator pickle file corresponding with model
        device: str
            If provided, will run inference using this device.
            By default, uses GPU with the most free memory, if available.

        """

        self._cache = cache
        logger.info(f"Initializing Sybil2 with cache={cache}")

        # Download if needed
        if isinstance(name_or_path, str) and (name_or_path in NAME_TO_FILE):
            name_or_path, calibrator_path = download_sybil(name_or_path, cache)

        elif not all(os.path.exists(p) for p in name_or_path):
            raise ValueError(
                "No saved model or local path: {}".format(
                    [p for p in name_or_path if not os.path.exists(p)]
                )
            )

        # Check calibrator path before continuing
        if (calibrator_path is not None) and (not os.path.exists(calibrator_path)):
            raise ValueError(f"Path not found for calibrator {calibrator_path}")

        # Set device.
        # If set manually, use it and stay there.
        # Otherwise, pick the most free GPU now and at predict time.
        self._device_flexible = True
        if device is not None:
            self.device = device
            self._device_flexible = False
        else:
            self.device = get_default_device()
        logger.info(f"Sybil2 using device={self.device} (flexible={self._device_flexible})")

        # Load model(s)
        self.lung_mask_model = self.load_lungmask_model("/data/rbg/users/pgmikhael/current/lungmask/checkpoints/unet_r231-d5d2fc3d.pth")

        self.segmentation_model = self.load_segmentation_model(name_or_path[0])
        
        self.confidence_model = self.load_confidence_model(name_or_path[0])

        self.ensemble = torch.nn.ModuleList()
        for path in name_or_path:
            self.ensemble.append(self.load_model(path))
        self.to(self.device)
        logger.info(f"Loaded Sybil2 ensemble with {len(self.ensemble)} model(s)")

        if calibrator_path is not None:
            self.calibrator = SimpleClassifierGroup.from_json_grouped(calibrator_path)
            logger.info(f"Loaded Sybil2 calibrator from {calibrator_path}")
        else:
            self.calibrator = None
            logger.info("No Sybil2 calibrator provided; using raw ensemble predictions")

    def load_model(self, path):
        logger.debug(f"Loading Sybil2 malignancy model from {path}")
        args = None
        model = Sybil17(args)
        logger.debug("Initialized Sybil17 malignancy model")
        return model
    
    def load_lungmask_model(self, path):
        # "/data/rbg/users/pgmikhael/current/lungmask/checkpoints/unet_r231-d5d2fc3d.pth"
        logger.info(f"Loading lung mask model from {path}")
        model = LMInferer(
            modelpath=path,
            tqdm_disable=True,
            batch_size=100, # double check
            force_cpu=True, # We force CPU here since the model is small and this avoids GPU memory issues when running with Sybil on the same GPU
        )
        logger.info("Lung mask model initialized on CPU")
        raise NotImplementedError("Lung segmentation is not yet implemented. Stay tuned!")
    
    def load_segmentation_model(self, path):
        # /data/rbg/scratch/lung_ct/checkpoints/5678b14bb8a563a32f448d19a7d12e6b/last.ckpt
        logger.info(f"Loading nodule segmentation model from {path}")
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        args = checkpoint["args"]
        model = nnUNet(args).load_state_dict(checkpoint["state_dict"]).eval()
        if self.device is not None:
            model.to(self.device)
        logger.debug(f"Nodule segmentation model ready on device={self.device}")
        return model
    
    def load_confidence_model(self, path):
        # /data/rbg/scratch/lung_ct/checkpoints/4296b4b6cda063e96d52aabfb0694a04/4296b4b6cda063e96d52aabfb0694a04epoch=9.ckpt
        logger.info(f"Loading confidence model from {path}")
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        args = checkpoint["args"]
        model = nnUNetConfidence(args).load_state_dict(checkpoint["state_dict"]).eval()
        if self.device is not None:
            model.to(self.device)
        logger.debug(f"Confidence model ready on device={self.device}")
        return model
    
    def register_exams(self, serie: Serie, tp2nodules: Dict[str, Dict]) -> Dict[str, Dict]:
        volumes = serie.get_volume()
        timepoints = sorted(tp2nodules.keys())
        logger.info(f"Registering {len(timepoints)} timepoint(s) for longitudinal tracking")

        for i in range(len(timepoints) - 1):
            past_tp = timepoints[i]       # fixed (earlier exam)
            current_tp = timepoints[i + 1]  # moving (later exam)
            logger.debug(f"Registering moving timepoint {current_tp} to fixed timepoint {past_tp}")

            past_meta = serie._meta[past_tp]
            current_meta = serie._meta[current_tp]

            # read first two DICOM headers for image geometry (no pixel data)
            past_dcm0 = pydicom.dcmread(past_meta.paths[0], stop_before_pixels=True)
            past_dcm1 = pydicom.dcmread(past_meta.paths[1], stop_before_pixels=True)
            current_dcm0 = pydicom.dcmread(current_meta.paths[0], stop_before_pixels=True)
            current_dcm1 = pydicom.dcmread(current_meta.paths[1], stop_before_pixels=True)

            def _ants_from_dicom_geometry(vol_hwz, dcm0, dcm1):
                """Build an ANTs image from a (H, W, D) numpy array with DICOM geometry."""
                iop = np.array(dcm0.ImageOrientationPatient, dtype=float)
                row_dir, col_dir = iop[:3], iop[3:]
                slice_dir = np.cross(row_dir, col_dir)
                ps = list(map(float, dcm0.PixelSpacing))
                slice_spacing = abs(
                    float(dcm1.ImagePositionPatient[2]) - float(dcm0.ImagePositionPatient[2])
                )
                origin = [float(x) for x in dcm0.ImagePositionPatient]
                spacing = [ps[0], ps[1], slice_spacing]
                direction = np.stack([row_dir, col_dir, slice_dir], axis=1)
                return ants.from_numpy(
                    vol_hwz.astype(np.float32),
                    origin=origin,
                    spacing=spacing,
                    direction=direction,
                )

            # build ANTs images from the already-loaded CT volumes
            past_vol = volumes[past_tp].segmentation_volume.squeeze(0).permute(1, 2, 0).numpy()
            current_vol = volumes[current_tp].segmentation_volume.squeeze(0).permute(1, 2, 0).numpy()

            past_ants = _ants_from_dicom_geometry(past_vol, past_dcm0, past_dcm1)
            current_ants = _ants_from_dicom_geometry(current_vol, current_dcm0, current_dcm1)

            # normalize and downsample for speed
            past_ants = ants.iMath(past_ants, "Normalize").astype("float32")
            current_ants = ants.iMath(current_ants, "Normalize").astype("float32")
            past_ants = resize_xy_fast(past_ants, downsample_factor=4)
            current_ants = resize_xy_fast(current_ants, downsample_factor=4)

            # register: fixed=past (earlier), moving=current (later)
            rigid = ants.registration(past_ants, current_ants, type_of_transform="Rigid")

            # point mapping is opposite to image mapping, so invert the forward transform
            reg_transform = ants.read_transform(rigid["fwdtransforms"][0])
            reg_transform = reg_transform.invert()

            # apply transform to each nodule center in the current timepoint
            for _, nodule_data in tp2nodules[current_tp]:
                # ANTs apply_to_point expects [x, y, z] (LPS) order;
                # our center is stored as (y, x, z), so swap axes 0 and 1
                center_ijk = [
                    nodule_data["center"][1],
                    nodule_data["center"][0],
                    nodule_data["center"][2],
                ]
                center_physical = transform_index_to_physicalpoint(
                    (current_dcm0, current_dcm1), center_ijk
                )
                center_past_physical = reg_transform.apply_to_point(center_physical)
                center_past_ijk = transform_physicalpoint_to_index(
                    (past_dcm0, past_dcm1), center_past_physical
                )
                x, y, z = center_past_ijk.tolist()
                nodule_data["centers_in_past_exam_ijk_space"] = (y, x, z)

            # delete temporary transform files written by ANTs
            for transform_file in rigid["fwdtransforms"] + rigid["invtransforms"]:
                if os.path.exists(transform_file):
                    os.remove(transform_file)

        return tp2nodules
    
    @torch.inference_mode()
    def _preprocess(self, serie: Serie) -> Dict[str, torch.Tensor]:
        version2_inputs = serie.get_volume()  # Dict timepoint -> InputV2(segmentation_volume, rve_volume, lungmask_volume)
        logger.debug(f"Preprocessing serie with {len(version2_inputs)} timepoint(s)")
        x = []
        nodule_x = []
        nodule_confidence = []
        nodule_ids_tracked = []
        nodule_tp_id = []
        nodule_volumes = []
        old_nodule_ids = []
        has_prior = []
        nodule_batch_id = []

        tp_data = {}  # timepoint -> {sparse_seg, nodule_ids, nodule_volumes, nodule_confidence}

        for timepoint in sorted(version2_inputs.keys()):
            lungmask_volume = version2_inputs[timepoint].lungmask_volume
            segmentation_volume = version2_inputs[timepoint].segmentation_volume
            logger.debug(f"Processing timepoint {timepoint}")

            # step 1: compute lung mask
            lung_mask = self.lung_mask_model.apply(lungmask_volume)

            # step 2: compute nodule segmentation
            seg_out = self.segmentation_model.predict(segmentation_volume)
            seg_out = F.softmax(seg_out, 1)
            nodule_seg = 1 * (seg_out[:, -1] > 0.5) # (D, H, W) nodule probability map

            # step 3: get connected components and sparsify
            if isinstance(lung_mask, np.ndarray):
                lung_mask_t = torch.tensor(lung_mask > 0, dtype=torch.float32)
            else:
                lung_mask_t = (lung_mask > 0).float()

            lung_mask_t = F.interpolate(lung_mask_t, size=(1024, 1024), mode="nearest")
            lung_mask_t = lung_mask_t.squeeze(1)
            lung_mask_t = (lung_mask_t > 0).float()

            combined_seg = (nodule_seg * lung_mask_t).float()
            isegmentation, nnodules = cc3d.connected_components(combined_seg.numpy(), return_N=True)
            isegmentation = torch.from_numpy(isegmentation.astype(np.float32))
            sparse_seg = isegmentation.to_sparse()

            meta = serie._meta[timepoint]
            volume_spacing = meta.voxel_spacing.prod().item()
            tp_nodule_vols = torch.bincount(sparse_seg.values().int())[1:] * volume_spacing / 4
            tp_nodule_ids = torch.arange(1, nnodules + 1)

            # filter components by volume < 10mm^3
            valid_mask = tp_nodule_vols >= 10
            valid_ids = tp_nodule_ids[valid_mask]
            valid_vols = tp_nodule_vols[valid_mask]
            logger.debug(
                f"Timepoint {timepoint}: {nnodules} connected component(s), {len(valid_ids)} retained after volume filtering"
            )

            seg_vals = sparse_seg.values()
            seg_idxs = sparse_seg.indices()
            keep_mask = torch.isin(seg_vals, valid_ids)
            sparse_seg = torch.sparse_coo_tensor(
                seg_idxs[:, keep_mask], seg_vals[keep_mask], sparse_seg.shape
            ).coalesce()

            # step 4: create input for confidence model
            # NOTE: reference_files/generate_predictions_for_confidence_model.py
            confidence_input = serie.prepare_for_confidence_model(sparse_seg, segmentation_volume, combined_seg)

            # step 5: compute confidence scores
            confidence_out = self.confidence_model(confidence_input)
            tp_confidence = confidence_out["logit"].sigmoid()  # (N_nodules,)

            # step 6: create input for malignancy model
            # NOTE: generate_luna25_patches.py
            # NOTE: nlst_luna25 process_item() in reference_files/luna25.py
            malignancy_input = serie.prepare_for_malignancy_model(sparse_seg, segmentation_volume)

            x.append(segmentation_volume)
            nodule_x.append(malignancy_input)

            tp_data[timepoint] = {
                "sparse_seg": sparse_seg,
                "nodule_ids": valid_ids,
                "nodule_volumes": valid_vols,
                "nodule_confidence": tp_confidence,
            }

        # Build per-timepoint nodule metadata for tracking
        tp2nodules = {}
        for tp, tp_info in tp_data.items():
            sparse_seg = tp_info["sparse_seg"]
            nodule_list = []
            for nid, vol in zip(tp_info["nodule_ids"], tp_info["nodule_volumes"]):
                mask = sparse_seg.values() == nid
                ys, xs, zs = sparse_seg.indices()[:, mask]
                ymin, ymax = ys.min().item() // 2, ys.max().item() // 2
                xmin, xmax = xs.min().item() // 2, xs.max().item() // 2
                zmin, zmax = zs.min().item(), zs.max().item()
                nodule_list.append((nid.item(), {
                    "volume": vol.item(),
                    "nodid_in_segmentation": nid.item(),
                    "coords": (ymin, ymax, xmin, xmax, zmin, zmax),
                    "center": ((ymin + ymax) // 2, (xmin + xmax) // 2, (zmin + zmax) // 2),
                    "screen_detected": True,
                }))
            tp2nodules[tp] = nodule_list

        # if multiple timepoints
        if len(tp2nodules) > 1:
            # step 7: register to first timepoint and transform all volumes to previous timepoint space
            tp2nodules = self.register_exams(serie, tp2nodules)

            # step 8: track nodules across timepoints using distance and assign nodule IDs
            tracked_nodules = link_nodules_by_center_distance(tp2nodules)
            logger.info(f"Tracked {len(tracked_nodules)} longitudinal nodule trajectory(ies)")
        else:
            # single timepoint: assign sequential track IDs
            tp = list(tp2nodules.keys())[0]
            tracked_nodules = {
                i + 1: {tp: meta}
                for i, (_, meta) in enumerate(tp2nodules[tp])
            }
            logger.info(f"Single timepoint detected; assigned {len(tracked_nodules)} track ID(s)")

        # assemble final per-nodule lists
        for track_id, track in tracked_nodules.items():
            tps_in_track = sorted(track.keys())
            first_tp = tps_in_track[0]
            for tp in tps_in_track:
                node_data = track[tp]
                tp_info = tp_data[tp]
                nid = node_data["nodid_in_segmentation"]

                nid_matches = (tp_info["nodule_ids"] == nid).nonzero(as_tuple=True)[0]
                conf = tp_info["nodule_confidence"][nid_matches[0]] if len(nid_matches) > 0 else torch.tensor(0.0)

                nodule_confidence.append(conf)
                nodule_ids_tracked.append(track_id)
                nodule_tp_id.append(tp)
                nodule_volumes.append(node_data["volume"])
                has_prior.append(tp != first_tp)
                old_nodule_ids.append(
                    track.get(first_tp, {}).get("nodid_in_segmentation") if tp != first_tp else nid
                )
                nodule_batch_id.append(track_id)

        return {
            "x": x,
            "nodule_x": nodule_x,
            "nodule_confidence": nodule_confidence,
            "nodule_ids": nodule_ids_tracked,
            "nodule_tps": nodule_tp_id,
            "nvolumes": nodule_volumes,
            "old_nodule_ids": old_nodule_ids,
            "has_prior": has_prior,
            "nodule_batch_id": nodule_batch_id,
        }
    
    @torch.inference_mode()
    def _predict(
        self,
        model: SybilNet,
        series: Union[Serie, List[Serie]],
    ) -> Prediction:
        """Run predictions over the given serie(s).

        Parameters
        ----------
        model: SybilNet
            Instance of SybilNet
        series : Union[Serie, Iterable[Serie]]
            One or multiple series to run predictions for.
        
        Returns
        -------
        Prediction
            Output prediction as risk scores.

        """
        if isinstance(series, Serie):
            series = [series]
        elif not isinstance(series, list):
            raise ValueError("Expected either a Serie object or list of Serie objects.")

        scores: List[List[float]] = []
        logger.debug(f"Running Sybil2 prediction for {len(series)} serie(s)")
        
        for serie in series:
            if not isinstance(serie, Serie):
                raise ValueError("Expected a list of Serie objects.")

            version2_inputs = self._preprocess(serie)
            logger.debug(
                f"Prepared model inputs with {len(version2_inputs['x'])} timepoint volume(s) and {len(version2_inputs['nodule_ids'])} tracked nodule entry(ies)"
            )

            if self.device is not None:
                for key in version2_inputs:
                    if isinstance(version2_inputs[key], torch.Tensor):
                        version2_inputs[key] = version2_inputs[key].to(self.device)
            
            version2_inputs["anatomy"] = ["chest_ct"] * len(version2_inputs['x'])
            out = model(version2_inputs)
            score = out["logit"].sigmoid().squeeze(0).cpu().numpy()
            scores.append(score.tolist())
            logger.debug("Computed Sybil2 risk score for one serie")
            

        return Prediction(scores=scores)

    def predict(
        self, series: Union[Serie, List[Serie]], return_attentions: bool = False, threads=0,
    ) -> Prediction:
        """Run predictions over the given serie(s) and ensemble

        Parameters
        ----------
        series : Union[Serie, Iterable[Serie]]
            One or multiple series to run predictions for.
        return_attentions : bool
            If True, returns attention scores for each serie. See README for details.
        threads : int
            Number of CPU threads to use for PyTorch inference.

        Returns
        -------
        Prediction
            Output prediction. See details for :class:`~sybil.model.Prediction`".

        """

        # Set CPU threads available to torch
        num_threads = _torch_set_num_threads(threads)
        logger.debug(f"Using {num_threads} CPU thread(s) for Sybil2 inference")
        logger.debug(f"Using {num_threads} threads for PyTorch inference")

        if self._device_flexible:
            self.device = self._pick_device()
            self.to(self.device)
            logger.info(f"Selected inference device dynamically: {self.device}")
        logger.debug(f"Beginning prediction on device: {self.device}")

        scores = []
        for sybil in self.ensemble:
            pred = self._predict(sybil, series)
            scores.append(pred.scores)

        scores = np.mean(np.array(scores), axis=0)
        calib_scores = self._calibrate(scores).tolist()
        logger.info(f"Completed Sybil2 ensemble inference for {len(scores)} serie(s)")

        return Prediction(scores=calib_scores)

