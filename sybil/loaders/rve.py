"""
Self-contained NIfTI → RVE processor.

Replicates the full ct_processor + video_exporter pipeline without requiring
the vision-engine config/pipeline infrastructure. Takes a NIfTI file, processes
it, exports it to disk in video_hevc format, and returns the output path so it
can be loaded later with rve.load_sample.

Usage
-----
    import rve
    from nifti_to_rve import NiftiToRVE

    proc = NiftiToRVE()
    out_path = proc("/path/to/scan.nii.gz", output_dir="/out", accession="ACC001")
    volume = rve.load_sample(str(out_path))
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F


class NiftiToRVE:
    """
    Self-contained processor: NIfTI file → RVE object (numpy volume or tarball).

    Default parameters match configs/ct_chest.yaml + configs/exporters/video_hevc.yaml.
    """

    def __init__(
        self,
        # --- processing (ct_chest.yaml) ---
        target_spacing: Tuple[float, float, float] = (1.25, 1.25, 1.25),
        crop_pad_size: Tuple[int, int] = (256, 256),
        max_slices: int = 256,
        # --- export (video_hevc.yaml) ---
        hu_min: int = -1024,
        hu_max: int = 3071,
        crf: int = 6,
        gop_size: int = 128,
        preset: str = "ultrafast",
        video_ext: str = ".mp4",
    ):
        self.target_spacing = list(target_spacing)
        self.crop_pad_size = list(crop_pad_size)
        self.max_slices = max_slices
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.crf = crf
        self.gop_size = gop_size
        self.preset = preset
        self.video_ext = video_ext if video_ext.startswith(".") else f".{video_ext}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        sitk_image: np.ndarray,
        output_dir: str,
        accession: str = "unknown",
        series_number: int = 1,
        extra_metadata: Optional[dict] = None,
    ) -> Path:
        """
        Process a NIfTI file and export it as a video_hevc RVE folder.

        Writes output_dir/<accession>.<series_number>.0/
          volume<ext>   — HEVC-encoded video
          metadata.json — processing + HU-mapping info

        The returned path can be passed directly to rve.load_sample().

        Parameters
        ----------
        nifti_path : str
            Path to input .nii or .nii.gz file.
        output_dir : str
            Directory under which the output folder is created.
        accession : str
            Identifier used for the output folder name.
        series_number : int
            Series number used for the output folder name.
        extra_metadata : dict, optional
            Additional keys merged into metadata.json at the top level.

        Returns
        -------
        Path
            Path to the output folder (pass to rve.load_sample).
        """
        sitk_image = self._resample(sitk_image)
        volume = sitk.GetArrayFromImage(sitk_image)  # (Z, Y, X)
        volume = self._ensure_int16(volume)
        volume = self._select_slices(volume)
        volume = self._center_crop_pad(volume)

        folder = Path(output_dir) / f"{accession}.{series_number}.0"
        folder.mkdir(parents=True, exist_ok=True)

        video_bytes = self._encode_video(volume)
        (folder / f"volume{self.video_ext}").write_bytes(video_bytes)

        metadata_str = self._build_metadata(volume, accession, series_number, extra_metadata)
        (folder / "metadata.json").write_text(metadata_str, encoding="utf-8")

        return folder

    # ------------------------------------------------------------------
    # Processing steps
    # ------------------------------------------------------------------

    def _resample(self, image: sitk.Image) -> sitk.Image:
        """Resample to target_spacing using linear interpolation."""
        original_spacing = image.GetSpacing()  # (x, y, z)
        original_size = image.GetSize()         # (x, y, z)

        new_size = [
            int(round(original_size[i] * original_spacing[i] / self.target_spacing[i]))
            for i in range(3)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(self.target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0)
        resampler.SetInterpolator(sitk.sitkLinear)
        return resampler.Execute(image)

    def _ensure_int16(self, volume: np.ndarray) -> np.ndarray:
        """Cast to int16 (safe for Hounsfield Units)."""
        if volume.min() >= -32768 and volume.max() <= 32767:
            return volume.astype(np.int16)
        return volume  # out-of-range: leave as-is, caller's problem

    def _select_slices(self, volume: np.ndarray) -> np.ndarray:
        """Select middle max_slices slices along the Z axis."""
        z = volume.shape[0]
        if z <= self.max_slices:
            return volume
        start = (z - self.max_slices) // 2
        return volume[start : start + self.max_slices]

    def _center_crop_pad(self, volume: np.ndarray) -> np.ndarray:
        """Center crop or pad the H and W dimensions to crop_pad_size."""
        original_dtype = volume.dtype
        pad_value = float(volume.min())
        t = torch.from_numpy(volume).float()  # (Z, H, W)

        _, cur_h, cur_w = t.shape
        tgt_h, tgt_w = self.crop_pad_size

        # Height
        dh = tgt_h - cur_h
        if dh < 0:
            h0 = (-dh) // 2
            t = t[:, h0 : h0 + tgt_h, :]
        elif dh > 0:
            p = (dh // 2, dh - dh // 2)
            t = F.pad(t, (0, 0, *p), mode="constant", value=pad_value)

        # Width
        dw = tgt_w - cur_w
        if dw < 0:
            w0 = (-dw) // 2
            t = t[:, :, w0 : w0 + tgt_w]
        elif dw > 0:
            p = (dw // 2, dw - dw // 2)
            t = F.pad(t, (*p, 0, 0), mode="constant", value=pad_value)

        return t.numpy().astype(original_dtype)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _encode_video(self, volume: np.ndarray) -> bytes:
        """
        Map HU values to uint16, pipe raw frames through ffmpeg libx265,
        and return the encoded video bytes.
        """
        hu_range = self.hu_max - self.hu_min
        clipped = np.clip(volume, self.hu_min, self.hu_max)
        v16 = ((clipped - self.hu_min) * 65535.0 / hu_range).astype(np.uint16)

        num_slices, height, width = v16.shape
        x265_params = f"keyint={self.gop_size}:min-keyint={self.gop_size}"

        with tempfile.NamedTemporaryFile(suffix=self.video_ext, delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-video_size", f"{width}x{height}",
                "-pixel_format", "gray16le",
                "-framerate", "25",
                "-i", "-",
                "-c:v", "libx265",
                "-preset", self.preset,
                "-crf", str(self.crf),
                "-pix_fmt", "yuv420p10le",
                "-x265-params", x265_params,
                "-color_range", "pc",
                "-colorspace", "bt709",
                "-color_primaries", "bt709",
                "-color_trc", "bt709",
                tmp_path,
            ]

            # Build all frame bytes in one shot
            frames = b"".join(
                np.ascontiguousarray(v16[i]).tobytes() for i in range(num_slices)
            )

            result = subprocess.run(
                cmd, input=frames, capture_output=True, timeout=300
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed (code {result.returncode}):\n"
                    + result.stderr.decode(errors="replace")
                )

            return Path(tmp_path).read_bytes()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _build_metadata(
        self,
        volume: np.ndarray,
        accession: str,
        series_number: int,
        extra: Optional[dict],
    ) -> str:
        meta = {
            "series_info": {
                "accession": accession,
                "series_number": series_number,
                "modality": "CT",
            },
            "processing_metadata": {
                "modality": "CT",
                "target_spacing": self.target_spacing,
                "crop_pad_target": self.crop_pad_size,
                "max_slices": self.max_slices,
                "final_shape": list(volume.shape),
                "value_range": "raw_hounsfield_units",
            },
            "export_info": {
                "format": "video",
                "encoder": "libx265",
                "codec": "HEVC",
                "crf": self.crf,
                "gop_size": self.gop_size,
                "preset": self.preset,
                "hu_mapping": {
                    "min": self.hu_min,
                    "max": self.hu_max,
                    "output_range": [0, 65535],
                },
            },
        }
        if extra:
            meta.update(extra)
        return json.dumps(meta, indent=2)

