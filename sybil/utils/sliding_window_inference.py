from collections.abc import Callable, Iterable, Sequence
from typing import Any
import torch
import torch.nn.functional as F

from monai.data.meta_tensor import MetaTensor
from monai.data.utils import (
    compute_importance_map,
    dense_patch_slices,
    get_valid_patch_size,
)
from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    convert_to_dst_type,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    look_up_option,
)
from monai.inferers.utils import (
    _create_buffered_slices,
    _compute_coords,
    _get_scan_interval,
    _flatten_struct,
    _pack_struct,
)
from sybil.utils.augmentations import PatchAugmentations
from tqdm import tqdm

_nearest_mode = "nearest-exact"


def sliding_window_inference_custom(
    inputs: torch.Tensor | MetaTensor,
    roi_size: Sequence[int] | int,
    sw_batch_size: int,
    predictor: Callable[
        ..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]
    ],
    overlap: Sequence[float] | float = 0.25,
    mode: BlendMode | str = BlendMode.CONSTANT,
    sigma_scale: Sequence[float] | float = 0.125,
    padding_mode: PytorchPadMode | str = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: torch.device | str | None = None,
    device: torch.device | str | None = None,
    progress: bool = False,
    roi_weight_map: torch.Tensor | None = None,
    process_fn: Callable | None = None,
    buffer_steps: int | None = None,
    buffer_dim: int = -1,
    with_coord: bool = False,
    augmentations: PatchAugmentations = None,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
    """
    Sliding window inference on `inputs` with `predictor`.

    The outputs of `predictor` could be a tensor, a tuple, or a dictionary of tensors.
    Each output in the tuple or dict value is allowed to have different resolutions with respect to the input.
    e.g., the input patch spatial size is [128,128,128], the output (a tuple of two patches) patch sizes
    could be ([128,64,256], [64,32,128]).
    In this case, the parameter `overlap` and `roi_size` need to be carefully chosen to ensure the output ROI is still
    an integer. If the predictor's input and output spatial sizes are not equal, we recommend choosing the parameters
    so that `overlap*roi_size*output_size/input_size` is an integer (for each spatial dimension).

    When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
    To maintain the same spatial sizes, the output image will be cropped to the original input size.

    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size: the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor ``patch_data`` in shape NCHW[D],
            The outputs of the function call ``predictor(patch_data)`` should be a tensor, a tuple, or a dictionary
            with Tensor values. Each output in the tuple or dict value should have the same batch_size, i.e. NM'H'W'[D'];
            where H'W'[D'] represents the output patch's spatial size, M is the number of output channels,
            N is `sw_batch_size`, e.g., the input shape is (7, 1, 128,128,128),
            the output could be a tuple of two tensors, with shapes: ((7, 5, 128, 64, 256), (7, 4, 64, 32, 128)).
            In this case, the parameter `overlap` and `roi_size` need to be carefully chosen
            to ensure the scaled output ROI sizes are still integers.
            If the `predictor`'s input and output spatial sizes are different,
            we recommend choosing the parameters so that ``overlap*roi_size*zoom_scale`` is an integer for each dimension.
        overlap: Amount of overlap between scans along each spatial dimension, defaults to ``0.25``.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
            Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
            When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
            spatial dimensions.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode for ``inputs``, when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        cval: fill value for 'constant' padding mode. Default: 0
        sw_device: device for the window data.
            By default the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.
        progress: whether to print a `tqdm` progress bar.
        roi_weight_map: pre-computed (non-negative) weight map for each ROI.
            If not given, and ``mode`` is not `constant`, this map will be computed on the fly.
        process_fn: process inference output and adjust the importance map per window
        buffer_steps: the number of sliding window iterations along the ``buffer_dim``
            to be buffered on ``sw_device`` before writing to ``device``.
            (Typically, ``sw_device`` is ``cuda`` and ``device`` is ``cpu``.)
            default is None, no buffering. For the buffer dim, when spatial size is divisible by buffer_steps*roi_size,
            (i.e. no overlapping among the buffers) non_blocking copy may be automatically enabled for efficiency.
        buffer_dim: the spatial dimension along which the buffers are created.
            0 indicates the first spatial dimension. Default is -1, the last spatial dimension.
        with_coord: whether to pass the window coordinates to ``predictor``. Default is False.
            If True, the signature of ``predictor`` should be ``predictor(patch_data, patch_coord, ...)``.
        args: optional args to be passed to ``predictor``.
        kwargs: optional keyword args to be passed to ``predictor``.

    Note:
        - input must be channel-first and have a batch dim, supports N-D sliding window.

    """
    buffered = buffer_steps is not None and buffer_steps > 0
    num_spatial_dims = len(inputs.shape) - 2
    if buffered:
        if buffer_dim < -num_spatial_dims or buffer_dim > num_spatial_dims:
            raise ValueError(
                f"buffer_dim must be in [{-num_spatial_dims}, {num_spatial_dims}], got {buffer_dim}."
            )
        if buffer_dim < 0:
            buffer_dim += num_spatial_dims
    overlap = ensure_tuple_rep(overlap, num_spatial_dims)
    for o in overlap:
        if o < 0 or o >= 1:
            raise ValueError(f"overlap must be >= 0 and < 1, got {overlap}.")
    compute_dtype = inputs.dtype

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    batch_size, _, *image_size_ = inputs.shape
    device = device or inputs.device
    sw_device = sw_device or inputs.device

    condition = kwargs.pop("condition", None)

    temp_meta = None
    if isinstance(inputs, MetaTensor):
        temp_meta = MetaTensor([]).copy_meta_from(inputs, copy_attr=False)
    inputs = convert_data_type(inputs, torch.Tensor, wrap_sequence=True)[0]
    roi_size = fall_back_tuple(roi_size, image_size_)

    # in case that image size is smaller than roi size
    image_size = tuple(
        max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims)
    )
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    if any(pad_size):
        inputs = F.pad(
            inputs,
            pad=pad_size,
            mode=look_up_option(padding_mode, PytorchPadMode),
            value=cval,
        )
        if condition is not None:
            condition = F.pad(
                condition,
                pad=pad_size,
                mode=look_up_option(padding_mode, PytorchPadMode),
                value=cval,
            )

    # Store all slices
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(
        image_size, roi_size, scan_interval, return_slice=not buffered
    )

    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows
    windows_range: Iterable
    if not buffered:
        non_blocking = False
        windows_range = range(0, total_slices, sw_batch_size)
    else:
        slices, n_per_batch, b_slices, windows_range = _create_buffered_slices(
            slices, batch_size, sw_batch_size, buffer_dim, buffer_steps
        )
        non_blocking, _ss = torch.cuda.is_available(), -1
        for x in b_slices[:n_per_batch]:
            if x[1] < _ss:  # detect overlapping slices
                non_blocking = False
                break
            _ss = x[2]

    # Create window-level importance map
    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    if valid_patch_size == roi_size and (roi_weight_map is not None):
        importance_map_ = roi_weight_map
    else:
        try:
            valid_p_size = ensure_tuple(valid_patch_size)
            importance_map_ = compute_importance_map(
                valid_p_size,
                mode=mode,
                sigma_scale=sigma_scale,
                device=sw_device,
                dtype=compute_dtype,
            )
            if len(importance_map_.shape) == num_spatial_dims and not process_fn:
                importance_map_ = importance_map_[
                    None, None
                ]  # adds batch, channel dimensions
        except Exception as e:
            raise RuntimeError(
                f"patch size {valid_p_size}, mode={mode}, sigma_scale={sigma_scale}, device={device}\n"
                "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
            ) from e
    importance_map_ = convert_data_type(
        importance_map_, torch.Tensor, device=sw_device, dtype=compute_dtype
    )[0]

    # stores output and count map
    output_image_list, count_map_list, sw_device_buffer, b_s, b_i = [], [], [], 0, 0  # type: ignore
    # for each patch
    for slice_g in tqdm(windows_range) if progress else windows_range:
        slice_range = range(
            slice_g,
            min(
                slice_g + sw_batch_size, b_slices[b_s][0] if buffered else total_slices
            ),
        )
        unravel_slice = [
            [slice(idx // num_win, idx // num_win + 1), slice(None)]
            + list(slices[idx % num_win])
            for idx in slice_range
        ]
        if sw_batch_size > 1:
            win_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(
                sw_device
            )
            if condition is not None:
                win_condition = torch.cat(
                    [condition[win_slice] for win_slice in unravel_slice]
                ).to(sw_device)
                kwargs["condition"] = win_condition
        else:
            win_data = inputs[unravel_slice[0]].to(sw_device)
            if condition is not None:
                win_condition = condition[unravel_slice[0]].to(sw_device)
                kwargs["condition"] = win_condition

        # ------------------------------------------------------------------------>
        win_data = augmentations(win_data)
        # win_data = augmentations({"image": win_data[0, 0].permute(1, 2, 0)})["image"]
        # win_data = win_data.permute(0, 3, 1, 2)[None]
        # ------------------------------------------------------------------------>

        if with_coord:
            seg_prob_out = predictor(win_data, unravel_slice, *args, **kwargs)
        else:
            seg_prob_out = predictor(win_data, *args, **kwargs)
        # convert seg_prob_out to tuple seg_tuple, this does not allocate new memory.
        dict_keys, seg_tuple = _flatten_struct(seg_prob_out)
        if process_fn:
            seg_tuple, w_t = process_fn(seg_tuple, win_data, importance_map_)
        else:
            w_t = importance_map_
        if len(w_t.shape) == num_spatial_dims:
            w_t = w_t[None, None]
        w_t = w_t.to(dtype=compute_dtype, device=sw_device)
        if buffered:
            c_start, c_end = b_slices[b_s][1:]
            if not sw_device_buffer:
                k = seg_tuple[0].shape[1]  # len(seg_tuple) > 1 is currently ignored
                sp_size = list(image_size)
                sp_size[buffer_dim] = c_end - c_start
                sw_device_buffer = [
                    torch.zeros(
                        size=[1, k, *sp_size], dtype=compute_dtype, device=sw_device
                    )
                ]
            for p, s in zip(seg_tuple[0], unravel_slice):
                offset = s[buffer_dim + 2].start - c_start
                s[buffer_dim + 2] = slice(offset, offset + roi_size[buffer_dim])
                s[0] = slice(0, 1)
                sw_device_buffer[0][s] += p * w_t
            b_i += len(unravel_slice)
            if b_i < b_slices[b_s][0]:
                continue
        else:
            sw_device_buffer = list(seg_tuple)

        for ss in range(len(sw_device_buffer)):
            b_shape = sw_device_buffer[ss].shape
            seg_chns, seg_shape = b_shape[1], b_shape[2:]
            z_scale = None
            if not buffered and seg_shape != roi_size:
                z_scale = [
                    out_w_i / float(in_w_i)
                    for out_w_i, in_w_i in zip(seg_shape, roi_size)
                ]
                w_t = F.interpolate(w_t, seg_shape, mode=_nearest_mode)
            if len(output_image_list) <= ss:
                output_shape = [batch_size, seg_chns]
                output_shape += (
                    [int(_i * _z) for _i, _z in zip(image_size, z_scale)]
                    if z_scale
                    else list(image_size)
                )
                # allocate memory to store the full output and the count for overlapping parts
                new_tensor: Callable = torch.empty if non_blocking else torch.zeros  # type: ignore
                output_image_list.append(
                    new_tensor(output_shape, dtype=compute_dtype, device=device)
                )
                count_map_list.append(
                    torch.zeros(
                        [1, 1] + output_shape[2:], dtype=compute_dtype, device=device
                    )
                )
                w_t_ = w_t.to(device)
                for __s in slices:
                    if z_scale is not None:
                        __s = tuple(
                            slice(int(_si.start * z_s), int(_si.stop * z_s))
                            for _si, z_s in zip(__s, z_scale)
                        )
                    count_map_list[-1][(slice(None), slice(None), *__s)] += w_t_
            if buffered:
                o_slice = [slice(None)] * len(inputs.shape)
                o_slice[buffer_dim + 2] = slice(c_start, c_end)
                img_b = b_s // n_per_batch  # image batch index
                o_slice[0] = slice(img_b, img_b + 1)
                if non_blocking:
                    output_image_list[0][o_slice].copy_(
                        sw_device_buffer[0], non_blocking=non_blocking
                    )
                else:
                    output_image_list[0][o_slice] += sw_device_buffer[0].to(
                        device=device
                    )
            else:
                sw_device_buffer[ss] *= w_t
                sw_device_buffer[ss] = sw_device_buffer[ss].to(device)
                _compute_coords(
                    unravel_slice, z_scale, output_image_list[ss], sw_device_buffer[ss]
                )
        sw_device_buffer = []
        if buffered:
            b_s += 1

    if non_blocking:
        torch.cuda.current_stream().synchronize()

    # account for any overlapping sections
    for ss in range(len(output_image_list)):
        output_image_list[ss] /= count_map_list.pop(0)

    # remove padding if image_size smaller than roi_size
    if any(pad_size):
        kwargs.update({"pad_size": pad_size})
        for ss, output_i in enumerate(output_image_list):
            zoom_scale = [
                _shape_d / _roi_size_d
                for _shape_d, _roi_size_d in zip(output_i.shape[2:], roi_size)
            ]
            final_slicing: list[slice] = []
            for sp in range(num_spatial_dims):
                si = num_spatial_dims - sp - 1
                slice_dim = slice(
                    int(round(pad_size[sp * 2] * zoom_scale[si])),
                    int(round((pad_size[sp * 2] + image_size_[si]) * zoom_scale[si])),
                )
                final_slicing.insert(0, slice_dim)
            output_image_list[ss] = output_i[(slice(None), slice(None), *final_slicing)]

    final_output = _pack_struct(output_image_list, dict_keys)
    if temp_meta is not None:
        final_output = convert_to_dst_type(final_output, temp_meta, device=device)[0]
    else:
        final_output = convert_to_dst_type(final_output, inputs, device=device)[0]

    return final_output  # type: ignore
