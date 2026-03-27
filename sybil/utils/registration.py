import os

# CRITICAL: Must be set BEFORE importing ants or itk
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
os.environ["NSLOTS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import ants
import shutil
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np


FLIPPER = np.eye(3)
FLIPPER[0, 0] = -1
FLIPPER[1, 1] = -1


def dicom_to_nifti(dcm_dir, save_img_file):
    """Convert DICOM series to NIfTI format.
    Args:
        dcm_dir (str): Directory containing DICOM files.
        save_img_file (str): Path to save the converted NIfTI file.
    Returns:
        None
    """
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dcm_dir)
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3D = series_reader.Execute()
    sitk.WriteImage(image3D, save_img_file, True)


def correct_affine(ants_image):
    origin = list(ants_image.origin)
    origin[0] = -origin[0]
    origin[1] = -origin[1]
    ants_image.set_origin(tuple(origin))
    ants_image.set_direction(ants_image.direction @ FLIPPER)
    return ants_image


def resize_xy_fast(img, downsample_factor=4):
    """
    Subsamples the image using numpy slicing.
    Instant, but requires manual metadata patching.
    """
    # 1. Slice the numpy array (Z, Y, X)
    #    Note: ANTsPy stores data as (X, Y, Z) internally,
    #    but .numpy() returns (Z, Y, X) or (X, Y, Z) depending on settings?
    #    Safer to use ANTs indexing directly.

    # ANTsPy objects support direct slicing: img[x_start:x_end:step, ...]
    # We slice X and Y by 'downsample_factor', keep Z intact.
    # ANTs slicing syntax is [x, y, z]
    data = img.numpy()
    data_small = data[::downsample_factor, ::downsample_factor]

    # 2. ANTsPy slicing *usually* updates spacing automatically,
    #    but it often messes up the Origin shift.
    #    Let's enforce the correct spacing manually to be safe.

    old_spacing = img.spacing
    new_spacing = list(old_spacing)
    new_spacing[0] *= downsample_factor
    new_spacing[1] *= downsample_factor

    img_small = ants.from_numpy(
        data_small, origin=img.origin, spacing=new_spacing, direction=img.direction
    )

    return img_small


def transform_index_to_physicalpoint(dcms, ijk):
    """Convert voxel indices (i, j, k) to physical LPS coordinates in mm.

    Parameters
    ----------
    dcms : tuple
        (dcm0, dcm1) — pydicom datasets for the first and second DICOM slices.
    ijk : array-like of shape (3,)
        Voxel indices (i=col, j=row, k=slice).

    Returns
    -------
    np.ndarray
        Physical coordinates [x, y, z] in mm (LPS patient coordinate system).
    """
    dcm0, dcm1 = dcms
    image_position = np.array(dcm0.ImagePositionPatient, dtype=float)
    orientation = np.array(dcm0.ImageOrientationPatient, dtype=float)
    pixel_spacing = np.array(dcm0.PixelSpacing, dtype=float)

    row_dir = orientation[0:3]
    col_dir = orientation[3:6]
    slice_dir = np.cross(row_dir, col_dir)
    slice_spacing = np.abs(
        float(dcm1.ImagePositionPatient[2]) - float(dcm0.ImagePositionPatient[2])
    )

    i, j, k = ijk
    physical = (
        image_position
        + row_dir * (i * pixel_spacing[0])
        + col_dir * (j * pixel_spacing[1])
        + slice_dir * (k * slice_spacing)
    )
    return physical


def transform_physicalpoint_to_index(dcms, coord):
    """Convert physical LPS coordinates in mm to voxel indices (i, j, k).

    Parameters
    ----------
    dcms : tuple
        (dcm0, dcm1) — pydicom datasets for the first and second DICOM slices.
    coord : array-like of shape (3,)
        Physical coordinates [x, y, z] in mm (LPS patient coordinate system).

    Returns
    -------
    np.ndarray
        Rounded voxel indices [i, j, k].
    """
    dcm0, dcm1 = dcms
    image_position = np.array(dcm0.ImagePositionPatient, dtype=float)
    orientation = np.array(dcm0.ImageOrientationPatient, dtype=float)
    pixel_spacing = np.array(dcm0.PixelSpacing, dtype=float)

    row_dir = orientation[0:3]
    col_dir = orientation[3:6]
    slice_dir = np.cross(row_dir, col_dir)
    slice_spacing = np.abs(
        float(dcm1.ImagePositionPatient[2]) - float(dcm0.ImagePositionPatient[2])
    )

    d = np.array(coord) - image_position
    i = np.dot(d, row_dir) / pixel_spacing[0]
    j = np.dot(d, col_dir) / pixel_spacing[1]
    k = np.dot(d, slice_dir) / slice_spacing
    return np.round(np.array([i, j, k])).astype(int)


def save_registration_transformation(fixed_path, moving_path, transform_path):
    """
    Register two exams using ANTs.

    Args:
        fixed_path (str): Path to the fixed image.
        moving_path (str): Path to the moving image.
        transform_path (str): Path to save the registration transformation.

    Returns:
        ants.ANTsImage: The registered image.
    """

    try:
        fixed = ants.image_read(fixed_path)
        moving = ants.image_read(moving_path)

        fixed = ants.iMath(fixed, "Normalize").astype("float32")
        moving = ants.iMath(moving, "Normalize").astype("float32")

        # downsample for speed
        fixed = resize_xy_fast(fixed, downsample_factor=4)
        moving = resize_xy_fast(moving, downsample_factor=4)

        rigid = ants.registration(fixed, moving, type_of_transform="Rigid")
        assert rigid["fwdtransforms"][0] == rigid["invtransforms"][0], (
            "Expected symmetric rigid transform"
        )
        shutil.move(
            rigid["fwdtransforms"][0],
            transform_path,
        )

    except Exception as e:
        print(f"Could not register {fixed_path} to {moving_path}. Message: {e}")
        return e
