import pydicom
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import itk


def pydicom_to_nifti(paths, output_path, return_nifti=False, save_nifti=True):
    """
    Reads DICOMs using pydicom/numpy, but saves using ITK to ensure
    perfect compatibility with ANTsPy.
    """

    # --- 1. Load Data into Numpy ---
    # We assume 'paths' is already sorted by Z-position
    slices = [pydicom.dcmread(p) for p in paths]

    # Create volume (Z, Y, X)
    # Note: ITK python wraps numpy as (Z, Y, X), so this matches perfectly.
    volume = np.stack(
        [
            s.pixel_array.astype(np.float32) * s.RescaleSlope + s.RescaleIntercept
            for s in slices
        ],
        axis=0,
    )

    # --- 2. Convert to ITK Image ---
    # This creates a wrapper, avoiding memory duplication
    itk_image = itk.image_view_from_array(volume)

    # --- 3. Extract Geometry (LPS) ---
    first_ds = slices[0]
    last_ds = slices[-1]

    # Spacing
    # DICOM PixelSpacing is [RowSpacing (Y), ColSpacing (X)]
    spacing_y, spacing_x = first_ds.PixelSpacing

    # Calculate Z-spacing/direction using the full stack extent (handles Gantry Tilt)
    # We do NOT use 'SliceThickness' or cross-products here.
    pos_first = np.array(first_ds.ImagePositionPatient, dtype=float)
    pos_last = np.array(last_ds.ImagePositionPatient, dtype=float)
    n_slices = len(slices)

    # Total vector from first to last slice
    stack_vector = pos_last - pos_first

    # The average step vector between slices
    step_vector = stack_vector / (n_slices - 1)

    # The magnitude of the step is the Z-spacing
    spacing_z = np.linalg.norm(step_vector)

    # Set Spacing (X, Y, Z)
    itk_image.SetSpacing([float(spacing_x), float(spacing_y), float(spacing_z)])

    # Origin (X, Y, Z) - The center of the first voxel
    itk_image.SetOrigin([float(x) for x in pos_first])

    # --- 4. Build Direction Matrix ---
    # ITK Direction is a 3x3 Matrix. Columns are the axis vectors.
    # Columns must be normalized (unit length).

    iop = np.array(first_ds.ImageOrientationPatient, dtype=float)
    row_cosines = iop[:3]  # X-axis orientation
    col_cosines = iop[3:]  # Y-axis orientation

    # Z-axis orientation (Normalized step vector)
    slice_cosines = step_vector / spacing_z

    # Construct the 3x3 matrix (flattened list or numpy array)
    # Matrix = [ X_vec, Y_vec, Z_vec ] (columns)
    # But ITK setDirection expects a flat list or matrix in row-major order?
    # PyITK expects: [[xx, yx, zx], [xy, yy, zy], [xz, yz, zz]]

    direction_matrix = np.eye(3)
    direction_matrix[0, 0] = row_cosines[0]
    direction_matrix[1, 0] = row_cosines[1]
    direction_matrix[2, 0] = row_cosines[2]

    direction_matrix[0, 1] = col_cosines[0]
    direction_matrix[1, 1] = col_cosines[1]
    direction_matrix[2, 1] = col_cosines[2]

    direction_matrix[0, 2] = slice_cosines[0]
    direction_matrix[1, 2] = slice_cosines[1]
    direction_matrix[2, 2] = slice_cosines[2]

    # ITK python usually accepts the numpy matrix directly
    itk_image.SetDirection(direction_matrix)

    # --- 5. Save ---
    # This handles compression (.nii.gz) and RAS conversion automatically
    if save_nifti:
        itk.imwrite(itk_image, output_path)
    if return_nifti:
        return volume, itk_image
    return volume


def sitk_to_nifti(dcm_dir, save_img_file):
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dcm_dir)
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3D = series_reader.Execute()
    sitk.WriteImage(image3D, save_img_file, True)
    return sitk.GetArrayFromImage(image3D).transpose(1, 2, 0)  # x, y, z


def read_with_sitk(dcm_dir, depth_first=False):
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dcm_dir)
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3D = series_reader.Execute()
    if depth_first:
        return sitk.GetArrayFromImage(image3D)
    else:
        return sitk.GetArrayFromImage(image3D).transpose(1, 2, 0)  # x, y, z


def transform_physicalpoint_to_index(dcm0, dcm1, coord):
    image_position = np.array(dcm0.ImagePositionPatient, dtype=float)
    orientation = np.array(dcm0.ImageOrientationPatient, dtype=float)
    pixel_spacing = np.array(dcm0.PixelSpacing, dtype=float)

    row_dir = orientation[0:3]
    col_dir = orientation[3:6]
    slice_dir = np.cross(row_dir, col_dir)  # normal to slice plane

    # --- Compute slice spacing ---
    # More robust than using SliceThickness directly
    slice_spacing = np.abs(
        float(dcm1.ImagePositionPatient[2]) - float(dcm0.ImagePositionPatient[2])
    )

    # --- World coordinate to transform ---
    world_coord = np.array(coord)  # (x, y, z) mm

    # --- Apply transformation ---
    d = world_coord - image_position
    i = np.dot(d, row_dir) / pixel_spacing[0]
    j = np.dot(d, col_dir) / pixel_spacing[1]
    k = np.dot(d, slice_dir) / slice_spacing
    ijk = np.array([i, j, k])
    return np.round(ijk).astype(int)
