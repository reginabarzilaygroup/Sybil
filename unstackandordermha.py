import SimpleITK as sitk
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Break a 3D MHA file into individual slice MHA files.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input 3D MHA file')
    parser.add_argument('--output', type=str, required=True, help='Path to the output directory for the sorted individual slice MHA files')
    return parser.parse_args()

args = parse_arguments()

input_mha_file = args.input
output_directory = args.output

def break_mha_into_slices(input_mha_file, output_folder):
    sitk_image = sitk.ReadImage(input_mha_file)
    ct_slices = sitk.GetArrayFromImage(sitk_image)
    spacing = sitk_image.GetSpacing()
    seriesuid, _ = os.path.splitext(os.path.basename(input_mha_file))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, slice_data in enumerate(ct_slices):
        slice_image = sitk.GetImageFromArray(slice_data)
        slice_image.SetOrigin(sitk_image.GetOrigin())
        slice_image.SetMetaData("SeriesUID", seriesuid)

        x_coordinate = slice_image.GetOrigin()[0]
        y_coordinate = slice_image.GetOrigin()[1]

        slice_image.SetMetaData("ImagePositionPatient", f"{x_coordinate} {y_coordinate}")
        
        if isinstance(spacing, tuple) and len(spacing) >= 3:
            slice_thickness = spacing[2]
        else:
            slice_thickness = None
        
        slice_image.SetMetaData("SliceThickness", str(slice_thickness))
        pixel_spacing_str = " ".join(map(str, spacing))
        slice_image.SetMetaData("PixelSpacing", pixel_spacing_str)

        slice_filename = os.path.join(output_folder, f"slice_{i:04d}.mha")
        sitk.WriteImage(slice_image, slice_filename)

break_mha_into_slices(input_mha_file, output_directory)
