import os
from urllib.request import urlopen


def download_file(url, filepath):
    response = urlopen(url)

    target_dir = os.path.dirname(filepath)
    if target_dir and not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Check if the request was successful
    if response.status == 200:
        with open(filepath, 'wb') as f:
            f.write(response.read())
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

    return filepath

def get_demo_data():
    demo_data_url = "https://www.dropbox.com/scl/fi/covbvo6f547kak4em3cjd/sybil_example.zip?rlkey=7a13nhlc9uwga9x7pmtk1cf1c&st=dqi0cf9k&dl=1"

    zip_file_name = "sybil_example.zip"
    cache_dir = os.path.expanduser("~/.sybil")
    zip_file_path = os.path.join(cache_dir, zip_file_name)
    os.makedirs(cache_dir, exist_ok=True)
    if not os.path.exists(zip_file_path):
        print(f"Downloading demo data to {zip_file_path}")
        download_file(demo_data_url, zip_file_path)

    demo_data_dir = os.path.join(cache_dir, "sybil_example")
    image_data_dir = os.path.join(demo_data_dir, "sybil_demo_data")
    if not os.path.exists(demo_data_dir):
        print(f"Extracting demo data to {demo_data_dir}")
        import zipfile
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(demo_data_dir)

    dicom_files = os.listdir(image_data_dir)
    dicom_files = [os.path.join(image_data_dir, x) for x in dicom_files]
    return dicom_files
