import os
import zipfile
import requests
import shutil
from m7016h import default_parameters as dp


def download_file(url, home_folder=dp.home_folder, force=False):
    local_filename = os.path.join(home_folder, url.split('/')[-1])
    # Check if file already exists locally
    if os.path.exists(local_filename) and not force:
        print(f"{local_filename} already exists locally.")
        return

    if force:
        print("Force flag is set. Deleting existing file.")
        shutil.rmtree(local_filename, ignore_errors=True)

    # Download the file
    with requests.get(url, stream=True) as response:
        response.raise_for_status()  # Ensure we raise an error for bad responses
        with open(local_filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    print(f"Downloaded {local_filename} from {url}.")


def unzip_file(zip_filepath, extract_to=dp.home_folder, force=False):
    train_dir = os.path.join(extract_to, dp.train_dir)
    val_dir = os.path.join(extract_to, dp.val_dir)
    test_dir = os.path.join(extract_to, dp.test_dir)

    # Check if the extracted directory already exists
    if os.path.isdir(train_dir) and os.path.isdir(val_dir) and os.path.isdir(test_dir) and not force:
        print(f"Directory {extract_to} already exists. Skipping extraction.")
        return

    if force:
        print("Force flag is set. Deleting existing file.")
        shutil.rmtree(train_dir, ignore_errors=True)
        shutil.rmtree(val_dir, ignore_errors=True)
        shutil.rmtree(test_dir, ignore_errors=True)

    # Unzip the file
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_filepath} to {extract_to}.")


def prepare_dataset(source_url, local_root_path=dp.home_folder, force_download=False, force_unzip=False):
    if not os.path.exists(local_root_path):
        os.makedirs(local_root_path)

    download_file(source_url, home_folder=local_root_path, force=force_download)
    unzip_file(os.path.join(local_root_path, source_url.split('/')[-1]), extract_to=local_root_path, force=force_unzip)
