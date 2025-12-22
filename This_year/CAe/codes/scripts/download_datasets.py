"""
Helper to download datasets used in assignments.
This script will not automatically fetch proprietary datasets requiring credentials (e.g., COCO with restricted access), but it will provide helpers and instructions.
"""
import os
import argparse
import gdown

DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')

def ensure_data_dirs():
    os.makedirs(DATA_ROOT, exist_ok=True)
    print('Data root:', os.path.abspath(DATA_ROOT))


def download_q1_image_captioning(dest):
    """Download Q1 Image Captioning dataset from Google Drive"""
    q1_dest = os.path.join(dest, 'q1_image_captioning')
    os.makedirs(q1_dest, exist_ok=True)

    # Google Drive file ID from the provided link
    file_id = '1Mh5GO9C9-WcDV2obFNkB8XT-rMv0L197'
    url = f'https://drive.google.com/uc?id={file_id}'

    print(f'Downloading Q1 Image Captioning dataset to: {q1_dest}')
    output_path = os.path.join(q1_dest, 'dataset.zip')

    try:
        gdown.download(url, output_path, quiet=False)
        print(f'Successfully downloaded dataset to: {output_path}')
        print('Please unzip the dataset manually and ensure the structure matches the expected format.')
        print('Expected structure:')
        print('  q1_image_captioning/')
        print('    images/          # Directory containing all images')
        print('    captions.csv     # CSV file with columns: image,caption')
    except Exception as e:
        print(f'Error downloading dataset: {e}')
        print('Please download manually from: https://drive.google.com/file/d/1Mh5GO9C9-WcDV2obFNkB8XT-rMv0L197/view?usp=sharing')


def download_urbansound(dest):
    print('Please download UrbanSound8K from https://urbansounddataset.weebly.com/urbansound8k.html and place it under:', dest)


def download_flickr_subset(dest):
    print('Place the Flickr/COCO subset (images + captions.csv) under', dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--q1', action='store_true', help='Download Q1 Image Captioning dataset')
    parser.add_argument('--urbansound', action='store_true')
    parser.add_argument('--flickr', action='store_true')
    args = parser.parse_args()
    ensure_data_dirs()
    if args.q1:
        download_q1_image_captioning(DATA_ROOT)
    if args.urbansound:
        download_urbansound(DATA_ROOT)
    if args.flickr:
        download_flickr_subset(DATA_ROOT)
