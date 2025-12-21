"""
Helper to download datasets used in assignments.
This script will not automatically fetch proprietary datasets requiring credentials (e.g., COCO with restricted access), but it will provide helpers and instructions.
"""
import os
import argparse

DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')

def ensure_data_dirs():
    os.makedirs(DATA_ROOT, exist_ok=True)
    print('Data root:', os.path.abspath(DATA_ROOT))


def download_urbansound(dest):
    print('Please download UrbanSound8K from https://urbansounddataset.weebly.com/urbansound8k.html and place it under:', dest)


def download_flickr_subset(dest):
    print('Place the Flickr/COCO subset (images + captions.csv) under', dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--urbansound', action='store_true')
    parser.add_argument('--flickr', action='store_true')
    args = parser.parse_args()
    ensure_data_dirs()
    if args.urbansound:
        download_urbansound(DATA_ROOT)
    if args.flickr:
        download_flickr_subset(DATA_ROOT)
