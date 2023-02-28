import glob
import shutil
import os
import argparse
import zipfile


def group_images(path):
    if not os.path.exists(path + '/dirty'):
        os.makedirs(path + '/dirty')

    if not os.path.exists(path + '/clean'):
        os.makedirs(path + '/clean')

    for image in glob.glob(path + '/*.jpg'):
        image_class = image.split('/')[-1].split('.')[0].split('_')[0]
        try:
            shutil.move(image, path + '/' + image_class)
        except FileNotFoundError:
            print('Something went wrong with the file ' + image)


def download_dataset(username, key):
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files('faizalkarim/cleandirty-road-classification', path="../data")
    with zipfile.ZipFile('../data/cleandirty-road-classification.zip', 'r') as zip_ref:
        zip_ref.extractall('../data')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', type=str, required=True, help='kaggle username')
    parser.add_argument('--key', type=str, required=True, help='api key')
    opt = parser.parse_args()

    download_dataset(opt.username, opt.key)
    group_images('../data/Images/Images')


if __name__ == '__main__':
    main()
