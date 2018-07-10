import os
import urllib.request
import zipfile
import shutil


data_dir = "./data"
dataset_url = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip"

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)
    print("Created data directory " + data_dir)


output_path = os.path.join(data_dir, "gtsrb_train")
output_path_zip = data_dir + "/gtsrb_train.zip"


if not os.path.isdir(output_path):

    if not os.path.isfile(output_path_zip):

        # Try to download the resource
        try:
            print("Downloading dataset (this could take a while) ...")
            file = urllib.request.urlopen(dataset_url)
            # Open output file and write content of resource to it
            open(output_path_zip, "wb").write(file.read())

        except urllib.request.HTTPError as error:
            print("HTTP Error while downloading the dataset: " + error.code)

        print("Successfully downloaded dataset")

    else:
        print("Already downloaded")


    print("Unzipping data (this could take a while) ...")

    # Unzip compressed file
    with zipfile.ZipFile(output_path_zip, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    # Move images folder
    shutil.move(data_dir + "/GTSRB/Final_Training/Images", output_path)
    shutil.rmtree(data_dir + "/GTSRB")

    print("Successfully unzipped dataset")

else:
    print("Already unzipped")

