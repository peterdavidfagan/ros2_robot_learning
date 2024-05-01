import os
import tarfile

import tensorflow_datasets as tfds
from huggingface_hub import HfApi

LOCAL_FILEPATH="/home/robot/Code/ros2_robotics_research_toolkit/src/applications/transporter_data_collection/data"
OUTPUT_FILENAME="data.tar.xz"
COMPRESSED_LOCAL_FILEPATH="/home/robot/Code/ros2_robotics_research_toolkit/src/applications/transporter_data_collection/data.tar.xz"
REPO_FILEPATH="/data.tar.xz"

if __name__=="__main__":
    # compress the file
    with tarfile.open(OUTPUT_FILENAME, "w:xz") as tar:
        tar.add(LOCAL_FILEPATH, arcname=".")

    # upload to huggingface
    api = HfApi()
    api.upload_file(
        repo_id="peterdavidfagan/transporter_networks",
        repo_type="dataset",
        path_or_fileobj=COMPRESSED_LOCAL_FILEPATH,
        path_in_repo="/data.tar.xz",
    )