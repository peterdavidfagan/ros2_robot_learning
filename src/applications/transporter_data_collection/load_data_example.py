import os
import tarfile

import tensorflow_datasets as tfds
from huggingface_hub import hf_hub_download

DATA_DIR="/home/robot"
FILENAME="data.tar.xz"
EXTRACTED_FILENAME="data"
FILEPATH=os.path.join(DATA_DIR, FILENAME)
EXTRACTED_FILEPATH=os.path.join(DATA_DIR, EXTRACTED_FILENAME)

# download data from huggingface
hf_hub_download(
        repo_id="peterdavidfagan/transporter_networks", 
        repo_type="dataset",
        filename=FILENAME,
        local_dir=DATA_DIR,
        )

# uncompress file
with tarfile.open(FILEPATH, 'r:xz') as tar:
    tar.extractall(path=DATA_DIR)
os.remove(FILEPATH)

# load with tfds
ds = tfds.builder_from_directory(EXTRACTED_FILEPATH).as_dataset()['train']

# basic inspection of data
print(ds.element_spec)
for eps in ds:
    print(eps["extrinsics"])
    for step in eps["steps"]:
        print(step["is_first"])
        print(step["is_last"])
        print(step["is_terminal"])
        print(step["action"])
