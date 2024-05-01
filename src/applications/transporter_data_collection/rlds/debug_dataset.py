import tensorflow_datasets as tfds
import rlds

from envlogger.backends import rlds_utils
from envlogger import reader

path = '/root/data_collection_ws/src/applications/transporter_data_collection/data'          
ds = tfds.builder_from_directory(path).as_dataset()['train']
print(ds.element_spec)

for eps in ds:
    print(eps["extrinsics"])
    for step in eps["steps"]:
        print(step["is_first"])
        print(step["is_last"])
        print(step["is_terminal"])
        print(step["action"])
#ds_steps = ds.flat_map(lambda episode: episode[rlds.STEPS])
#ds_steps = ds_steps.as_numpy_iterator()
#for sample in ds_steps:
#    print(sample)