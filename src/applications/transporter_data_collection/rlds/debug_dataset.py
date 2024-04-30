import tensorflow_datasets as tfds

from envlogger.backends import rlds_utils
from envlogger import reader
path = '/root/data_collection_ws/src/applications/transporter_data_collection/rlds/dummy_data/transporter'

with reader.Reader(data_directory = path) as r:
    for idx, episode in enumerate(r.episodes):
        print("episode: {}".format(idx))
        for idx, step in enumerate(episode):
            print("step: {}".format(idx))
            print(step.timestep.observation)
            print(step.timestep.reward)
            print(step.timestep.discount)
            print(step.action)
           
# builder = tfds.builder_from_directory(path)
# builder = rlds_utils.maybe_recover_last_shard(builder)
# ds = builder.as_dataset()
