"""convert riegli to rlds."""

from __future__ import annotations


import numpy as np
import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.rlds import rlds_base


_DESCRIPTION = """Placeholder."""

_CITATION = """Placeholder."""

_HOMEPAGE = 'placeholder.'


class Transporter(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for transporter network data."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  _DATA_PATHS = {
      'example': 'placeholder',
       }

  # pytype: disable=wrong-keyword-args
  BUILDER_CONFIGS = [
      rlds_base.DatasetConfig(
          name='transporter',
          observation_info={
            "overhead_camera/rgb": tfds.features.Tensor(shape=(621,1104, 3), dtype=np.uint8),
            "overhead_camera/depth": tfds.features.Tensor(shape=(621,1104), dtype=np.float32),
            },
          action_info=tfds.features.Tensor(shape=(7,), dtype=np.float64),
          reward_info=np.float64,
          discount_info=np.float64,
          citation=_CITATION,
          homepage=_HOMEPAGE,
          overall_description=_DESCRIPTION,
          description='placeholder',
          supervised_keys=None,  # pytype: disable=wrong-arg-types  # gen-stub-imports
      ),
  ]

  # pytype: enable=wrong-keyword-args


  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return rlds_base.build_info(self.builder_config, self)

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = "/root/data_collection_ws/src/applications/transporter_data_collection/rlds/dummy_data/transporter"
    from pathlib import Path
    path = Path(path)
    print("GOT HERE")
    return {
        'train': self._generate_examples(path),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    return rlds_base.generate_examples(path)