import datasets
from datasets import DownloadManager, DatasetInfo


class BindingDataset(datasets.GeneratorBasedBuilder):

    def _info(self) -> DatasetInfo:
        pass

    def _split_generators(self, dl_manager: DownloadManager):
        pass

    def _generate_examples(self, **kwargs):
        pass