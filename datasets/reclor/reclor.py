import json
import os
import datasets


class Reclor(datasets.GeneratorBasedBuilder):
    """ReAding Comprehension Dataset From Examination dataset from CMU"""

    VERSION = datasets.Version("0.1.0")

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description="reclor",
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "options": datasets.features.Sequence(datasets.Value("string"))
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="reclor",
            citation="reclor",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to

        print('data_files passed to _split_generators: ', self.config.data_files)
        downloaded_files = dl_manager.download_and_extract(self.config.data_files)
        results = []
        if "train" in downloaded_files:
            results.append(
                datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                        gen_kwargs={"files": downloaded_files["train"]}))
        if "validation" in downloaded_files:
            results.append(
                datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                                        gen_kwargs={"files": downloaded_files["validation"]}))
        return results

    def _generate_examples(self, files):
        """Yields examples."""
        # print('files: ', files)
        for filename in files:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for id_, row in enumerate(data):
                    yield id_, {
                        "context": row["context"],
                        "question": row["question"],
                        "options": row["answers"],
                        "answer": str(row["label"]),
                        "id": row["id_string"],
                    }

