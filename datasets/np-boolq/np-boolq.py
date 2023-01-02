"""TODO(boolq): Add a description here."""

import json

import datasets

# TODO(boolq): BibTeX citation
_CITATION = """\
@article{khashabi2020naturalperturbations,
  title={Natural Perturbation for Robust Question Answering},
  author={D. Khashabi and T. Khot and A. Sabhwaral},
  journal={arXiv preprint},
  year={2020}
}
"""


class NPBoolq(datasets.GeneratorBasedBuilder):
    """TODO(boolq): Short description of my dataset."""

    # TODO(boolq): Set up version.
    VERSION = datasets.Version("0.1.0")

    def _info(self):
        # TODO(boolq): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description='np boolq',
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("bool"),
                    "passage": datasets.Value("string")
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://github.com/allenai/natural-perturbations",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(boolq): Downloads the data and defines the splits

        downloaded_files = dl_manager.download_and_extract(self.config.data_files)

        results = []
        if "train" in downloaded_files:
            results.append(
                datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}))
        if "validation" in downloaded_files:
            results.append(datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                                                   gen_kwargs={"filepath": downloaded_files["validation"]}))
        return results

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(boolq): Yields (key, example) tuples from the dataset
        print('filepath[0]:', filepath[0])
        with open(filepath[0], 'r') as f:
            # for id_, row in enumerate(f):
                # data = json.loads(row)

            for i, line in enumerate(f.readlines()):
                line = line.strip()
                data = json.loads(line)
                question = data["question"]
                answer = True if data["hard_label"] in ["True", "true"] else False
                passage = data["passage"]
                yield i, {"id": data["question_id"], "question": question, "answer": answer, "passage": passage}
