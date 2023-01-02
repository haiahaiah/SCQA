
import json

import datasets

_CITATION = """\
@inproceedings{gardner2020evaluating,
  title={Evaluating Models’ Local Decision Boundaries via Contrast Sets},
  author={Gardner, Matt and Artzi, Yoav and Basmov, Victoria and Berant, Jonathan and Bogin, Ben and Chen, Sihao and Dasigi, Pradeep and Dua, Dheeru and Elazar, Yanai and Gottumukkala, Ananth and others},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2020},
  pages={1307--1323},
  year={2020}
}
"""


class ContrastBoolq(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.1.0")

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description='contrast boolq',
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
            homepage="https://github.com/allenai/contrast-sets",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

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
        print('filepath[0]:', filepath[0])
        id_1 = -1
        with open(filepath[0], 'r') as f:
            data_list = json.load(f)["data"]
            for data in data_list:
                questions = [data["question"]]
                answers = [data["answer"]]
                passage = data["paragraph"]
                id_1 += 1
                for perturbed_q in data["perturbed_questions"]:
                    q = perturbed_q["perturbed_q"]
                    a = perturbed_q["answer"]
                    questions.append(q)
                    answers.append(a)

                for i in range(len(questions)):
                    q = questions[i]
                    a = True if answers[i] in ["True", "TRUE", "true"] else False
                    id_ = str(id_1) + "-" + str(i)
                    yield id_, {"id": id_, "question": q, "answer": a, "passage": passage}
