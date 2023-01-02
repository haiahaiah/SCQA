"""TODO(race): Add a description here."""

import json
import glob
import datasets

_CITATION = """\
@article{lai2017large,
    title={RACE: Large-scale ReAding Comprehension Dataset From Examinations},
    author={Lai, Guokun and Xie, Qizhe and Liu, Hanxiao and Yang, Yiming and Hovy, Eduard},
    journal={arXiv preprint arXiv:1704.04683},
    year={2017}
}
"""

_DESCRIPTION = """\
Race is a large-scale reading comprehension dataset with more than 28,000 passages and nearly 100,000 questions. The
 dataset is collected from English examinations in China, which are designed for middle school and high school students.
The dataset can be served as the training and test sets for machine comprehension.

"""

_URL = "http://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz"


class Race(datasets.GeneratorBasedBuilder):
    """ReAding Comprehension Dataset From Examination dataset from CMU"""

    VERSION = datasets.Version("0.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="high", description="Exams designed for high school students", version=VERSION),
        datasets.BuilderConfig(
            name="middle", description="Exams designed for middle school students", version=VERSION
        ),
        datasets.BuilderConfig(
            name="all", description="Exams designed for both high school and middle school students", version=VERSION
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
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
            homepage="http://www.cs.cmu.edu/~glai1/data/race/",
            citation=_CITATION,
        )

    # def _split_generators(self, dl_manager):
    #     """Returns SplitGenerators."""
    #     # Downloads the data and defines the splits
    #     # dl_manager is a datasets.download.DownloadManager that can be used to
    #     archive = dl_manager.download(_URL)
    #     case = str(self.config.name)
    #     if case == "all":
    #         case = ""
    #     return [
    #         datasets.SplitGenerator(
    #             name=datasets.Split.TEST,
    #             # These kwargs will be passed to _generate_examples
    #             gen_kwargs={"train_test_or_eval": f"RACE/test/{case}", "files": dl_manager.iter_archive(archive)},
    #         ),
    #         datasets.SplitGenerator(
    #             name=datasets.Split.TRAIN,
    #             # These kwargs will be passed to _generate_examples
    #             gen_kwargs={"train_test_or_eval": f"RACE/train/{case}", "files": dl_manager.iter_archive(archive)},
    #         ),
    #         datasets.SplitGenerator(
    #             name=datasets.Split.VALIDATION,
    #             # These kwargs will be passed to _generate_examples
    #             gen_kwargs={"train_test_or_eval": f"RACE/dev/{case}", "files": dl_manager.iter_archive(archive)},
    #         ),
    #     ]

    # YH
    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        case = str(self.config.name)
        if case == "all":
            case = ""

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
                # data_raw = f.readline().strip()     # read txt with json style
                # data = json.loads(data_raw)
                # questions = data["questions"]
                # answers = data["answers"]
                # options = data["options"]
                # for i in range(len(questions)):
                #     question = questions[i]
                #     answer = ord(answers[i]) - ord('A')
                #     option = options[i]
                #     yield f"{filename}_{i}", {
                #         "id": data["id"] + "-" + str(i),
                #         "context": data["article"],
                #         "question": question,
                #         "answer": answer,
                #         "options": option,
                #     }

                data = json.load(f)
                # data_raw = f.readline().strip()     # read txt with json style
                # data = json.loads(data_raw)
                for id, value in data.items():
                    question = value["question"]
                    answer = value["label"]
                    context = value["context"]
                    options = value["options"]
                    for i in range(len(options)):
                        option = options[i]
                        yield f"{filename}_{i}", {
                            "id": data["id"] + "-" + str(i),
                            "context": context,
                            "question": question,
                            "answer": answer,
                            "options": option,
                        }
