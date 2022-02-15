import conllu
import datasets
import nlp

_DESCRIPTION = """
"""

_NAMES = [
    "combined_final"
]

_UD_DATASETS = {
    "combined_final": {
        "train": "../datasets/final_pos_data/en_final.conllu",
        "dev": "../datasets/final_pos_data/en_final.conllu",
        "test": "../datasets/final_pos_data/en_final.conllu",
    }
}

_DESCRIPTIONS = {
    "combined_final": ""
}

# class UniversaldependenciesConfig(datasets.BuilderConfig):
#     """BuilderConfig for Universal dependencies"""

#     def __init__(self, data_url, **kwargs):
#         super(UniversaldependenciesConfig, self).__init__(version=datasets.Version("2.7.0", ""), **kwargs)

#         self.data_url = data_url


class UniversalDependencies(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("2.7.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=name,
           )
        for name in _NAMES
    ]
    BUILDER_CONFIG_CLASS = datasets.BuilderConfig

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "idx": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "label": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "en",
                                "nl",
                                "pt",
                                "hi",
                                "es",
                                "tl",
                                "zh"
                            ]
                        )
                    )
                }
            ),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        

        if isinstance(self.config.data_files, (str, list, tuple)):
             files = self.config.data_files
             if isinstance(files, str):
                 files = [files]
             return [nlp.SplitGenerator(
                         name=nlp.Split.TRAIN,
                         gen_kwargs={"files": files})]
        splits = []
        for split_name in [nlp.Split.TRAIN, nlp.Split.VALIDATION, nlp.Split.TEST]:
            if split_name in self.config.data_files:
                files = self.config.data_files[split_name]
                if isinstance(files, str):
                    files = [files]
                splits.append(
                    nlp.SplitGenerator(
                            name=split_name,
                            gen_kwargs={"files": files}))
        return splits

    def _generate_examples(self, files):
        id = 0
        for path in files:
            with open(path, "r", encoding="utf-8") as data_file:
                tokenlist = list(conllu.parse_incr(data_file))
                for sent in tokenlist:
                    if "sent_id" in sent.metadata:
                        idx = sent.metadata["sent_id"]
                    else:
                        idx = id

                    tokens = [token["form"] for token in sent]

                    if "text" in sent.metadata:
                        txt = sent.metadata["text"]
                    else:
                        txt = " ".join(tokens)

                    yield id, {
                        "idx": str(idx),
                        "text": txt,
                        "tokens": [token["form"] for token in sent],
                        "label": [token["lemma"] for token in sent]
                    }
                    id += 1