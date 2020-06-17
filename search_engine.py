from utils.io_utils import load_json, load_pickle
from utils.tokenization import BasicTokenizer
from ranking.ranker import RankerBase
from index.indexer import WikiParser


rank_strategy = {
    "base": RankerBase
}

parser_strategy = {
    "WikiParser": WikiParser
}


class SearchEngine():
    def __init__(self, index_config_path, ranker_name):
        self.index_cfg = load_json(index_config_path)
        self.index_pst = load_pickle(self.index_cfg["index_path"])
        self.page_count = self.index_cfg["page_count"]
        self.vocab = load_json(self.index_cfg["vocab_path"])
        self.tokenizer = BasicTokenizer(never_split=[])
        self.ranker = rank_strategy[ranker_name]()
        self.parser = parser_strategy[self.index_cfg["data_parser"]]


    def query(self, word):
        word_pst = self.index_pst[word]
        print(word_pst)


    def change_ranker(self, ranker_name):
        pass


def test():
    cfg_path = "data/index_test.json"
    ranker_name = "base"
    sg = SearchEngine(cfg_path, ranker_name)
    sg.query("delegitim")


if __name__ == "__main__":
    test()