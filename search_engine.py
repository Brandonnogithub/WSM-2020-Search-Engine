import time
import json
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
        # self.tokenizer = BasicTokenizer(never_split=[])
        self.ranker_name = ranker_name
        self.ranker = rank_strategy[ranker_name]()
        self.parser = parser_strategy[self.index_cfg["name"]]()

        # source file
        self.fstream = open(self.index_cfg["data_path"], "r", encoding="utf8")
        self.page_positions = load_pickle(self.index_cfg["page_positions_path"])


    def query(self, sen):
        bow = self.parser.preprocess_sen(sen)
        psts = []
        for w in bow:
            psts.append(self.index_pst[w])
        docID_list = self.ranker.ranking(bow, psts)
        return docID_list


    def change_ranker(self, ranker_name):
        if self.ranker_name != ranker_name:
            self.ranker = rank_strategy[ranker_name]()
            self.ranker_name = ranker_name


    def get_docs(self, docID):
        self.fstream.seek(self.page_positions[docID], 0)
        return json.loads(self.fstream.readline())


def test():
    cfg_path = "data/index_test.json"
    ranker_name = "base"
    sg = SearchEngine(cfg_path, ranker_name)
    t = time.time()
    res = sg.query("delegitim")
    t = time.time() - t
    print(res)
    print(t)


if __name__ == "__main__":
    test()