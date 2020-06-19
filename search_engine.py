import time
import json
from collections import defaultdict
from utils.io_utils import load_json, load_pickle
from utils.tokenization import BasicTokenizer
from ranking.ranker import RankerBase, TfidfRanker, BM25Ranker, VSMRanker, SLMARanker, SLMDRanker
from index.indexer import WikiParser


parser_strategy = {
    "WikiParser": WikiParser
}


class SearchEngine():
    def __init__(self, index_config_path, ranker_name):
        self.index_cfg = load_json(index_config_path)
        self.index_pst = defaultdict(lambda:[[],[]], load_pickle(self.index_cfg["index_path"]))
        self.page_count = self.index_cfg["page_count"]
        self.vocab = load_json(self.index_cfg["vocab_path"])
        # self.tokenizer = BasicTokenizer(never_split=[])
        self.ranker = None
        self._start_ranker(ranker_name)
        self.parser = parser_strategy[self.index_cfg["name"]]()

        # source file
        self.fstream = open(self.index_cfg["data_path"], "r", encoding="utf8")
        self.page_positions = load_pickle(self.index_cfg["page_positions_path"])

    
    def _start_ranker(self, ranker_name):
        if ranker_name == "base":
            self.ranker = RankerBase()
        elif ranker_name == "Tfidf":
            self.ranker = TfidfRanker(self.page_count, self.index_cfg["page_len_path"], last_ranker=self.ranker)
        elif ranker_name == "bm25":
            self.ranker = BM25Ranker(self.page_count, self.index_cfg["page_len_path"], last_ranker=self.ranker)
        elif ranker_name == "VSM-tf":
            self.ranker = VSMRanker(self.page_count, self.index_cfg["page_len_path"], self.index_cfg["page_word_index_path"], self.index_pst, use_tf=True, last_ranker=self.ranker)
        elif ranker_name == "VSM-tfidf":
            self.ranker = VSMRanker(self.page_count, self.index_cfg["page_len_path"], self.index_cfg["page_word_index_path"], self.index_pst, use_tf=False, last_ranker=self.ranker)
        elif ranker_name == "SLM-A":
            self.ranker = SLMARanker(self.page_count, self.index_cfg["page_len_path"], self.index_cfg["page_word_index_path"], len(self.vocab), delta=0.5, last_ranker=self.ranker)
        elif ranker_name == "SLM-D":
            self.ranker = SLMDRanker(self.page_count, self.index_cfg["page_len_path"], self.index_cfg["page_word_index_path"], self.vocab, lambda_=0.5, last_ranker=self.ranker)
        else:
            self.ranker = None
        self.ranker_name = ranker_name


    def query(self, sen, only_title=False):
        bow = self.parser.preprocess_sen(sen)
        if only_title:
            for i, x in enumerate(bow):
                bow[i] = x + ".t"
        psts = []
        for w in bow:
            psts.append(self.index_pst[w])
        docID_list = self.ranker.ranking(bow, psts)
        return docID_list


    def change_ranker(self, ranker_name):
        if self.ranker_name != ranker_name:
            self._start_ranker(ranker_name)


    def get_docs(self, docID):
        self.fstream.seek(self.page_positions[docID], 0)
        return json.loads(self.fstream.readline())


def test():
    cfg_path = "data/index_test.json"
    ranker_name = "Tfidf"
    sg = SearchEngine(cfg_path, ranker_name)
    t = time.time()
    res = sg.query("jflskdjg")
    print(res)

    with open("data/processed/wiki_00", 'r', encoding='utf-8') as f:
        for r in res:
            print(sg.page_positions[r[0]])
            f.seek(sg.page_positions[r[0]], 0)
            context = f.readline()
            # print(context)
            line = json.loads(context)
            # print(line['id'], line['title'])

    t = time.time() - t
    print(res)
    print(t)


if __name__ == "__main__":
    test()