import time
import json
from collections import defaultdict
from utils.io_utils import load_json, load_pickle
from utils.tokenization import BasicTokenizer
from ranking.ranker import RankerBase, TfidfRanker, BM25Ranker, VSMRanker, SLMARanker, SLMDRanker
from index.indexer import WikiParser
from string import punctuation


parser_strategy = {
    "WikiParser": WikiParser
}


class SearchEngine():
    def __init__(self, index_config_path, ranker_name):
        self.index_cfg = load_json(index_config_path)
        self.index_pst = defaultdict(lambda:[[],[]], load_pickle(self.index_cfg["index_path"]))
        self.page_count = self.index_cfg["page_count"]
        self.vocab = load_json(self.index_cfg["vocab_path"])
        self.total_word = 0
        for v in self.vocab.values():
            self.total_word += v

        # self.tokenizer = BasicTokenizer(never_split=[])
        self.ranker = None
        self._start_ranker(ranker_name)
        self.parser = parser_strategy[self.index_cfg["name"]]()

        # source file
        self.fstream = open(self.index_cfg["data_path"], "r", encoding="utf8")
        self.page_positions = load_pickle(self.index_cfg["page_positions_path"])

        self.punc = " " + punctuation + "\n"

    
    def _start_ranker(self, ranker_name):
        if ranker_name == "base":
            self.ranker = RankerBase(self.page_count, self.index_cfg["page_len_path"], self.index_cfg["page_word_index_path"], last_ranker=self.ranker)
        elif ranker_name == "Tfidf":
            self.ranker = TfidfRanker(self.page_count, self.index_cfg["page_len_path"], self.index_cfg["page_word_index_path"], last_ranker=self.ranker)
        elif ranker_name == "bm25":
            self.ranker = BM25Ranker(self.page_count, self.index_cfg["page_len_path"], self.index_cfg["page_word_index_path"], last_ranker=self.ranker)
        elif ranker_name == "VSM-tf":
            self.ranker = VSMRanker(self.page_count, self.index_cfg["page_len_path"], self.index_cfg["page_word_index_path"], self.index_pst, use_tf=True, last_ranker=self.ranker)
        elif ranker_name == "VSM-tfidf":
            self.ranker = VSMRanker(self.page_count, self.index_cfg["page_len_path"], self.index_cfg["page_word_index_path"], self.index_pst, use_tf=False, last_ranker=self.ranker)
        elif ranker_name == "SLM-A":
            self.ranker = SLMARanker(self.page_count, self.index_cfg["page_len_path"], self.index_cfg["page_word_index_path"], len(self.vocab), delta=0.5, last_ranker=self.ranker)
        elif ranker_name == "SLM-D":
            self.ranker = SLMDRanker(self.page_count, self.index_cfg["page_len_path"], self.index_cfg["page_word_index_path"], self.vocab, self.total_word, lambda_=0.5, last_ranker=self.ranker)
        else:
            self.ranker = None
        self.ranker_name = ranker_name


    def query(self, sen, only_title=False):
        bow = self.parser.preprocess_sen(sen)
        if only_title:
            query_bow = []
            for x in bow:
                query_bow.append(x + ".t")
        else:
            query_bow = bow
        psts = []
        for w in query_bow:
            psts.append(self.index_pst[w])
        docID_list = self.ranker.ranking(bow, psts)
        return docID_list, bow


    def change_ranker(self, ranker_name):
        if self.ranker_name != ranker_name:
            self._start_ranker(ranker_name)


    def get_docs(self, docID):
        self.fstream.seek(self.page_positions[docID], 0)
        return json.loads(self.fstream.readline())


    def highlight(self, bow, context):
        tmp = []
        tag = True  # last is punc
        for i, s in enumerate(line):
            if s in punc:
                if tag:
                    continue
                pos_e = i
                word = line[pos_s:pos_e]
                word = sg.parser.preprocess_word(word)
                if word in bow:
                    tmp.append((pos_s, pos_e))
                tag = True
            else:
                if tag:
                    pos_s = i
                    tag = False
        return tmp


def test():
    cfg_path = "data/index_test.json"
    ranker_name = "Tfidf"
    sg = SearchEngine(cfg_path, ranker_name)
    res, bow = sg.query("test machine")
    res = res[1:10]
    print(res)

    t = time.time()
    count = 0
    punc = " " + punctuation + "\n"

    with open("data/processed/wiki_00", 'r', encoding='utf-8') as f:
        for r in res:
            count += 1
            f.seek(sg.page_positions[r[0]], 0)
            context = f.readline()
            line = json.loads(context)["text"]
            # print(line['id'], line['title'])

    t = time.time() - t
    print(t)


if __name__ == "__main__":
    test()