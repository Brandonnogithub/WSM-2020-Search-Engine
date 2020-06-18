from math import log10
from utils.io_utils import load_json, load_pickle
from collections import defaultdict

class RankerBase():
    # keep all
    def __init__(self):
        pass


    def ranking(self, bow, psts):
        # need optimization
        res = set()
        for pst in psts:
            for doc in pst[0]:
                res.add(doc)
        return res


class TfidfRanker(RankerBase):
    # using tf idf to rank
    def __init__(self, doc_total, doc_len_path):
        super(TfidfRanker, self).__init__()
        self.doc_total = doc_total                  # 6047512 for wiki
        self.doc_len = load_pickle(doc_len_path)    # a list


    def ranking(self, bow, psts):
        res = defaultdict(lambda:0)
        for i, word in enumerate(bow):
            doc_list, fre_list = psts[i]
            doc_count = len(doc_list) + 1
            for j, docID in enumerate(doc_list):
                tf = fre_list[j] / self.doc_len[docID]
                idf = log10(self.doc_total / doc_count)
                res[docID] += tf * idf
        return sorted(res.items(), key=lambda x:x[1], reverse=True)
