from math import log10, log
from utils.io_utils import load_json, load_pickle
from collections import defaultdict



class RankerBase():
    # keep all
    def __init__(self, doc_total, doc_len_path, doc_word_index_path, last_ranker=None):
        self.doc_total = doc_total
        if last_ranker:
            self.doc_len = last_ranker.doc_len
            self.doc_word_index = last_ranker.doc_word_index
        else:
            self.doc_len = load_pickle(doc_len_path)    # a list
            self.doc_word_index = load_pickle(doc_word_index_path)  # list of dicts



    def ranking(self, bow, psts):
        # need optimization
        res = set()
        for pst in psts:
            for doc in pst[0]:
                res.add(doc)
        f_res = []
        for i in res:
            f_res.append((i,0))
        return f_res


class TfidfRanker(RankerBase):
    # using tf idf to rank
    def __init__(self, doc_total, doc_len_path, doc_word_index_path, last_ranker=None):
        super(TfidfRanker, self).__init__(doc_total, doc_len_path, doc_word_index_path, last_ranker=last_ranker)


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


class BM25Ranker(RankerBase):
    # using bm25, which is similar to tf-idf
    def __init__(self, doc_total, doc_len_path, doc_word_index_path, last_ranker=None):
        super(BM25Ranker, self).__init__(doc_total, doc_len_path, doc_word_index_path, last_ranker=last_ranker)
        self.k1 = 2
        self.b = 0.75

        tmp = 0
        for i in self.doc_len:
            tmp += i
        self.avglen = tmp / self.doc_total


    def ranking(self, bow, psts):
        res = defaultdict(lambda:0)
        for i, word in enumerate(bow):
            doc_list, fre_list = psts[i]
            doc_count = len(doc_list)
            for j, docID in enumerate(doc_list):
                k = self.k1 * (1 - self.b + self.b * self.doc_len[docID] / self.avglen)
                r = fre_list[j] * (self.k1 + 1) / (fre_list[j] + k)
                idf = log10((self.doc_total - doc_count + 0.5) / (doc_count + 0.5))
                res[docID] += r * idf
        return sorted(res.items(), key=lambda x:x[1], reverse=True)


class VSMRanker(RankerBase):
    # vector space model, use tf or normalized tf
    def __init__(self, doc_total, doc_len_path, doc_word_index_path, index, use_tf=True, last_ranker=None):
        super(VSMRanker, self).__init__(doc_total, doc_len_path, doc_word_index_path, last_ranker=last_ranker)

        self.use_tf = use_tf
        if use_tf:
            self.index = None
        else:
            self.index= index


    def ranking(self, bow, psts):
        bow_dict = defaultdict(lambda:0)
        for word in bow_dict:
            bow_dict[word] += 1

        res = defaultdict(lambda:0)
        visted = defaultdict(lambda:False)

        if self.use_tf:
            for pst in psts:
                doc_list, _ = pst
                for docID in doc_list:
                    if not visted[docID]:
                        word_index = self.doc_word_index[docID]
                        tmp = 0
                        for word in bow_dict:
                            tmp += bow_dict[word] * word_index[word]
                        res[docID] = tmp / word_index["_sum"]  # word_index["_sum"] may be divided by 0, but here for query it's impossible
                        visted[docID] = True
        else:
            # use tf-idf
            for pst in psts:
                doc_list, _ = pst
                for docID in doc_list:
                    if not visted[docID]:
                        word_index = self.doc_word_index[docID]
                        tmp = 0
                        for word in bow_dict:
                            tmp += bow_dict[word] * word_index[word]
                        res[docID] = tmp / word_index["_sum_tfidf"] # word_index["_sum_tfidf"] may be divided by 0, but here for query it's impossible
                        visted[docID] = True

        return sorted(res.items(), key=lambda x:x[1], reverse=True)


class SLMARanker(RankerBase):
    # statistic language model with additive smoothing
    '''
    P(D)*P(q|D), P(D) is the same, P(q|D) = \Pi P(w|D) (using log, equal to sum(logP(w|D)))
    MLE: P(w|D) = c(w|D) / sum(c(w|D))
    Laplace smoothing: delta=1, P(w|D) = (c(w|D) + 1) / (doc_len + |Vocab|)
    '''
    def __init__(self, doc_total, doc_len_path, doc_word_index_path, voc_len, delta=0.5, last_ranker=None):
        super(SLMARanker, self).__init__(doc_total, doc_len_path, doc_word_index_path, last_ranker)
        self.delta = delta  # (0,1]
        self.voc_len = voc_len * delta


    def ranking(self, bow, psts):
        res = {}
        visted = defaultdict(lambda:False)

        for pst in psts:
            doc_list, _ = pst
            for docID in doc_list:
                if not visted[docID]:
                    score = 0
                    tmp_index = self.doc_word_index[docID]
                    for word in bow:
                        if word in tmp_index:
                            score += log((tmp_index[word] + self.delta) / (self.doc_len[docID] + self.voc_len))     # doc_len[docID] may be divided by 0, but here for query it's impossible
                        else:
                            score += log(self.delta / (self.doc_len[docID] + self.voc_len))                         # doc_len[docID] may be divided by 0, but here for query it's impossible
                    res[docID] = score
                    visted[docID] = True

        return sorted(res.items(), key=lambda x:x[1], reverse=True)


class SLMDRanker(RankerBase):
    # statistic language model with dirichlet smoothing
    '''
    P(D)*P(q|D), P(D) is the same, P(q|D) = \Pi P(w|D) (using log, equal to sum(logP(w|D)))
    MLE: P(w|D) = c(w|D) / sum(c(w|D))
    '''
    def __init__(self, doc_total, doc_len_path, doc_word_index_path, vocab, total_word, lambda_=0.5, last_ranker=None):
        super(SLMDRanker, self).__init__(doc_total, doc_len_path, doc_word_index_path, last_ranker)
        self.lambda_ = lambda_  # [0,1]
        self.vocab = vocab
        self.total_word = total_word


    def ranking(self, bow, psts):
        res = {}
        visted = defaultdict(lambda:False)

        for pst in psts:
            doc_list, _ = pst
            for docID in doc_list:
                if not visted[docID]:
                    score = 0
                    tmp_index = self.doc_word_index[docID]
                    for word in bow:
                        if word in tmp_index:
                            score += log(self.lambda_ * (tmp_index[word] / self.doc_len[docID]) + (1 - self.lambda_) * self.vocab[word] / self.total_word)
                        else:
                            score += log((1 - self.lambda_) * self.vocab[word] / self.total_word)
                    res[docID] = score
                    visted[docID] = True

        return sorted(res.items(), key=lambda x:x[1], reverse=True)
