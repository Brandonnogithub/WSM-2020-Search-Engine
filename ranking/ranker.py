class RankerBase():
    # keep all
    def __init__(self):
        pass


    def ranking(self, bow, psts, **kwargs):
        # need optimization
        res = set()
        for pst in psts:
            for doc in pst[0]:
                res.add(doc)
        return res


class TfidfRanker(RankerBase):
    # using tf idf to rank
    def __init__(self):
        super(TfidfRanker, self).__init__()
        pass

    def ranking(self, bow, psts, **kwargs):
        pass