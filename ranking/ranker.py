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