import os
import time
import re
import pickle
import nltk
import argparse
import json
import settings
from math import log10
from collections import defaultdict
from utils.io_utils import load_json, dump_pickle, dump_json, load_pickle
from utils.tokenization import BasicTokenizer
from string import punctuation
from array import array
from tqdm import tqdm


page_total = 6047512


class WikiParser():
    # a class to parser wiki dataset
    def __init__(self, data_path="", output_dir="", debug=False):
        self.name = "WikiParser"
        self.data_path = data_path
        self.STOP_WORDS_PATH = os.path.join("index", "stop_words.json")
        self.stemmer_path = output_dir + "/stem.json"
        self.stemmer = nltk.stem.SnowballStemmer('english')  # You can use porter or other stemmer
        
        if os.path.exists(self.stemmer_path):
            self.stem_word_dict = load_json(self.stemmer_path)
            self.stem_word_dict = defaultdict(lambda:False, self.stem_word_dict)
        else:
            self.stem_word_dict = defaultdict(lambda:False)
        self.stop_words = set(load_json(self.STOP_WORDS_PATH)["stop_words"])
        self.tokenizer = BasicTokenizer(never_split=[])
        self.punctuation = re.compile(r"[{}]+".format(punctuation))

        self.page_count = 0 # denote which number of wiki page it is
        self.page_positions = dict()    # store position of each page in source file
        self.page_positions_path = output_dir + "/page_positions.pickle"
        self.page_len_path = output_dir + "/page_len.pickle"    # a list to store length of each page
        self.page_len_list = []
        self.page_len = 0

        self.page_word_index_path = output_dir + "/page_word_idnex.pickle"
        self.page_word_index = []

        self.debug = debug

    
    def preprocess_sen(self, sen):
        sen = sen.strip()   # remove "\n"
        sen = re.sub(self.punctuation, " ", sen).lower()    # remove punctuation
        sen_list = sen.split()      # tokenize

        # stemming
        res = []
        for word in sen_list:
            if word in self.stop_words:
                continue
            stem_word = self.stem_word_dict[word]
            if not stem_word:
                stem_word = self.stemmer.stem(word) # do stemming
                self.stem_word_dict[word]=stem_word
            res.append(stem_word)
        return res
            

    def preprocess_word(self, word):
        word = word.lower()
        if word in self.stop_words:
            return ""
        else:
            stem_word = self.stem_word_dict[word]
            if not stem_word:
                stem_word = self.stemmer.stem(word) # do stemming
                self.stem_word_dict[word]=stem_word
            return stem_word


    
    def _readlines(self):
        # read wiki data page by page
        with open(self.data_path, "r", encoding="utf8") as f:
            seek_pos = f.tell()
            c_page = f.readline()
            while c_page:
                c_page = json.loads(c_page)
                yield c_page, seek_pos

                seek_pos = f.tell()
                c_page = f.readline()


    def iter_pages(self, pages_per_file=0):
        
        # return a generator, next() to get next page info
        ''' 
        pages_per_file: #pages to store in each file (in case the index is larger than memory). default is 0 means all in a single file
        '''
        # don't use defaultdict, it can not be pickled
        word_index = dict()   # here i use word.t to represent word in title, fist list of tuple is doc id list, sencond list of tuple is frequency list

        for c_page, seek_pos in tqdm(self._readlines(), total=page_total):
            word_counter = defaultdict(lambda:0)
            self.page_positions[self.page_count] = seek_pos
            self.page_len = 0

            # make index for title
            title = c_page["title"]
            title_list = self.preprocess_sen(title)
            for word in title_list:
                word_counter[word] += 1
            for word in word_counter:
                tmp = word + ".t"
                if tmp in word_index:
                    word_index[tmp][0].append(self.page_count)    # add doc id
                    word_index[tmp][1].append(word_counter[word])   # add word freq
                else:
                    word_index[tmp] = (array("L", [self.page_count]), array("L", [word_counter[word]]))
                    # word_index[tmp] = ([self.page_count], [word_counter[word]])

            # make index for text
            text = c_page["text"]
            word_counter = defaultdict(lambda:0)
            text_list = text.split("\n")
            for item in text_list:
                if item:
                    if item.startswith("Section:"):
                        item = item[11:]
                    item_list = self.preprocess_sen(item)
                    self.page_len += len(item_list)
                    for word in item_list:
                        word_counter[word] += 1
            for word in word_counter:
                if word in word_index:
                    word_index[word][0].append(self.page_count)
                    word_index[word][1].append(word_counter[word])
                else:
                    word_index[word] = (array("L", [self.page_count]), array("L", [word_counter[word]]))
                    # word_index[word] = ([self.page_count], [word_counter[word]])
            
            # whether use multi files to store index
            # if pages_per_file > 0 and self.page_count >= pages_per_file:
            #     yield word_index
            #     word_index = defaultdict(lambda:(array("L",[]),array("L",[])))
            self.page_len_list.append(self.page_len)

            sum_v = 0
            for v in word_counter.values():
                sum_v += v ** 2
            sum_v = sum_v ** 0.5
            word_counter["_sum"] = sum_v
            self.page_word_index.append(dict(word_counter))

            self.page_count += 1
            if self.debug and self.page_count >= 10000:
                break

        if not self.debug:
            assert page_total == self.page_count

        # yield word_index
        return word_index


    def save_dicts(self):
        dump_pickle(self.page_positions, self.page_positions_path)
        dump_pickle(self.page_len_list, self.page_len_path)
        dump_pickle(self.page_word_index, self.page_word_index_path)
        dump_json(self.stem_word_dict, self.stemmer_path, indent=4)


    def get_cfg(self):
        cfg = {}
        for k, v in self.__dict__.items():
            if type(v) == type("string") or type(v) == type(1):
                cfg[k] = v
        return cfg



class IndexMaker():
    # A class to make index
    def __init__(self, data_parser, output_path, pages_per_file=0):
        self.data_parser = data_parser
        self.output_path = output_path
        self.index_path = output_path + "/index.pickle"
        
        self.page_count = 0     # wiki pages

        self.vocab_path = output_path + "/vocab.json"   # use pickle if store file position
        self.vocab = defaultdict(lambda:0)  # this vocab just stores frequency of words, if the index is too large and needs to store in disk, use (frequency, file_postition)

        # index on disk
        self.word_positions_path = output_path + "/word_positions.pickle"
        self.pages_per_file = pages_per_file
        self.file_count = 0     # index files


    def _write_file(self, index_data):
        '''
        index_data: data to write
        dst: distinguish used in the file name
        '''
        file = self.output_path + "/" + str(self.file_count) + ".txt"
        pass    # change index to text and save


    def _merge_index(self):
        pass


    def _update_vocab(self, index):
        for word in index:
            if word.endswith(".t"):
                continue 
            (_, fl) = index[word]
            for fre in fl:
                self.vocab[word] += fre


    def make_index_large(self):
        for one_index in self.data_parser.iter_pages(self.pages_per_file):
            self._update_vocab(one_index)
            self._write_file(one_index)
            self.file_count += 1

        self.data_parser.save_page_position()
        self.page_count = self.data_parser.page_count

        dump_json(self.vocab, self.vocab_path)
        self._merge_index()


    def make_index(self):
        one_index = self.data_parser.iter_pages(self.pages_per_file)
        self._update_vocab(one_index)
        dump_pickle(one_index, self.index_path)
        # dump_json(one_index, self.index_path)

        self.data_parser.save_dicts()
        self.page_count = self.data_parser.page_count

        dump_json(self.vocab, self.vocab_path, indent=4)


    def save_cfg(self, path):
        cfg = self.data_parser.get_cfg()
        for k, v in self.__dict__.items():
            if type(v) == type("string") or type(v) == type(1):
                cfg[k] = v
        dump_json(cfg, path, indent=4)


def update_page_word_index():
    # used for VSM tfidf 
    index_cfg = load_json(settings.cfg_path)
    doc_word_index = load_pickle(index_cfg["page_word_index_path"]) # list
    page_count =index_cfg["page_count"]
    doc_len = load_pickle(index_cfg["page_len_path"])
    index = defaultdict(lambda:[[],[]], load_pickle(index_cfg["index_path"]))

    for i, x in enumerate(doc_word_index):
        sum_v = 0
        for k, v in x.items():
            sum_v += (v * log10(page_count / (len(index[k][0]) + 1))) ** 2
        sum_v = (sum_v / doc_len[i]) ** 0.5
        x["_sum_tfidf"] = sum_v
    dump_pickle(doc_word_index, index_cfg["page_word_index_path"])
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir", "-i", default="data/processed/wiki_00", type=str, help="The input data dir. Should contain the .xml file")
    parser.add_argument("--output_dir", "-o", default="data/index_test", type=str, help="The output data folder. Save index")
    parser.add_argument("--debug", "-d", action='store_true', help="use 10,000 pages to debug")
    args = parser.parse_args()

    # make dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    start = time.time() # time start

    wiki_parser = WikiParser(args.input_dir, args.output_dir, args.debug)
    index_maker = IndexMaker(wiki_parser, args.output_dir)
    index_maker.make_index()

    # save index info
    index_maker.save_cfg(settings.cfg_path)

    # update page word index
    update_page_word_index()

    end = time.time()   # time end
    print("Time taken - " + str(end - start) + " s")
