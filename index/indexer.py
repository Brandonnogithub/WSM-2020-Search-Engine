import operator
import os
import time
import xml.etree.cElementTree as et
import re
import pickle
import nltk
import argparse
from math import log10
from heapq import heapify, heappop, heappush
from collections import defaultdict
from utils.io_utils import load_json


class WikiParser():
    # a class to parser wiki dataset
    def __init__(self, wiki_path):
        self.wiki_path = wiki_path
        self.STOP_WORDS_PATH = os.path.join("index", "stop_words.json")
        self.stemmer = nltk.stem.SnowballStemmer('english')  # You can use porter or other stemmer
        self.stem_word_dict = dict()
        self.stop_words = load_json(self.STOP_WORDS_PATH)["stop_words"]

        self.file_count = 0 # denote which number of wiki page it is


    def _preprocee_word(self, word):
        # return stemmed word and update stem word dict
        word = word.strip()
        word = word.lower() # convert into lower case
        if word not in self.stem_word_dict :
            stem_word = self.stemmer.stem(word) # do stemming
            self.stem_word_dict[word]=stem_word
        else :
            stem_word = self.stem_word_dict[word]
        return stem_word


    def iter_pages(self, pages_per_file):
        
        # return a generator, next() to get next page info
        ''' 
        pages_per_file: #pages to store in each file 
        '''
        # Defaut list of title,text,infobox,output_index.........inverted index
        title_index = defaultdict(list)
        text_index = defaultdict(list)
        category_index = defaultdict(list)
        infobox_index = defaultdict(list)

        for event, elem in self.context:
            tag =  re.sub(r"{.*}", "", elem.tag)

            # page tag start, initial all dicts and update counter
            if event == "start" :
                if tag == "page" :
                    title_tag_words =  dict()
                    infobox_words =  dict()
                    text_tag_words =  dict()
                    category_words =  dict()
                    self.page_count = self.page_count + 1

            # tag end
            if event == "end" :
                # text tag
                if tag == "text" :
                    text = str(elem.text)
                    text = self.regExp1.sub('',text)
                    text = self.regExp2.sub('',text)
                    text = self.regExp3.sub('',text)
                    text = self.regExp4.sub('',text)
                    try :
                        tempword = re.findall(self.categoty_re, text); # get all data between [[Category : ----- ]]
                        if tempword :
                            for temp in tempword :
                                temp = re.split(self.pattern, temp);#print(pattern)
                                for t in temp :
                                    t = self._preprocee_word(t)
                                    if t :
                                        if len(t) <= 2 :
                                            continue
                                        if  t not in self.stop_words :
                                            if t not in category_words:
                                                category_words[t] = 1
                                            else : 
                                                category_words[t] += 1

                        tempword = re.findall(self.infobox_re, text) # get all data between infobox{{ ----- }}
                        if tempword :
                            for temp in tempword :
                                for word in temp : 
                                    temp = re.split(pattern, word);#print(pattern)
                                    for t in temp :
                                        t = self._preprocee_word(t)
                                        if t :
                                            if len(t) <= 2 :
                                                continue
                                            if  t not in stop_words :
                                                if t not in infobox_words :
                                                    infobox_words[t] = 1
                                                else :
                                                    infobox_words[t] += 1
                    except :
                        pass

                    try :
                        text = text.lower()
                        text = re.split(pattern, text)

                        for word in text :
                            if word :
                                if word not in self.stem_word_dict :
                                    stem_word = self.stemmer.stem(word) # do stemming
                                    self.stem_word_dict[word]=stem_word
                                else :
                                    stem_word = self.stem_word_dict[word]
                                word = stem_word
                                if word not in self.stop_words :
                                    if len(word) <= 2 :
                                            continue
                                    if word not in text_tag_words :
                                        text_tag_words[word] = 1
                                    else :
                                        text_tag_words[word] += 1

                    except :
                        pass     
                if tag == "title" :
                    text = elem.text
                    try :
                        title_string = text
                        title_position.append(title_tags.tell())
                        
                        text = text.lower()
                        title_tags.write( title_string+"\n")
                        text = re.split(self.pattern, text)

                        for word in text :
                            if word :
                                if word not in self.stem_word_dict :
                                    stem_word = self.stemmer.stem(word) # do stemming
                                    self.stem_word_dict[word]=stem_word
                                else :
                                    stem_word = self.stem_word_dict[word]
                                word = stem_word
                                if word not in stop_words :
                                    if len(word) <= 2 :
                                        continue
                                    if word not in title_tag_words :
                                        title_tag_words[word] = 1
                                    else :
                                        title_tag_words[word] += 1

                    except :
                        pass

                # Posting list start
                if tag == "page" :

                    doc_id = str(self.page_count) # get document ID ==> Wiki page number
                    
                    for word in text_tag_words :
                        s = doc_id + ":" 
                        s = s + str(text_tag_words[word]); # doc_id  : frequency
                        text_index[word].append(s)

                    for word in infobox_words :
                        s = doc_id + ":" 
                        s = s + str(infobox_words[word])
                        infobox_index[word].append(s)
                    
                    for word in title_tag_words :
                        s = doc_id + ":" 
                        s = s + str(title_tag_words[word])
                        title_index[word].append(s)

                    for word in category_words :
                        s = doc_id + ":"
                        s = s + str(category_words[word])
                        category_index[word].append(s)
                    
                    if self.page_count % 50000 == 0 :
                        self.stem_word_dict = {}
                        
                    if self.page_count % pages_per_file == 0 :
                        yield title_index, text_index, category_index, infobox_index
                    
                        title_index.clear()
                        text_index.clear()
                        category_index.clear()
                        infobox_index.clear()

                elem.clear()

        yield title_index, text_index, category_index, infobox_index


class IndexMaker():
    # A class to make index
    def __init__(self, data_parser, index_path):
        self.data_parser = data_parser
        self.index_path = index_path
        self.title_tags_path = index_path + "/title_tags.txt"
        self.t_file = index_path + "/title_positions.pickle"
        self.word_positions_path = index_path + "/word_positions.pickle"

        self.pages_per_file = 40000
        self.file_count = 0
        self.page_count = 0

        self.title_dst = "t"
        self.text_dst = "b"
        self.category_dst = "c"
        self.infobox_dst = "i"
        self.field_chars = ["t", "b", "i", "c"]


    def _write_file(self, index_data, dst):
        '''
        index_data: data to write
        dst: distinguish used in the file name
        '''
        file = self.index_path + "/" + dst + str(self.file_count) + ".txt"
        outfile = open(file, "w+")
        for word in sorted(index_data) :
            posting_list = ",".join(index_data[word])
            index = word + "-" + posting_list
            outfile.write(index+"\n")
        outfile.close()


    def _get_source(self):
        title_tags = open(self.title_tags_path, "w+")
        title_position = list()

        for title_index, text_index, category_index, infobox_index in self.data_parser.iter_pages(title_position, title_tags, self.pages_per_file):
            # print(title_index)
            # print(text_index)
            # print(category_index)
            # print(infobox_index)
            # raise Exception()
            self._write_file(title_index, self.title_dst)
            self._write_file(text_index, self.text_dst)
            self._write_file(category_index, self.category_dst)
            self._write_file(infobox_index, self.infobox_dst)

            self.file_count += 1

        file = open(self.t_file, "wb+")
        pickle.dump(title_position, file)
        file.close()

        self.page_count = self.data_parser.page_count


    def _make_index(self):
        output_files = list()
        word_position = dict() # store word & its occurence/file pointer in title file, infobox file, body file
        # abc word : { { t : fpt1_val}, { b : fpt2_val}, { c : fpt3_val}, { b : fpt4_val} }

        for f in self.field_chars :
            heap = []
            flag1 = 1
            input_files = []
            file = self.index_path + "/" + f + ".txt"
            fp = open(file, "w+")
            output_files.append(fp)
            outfile_index = len(output_files) - 1

            for i in range(self.file_count) :
                file =  self.index_path + "/" + f + str(i) + ".txt"
                if os.stat(file).st_size == 0 :
                    try :
                        del input_files[i]
                        os.remove(file)
                    except :
                        pass
                else :
                    fp = open(file, "r")
                    input_files.append(fp)

            if len(input_files) == 0 :
                flag1 = 0
                break

            for i in range(self.file_count) :
                try :
                    s = input_files[i].readline()[:-1]
                    heap.append((s, i))
                except :
                    flag1 = 0
                
            i = 0
            heapify(heap)

            try :
                while i < self.file_count :
                    s, index = heappop(heap)
                    word = s[: s.find("-")]
                    posting_list = s[s.find("-") + 1 :]

                    next_line = input_files[index].readline()[: -1]
                    if next_line :
                        heappush(heap, (next_line, index))
                    else :
                        i = i + 1 # one files ends here

                    if i == self.file_count :
                        flag1 = 0
                        break

                    while i < self.file_count :
                        next_s, next_index = heappop(heap)
                        next_word = next_s[: next_s.find("-")]
                        next_posting_list = next_s[next_s.find("-") + 1 :]
                        if next_word == word :
                            posting_list = posting_list + "," + next_posting_list
                            next_new_line = input_files[next_index].readline()
                            if next_new_line :
                                heappush(heap, (next_new_line, next_index))
                            else : # one files ends here
                                i = i + 1
                        else :
                            heappush(heap, (next_s, next_index))
                            break

                    if word not in word_position :
                        word_position[word] = dict()
                    word_position[word][f] = output_files[outfile_index].tell()
                    postings = posting_list.split(",")
                    documents = dict()
                    idf = log10(self.page_count / len(postings))
                    for posting in postings :
                        doc_id = posting[ : posting.find(":")]
                        freq = int( posting[posting.find(":") + 1 :] )
                        tf = 1 + log10(freq)
                        documents[str(doc_id)] = round(tf*idf, 2)

                    documents = sorted(documents.items(), key = operator.itemgetter(1), reverse = True)
                    
                    top_posting_list_result = ""
        #             number_of_results = 1
                    for document in documents :
                        top_posting_list_result = top_posting_list_result + document[0] + ":" + str(document[1]) + ","
        #                 number_of_results = number_of_results + 1
        #                 if number_of_results > 10 :
        #                     break

                    top_posting_list_result = top_posting_list_result[ : -1 ] # to remove last extra comma ","
                    output_files[outfile_index].write( top_posting_list_result+"\n" )

            except IndexError :
                pass

            output_files[outfile_index].close()

            try :
                for i in range(self.file_count) :
                    file = self.index_path + "/" + f + str(i) + ".txt"
                    input_files[i].close()
                    os.remove(file)
            except :
                pass

        file = open(self.word_positions_path, "wb+")
        pickle.dump(word_position, file)
        file.close()

        print(word_position)


    def build(self):
        self._get_source()
        self._make_index()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir", "-i", default="data/processed/wiki_o1", type=str, help="The input data dir. Should contain the .xml file")
    parser.add_argument("--output_dir", "-o", default="data/index", type=str, help="The output data folder. Save index")
    args = parser.parse_args()

    start = time.time() # time start

    wiki_parser = WikiParser(args.input_dir)
    index_maker = IndexMaker(wiki_parser, args.output_dir)
    index_maker.build()

    end = time.time()   # time end
    print("Time taken - " + str(end - start) + " s")




""" all intermediate t_1, t_2 file storing 
    
    word1 - doc_id1 : freq, doc_id2 : freq
    word2 - doc_id2 : freq, doc_id3 : freq
    
    all these words are in sorted order.
    for each word --> doc_id already in soreted order.. as we travsering document in an increasing doc_id only.
"""
