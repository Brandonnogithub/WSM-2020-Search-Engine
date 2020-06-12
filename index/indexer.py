import operator
import os
import time
import xml.etree.cElementTree as et
import re
import sys
import pickle
import base64
import nltk
import argparse
from math import *
from heapq import *
from collections import *
from utils.io_utils import load_json


class WikiParser():
    # a class to parser wiki dataset
    def __init__(self, wiki_path):
        self.wiki_path = wiki_path
        self.STOP_WORDS_PATH = os.path.json("index", "stop_words.json")
        self.stemmer = nltk.stem.SnowballStemmer('english')  # You can use porter or other stemmer
        self.stem_word_dict = dict()
        self.stop_words = load_json(STOP_WORDS_PATH)["stop_words"]
        
        # RE to parser xml file
        self.regExp1 = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',re.DOTALL)   # RE to remove urls
        self.regExp2 = re.compile(r'{\|(.*?)\|}',re.DOTALL)             # RE to remove tags & css
        self.regExp3 = re.compile(r'{{v?cite(.*?)}}',re.DOTALL)         # Regular Expression to remove {{cite **}} or {{vcite **}}
        self.regExp4 = re.compile(r'\[\[file:(.*?)\]\]',re.DOTALL)      # Regular Expression to remove [[file:]]
        self.pattern = re.compile("[^a-zA-Z0-9]")                       # pattern to get only alphnumeric text
        self.categoty_re = "\[\[Category:(.*?)\]\]"
        self.infobox_re="{{Infobox((.|\n)*?)}}"

        self.file_count = 0
        self.pages_per_file = 40000
        self.page_count = 0 # denote which number of wiki page it is

        context = et.iterparse(wiki_path, events=("start", "end"))
        self.context = iter(context)


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



class IndexManager():
    # A class to make index or load index
    def __init__(self, data_parser, index_path):
        self.data_parser = data_parser
        self.index_path = index_path
        self.title_tags = open(index_path+"/title_tags.txt", "w+")

        # Defaut list of title,text,infobox,output_index.........inverted index
        self.title_index = defaultdict(list)
        self.text_index = defaultdict(list)
        self.category_index = defaultdict(list)
        self.infobox_index = defaultdict(list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir", "-i", default="data/source/enwiki-latest-pages-articles26.xml", type=str, help="The input data dir. Should contain the .xml file")
    parser.add_argument("--output_dir", "-o", default="data/index", type=str, help="The output data folder. Save index")
    args = parser.parse_args()

    start = time.time()



output_files = list()
title_position = list()
word_position = dict()



for event, elem in context :
    tag =  re.sub(r"{.*}", "", elem.tag)

    if event == "start" :
        if tag == "page" :
            title_tag_words =  dict()
            infobox_words =  dict()
            text_tag_words =  dict()
            category_words =  dict()
            page_count = page_count + 1
           
    if event == "end" :
        
        if tag == "text" :
            text = str(elem.text)
            text = regExp1.sub('',text)
            text = regExp2.sub('',text)
            text = regExp3.sub('',text)
            text = regExp4.sub('',text)
            try :
                
                tempword = re.findall(categoty_re, text); # get all data between [[Category : ----- ]]
                if tempword :
                    for temp in tempword :
                        temp = re.split(pattern, temp);#print(pattern)
                        for t in temp :
                            t = preprocee_word(t)
                            if t :
                                if len(t) <= 2 :
                                    continue
                                if  t not in stop_words :
                                    if t not in category_words:
                                        category_words[t] = 1
                                    else : 
                                        category_words[t] += 1

                tempword = re.findall(infobox_re, text) # get all data between infobox{{ ----- }}
                if tempword :
                    for temp in tempword :
                        for word in temp : 
                            temp = re.split(pattern, word);#print(pattern)
                            for t in temp :
                                t = preprocee_word(t)
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
                text = text.lower();
                text = re.split(pattern, text);

                for word in text :
                    if word :
                        if word not in stem_word_dict :
                            stem_word = stemmer.stem(word) # do stemming
                            stem_word_dict[word]=stem_word
                        else :
                            stem_word = stem_word_dict[word]
                        word = stem_word
                        if word not in stop_words :
                            if len(word) <= 2 :
                                    continue
                            if word not in text_tag_words :
                                text_tag_words[word] = 1
                            else :
                                text_tag_words[word] += 1

            except :
                pass     
        if tag == "title" :
            text = elem.text;
            try :
                title_string = text
                title_position.append(title_tags.tell())
                
                text = text.lower();
                title_tags.write( title_string+"\n")
                text = re.split(pattern, text);

                for word in text :
                    if word :
                        if word not in stem_word_dict :
                            stem_word = stemmer.stem(word) # do stemming
                            stem_word_dict[word]=stem_word
                        else :
                            stem_word = stem_word_dict[word]
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

            doc_id = str(page_count) # get document ID ==> Wiki page number
            
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
               
            if page_count % 50000 == 0 :
                stem_word_dict = {}
                
            if page_count % pages_per_file == 0 :
                
                file = index_path + "/t" + str(file_count) + ".txt"
                outfile = open(file, "w+")
                for word in sorted(title_index) :
                    posting_list = ",".join(title_index[word])
                    index = word + "-" + posting_list
                    outfile.write(index+"\n")
                outfile.close()

                file = index_path + "/b" + str(file_count) + ".txt"
                outfile = open(file, "w+")
                for word in sorted(text_index) :
                    posting_list = ",".join(text_index[word])
                    index = word + "-" + posting_list
                    outfile.write(index+"\n")
                outfile.close()

                file = index_path + "/c" + str(file_count) + ".txt"
                outfile = open(file, "w+")
                for word in sorted(category_index) :
                    posting_list = ",".join(category_index[word])
                    index = word + "-" + posting_list
                    outfile.write(index+"\n")
                outfile.close()

                file = index_path + "/i" + str(file_count) + ".txt"
                outfile = open(file, "w+")
                for word in sorted(infobox_index) :
                    posting_list = ",".join(infobox_index[word])
                    index = word + "-" + posting_list
                    outfile.write(index+"\n")
                outfile.close()

                
                title_index.clear()
                text_index.clear()
                file_count = file_count + 1
                category_index.clear()
                infobox_index.clear()


        elem.clear()


# In[17]:


file = index_path + "/t" + str(file_count) + ".txt"
outfile = open(file, "w+")
for word in sorted(title_index) :
    posting_list = ",".join(title_index[word])
    index = word + "-" + posting_list
    outfile.write(index+"\n")
outfile.close()

file = index_path + "/b" + str(file_count) + ".txt"
outfile = open(file, "w+")
for word in sorted(text_index) :
    posting_list = ",".join(text_index[word])
    index = word + "-" + posting_list
    outfile.write(index+"\n")
outfile.close()

file = index_path + "/c" + str(file_count) + ".txt"
outfile = open(file, "w+")
for word in sorted(category_index) :
    posting_list = ",".join(category_index[word])
    index = word + "-" + posting_list
    outfile.write(index+"\n")
outfile.close()

file = index_path + "/i" + str(file_count) + ".txt"
outfile = open(file, "w+")
for word in sorted(infobox_index) :
    posting_list = ",".join(infobox_index[word])
    index = word + "-" + posting_list
    outfile.write(index+"\n")
outfile.close()

file_count = file_count + 1

""" all intermediate t_1, t_2 file storing 
    
    word1 - doc_id1 : freq, doc_id2 : freq
    word2 - doc_id2 : freq, doc_id3 : freq
    
    all these words are in sorted order.
    for each word --> doc_id already in soreted order.. as we travsering document in an increasing doc_id only.
"""


# In[18]:


t_file = index_path+"/title_positions.pickle"
file = open(t_file, "wb+")
pickle.dump(title_position, file)
file.close()


# In[19]:


word_position = dict() # store word & its occurence/file pointer in title file, infobox file, body file
# abc word : { { t : fpt1_val}, { b : fpt2_val}, { c : fpt3_val}, { b : fpt4_val} }


# In[20]:


field_chars = ["t", "b", "i", "c"]

for f in field_chars :
    heap = []
    flag1 = 1 ;
    input_files = []
    file = index_path + "/" + f + ".txt"
    fp = open(file, "w+")
    output_files.append(fp)
    outfile_index = len(output_files) - 1

    for i in range(file_count) :
        file =  index_path + "/" + f + str(i) + ".txt"
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
        flag1 = 0;
        break

    for i in range(file_count) :
        try :
            s = input_files[i].readline()[:-1]
            heap.append((s, i))
        except :
            flag1 = 0;
        
    i = 0
    heapify(heap)

    try :
        while i < file_count :
            s, index = heappop(heap)
            word = s[: s.find("-")]
            posting_list = s[s.find("-") + 1 :]

            next_line = input_files[index].readline()[: -1]
            if next_line :
                heappush(heap, (next_line, index))
            else :
                i = i + 1 # one files ends here

            if i == file_count :
                flag1 = 0;
                break

            while i < file_count :
                
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
            idf = log10(page_count / len(postings))
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
        for i in range(file_count) :
            file = index_path + "/" + f + str(i) + ".txt"
            input_files[i].close()
            os.remove(file)
    except :
        pass


# In[21]:


end = time.time()

print("Time taken - " + str(end - start) + " s")


# In[22]:


file = open(index_path+"/word_positions.pickle", "wb+")
pickle.dump(word_position, file)
file.close()


# In[23]:


# print(word_position)

