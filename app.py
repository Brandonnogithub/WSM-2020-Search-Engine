import os
import math
import re
import time
import argparse
import settings
from urllib import parse
from flask import Flask, request, render_template
from search_engine import SearchEngine
import nltk


# init flask app and env variables
app = Flask(__name__)
host = os.getenv("HOST")
port = os.getenv("PORT")

# init search engine
cfg_path = settings.cfg_path
ranker_name_list = ["base", "Tfidf", "bm25", "VSM-tf", "VSM-tfidf", "SLM-A", "SLM-D"]
checked = ["", "checked='true", "", "", "", "", ""]
selected = 1
only_title = None
sg = SearchEngine(cfg_path, ranker_name_list[selected])

hits = 10
num_ra = 7
max_show_pages = 5


@app.route("/")
def main():
    return render_template("main.html")


@app.route("/search/", methods=['GET'])
def search():
    # try:
    """
        TODO: excute search
    """
    global checked
    global range_pages
    global query
    global doc_id
    global maxi
    global selected
    global only_title
    
    # GET data
    t_start = time.time()
    flag_only_title = False
    query = request.args.get("query", None)
    try:
        selected = int(request.args.get('order')) - 1
        for i in range(num_ra):
            if i == selected:
                checked[i] = 'checked="true"'
            else:
                checked[i] = ''
        
        only_title = request.args.get('onlytitle')
        if only_title:
            only_title = "checked"
            flag_only_title = True
    except:
        pass
    # print(request.args.get('order'), only_title)
    sg._start_ranker(ranker_name_list[selected])
    doc_id_score = sg.query(query, only_title=flag_only_title)
    doc_id = [ele[0] for ele in doc_id_score]
    doc_num = len(doc_id)
    if doc_num == 0:
        matched = False
    else:
        matched = True

    maxi = math.ceil(doc_num / hits)
    range_pages = range(1, maxi+1 if maxi<=max_show_pages else max_show_pages+1)
    first_page_results = cut_page(0)
    response_time = round(time.time() - t_start, 4)

    # print(first_page_results[0].keys())

    # show the list of matching results
    return render_template('index.html', query=query,
                            response_time=response_time,
                            total=doc_num,
                            range_pages=range_pages,
                            results=first_page_results,
                            page=1,
                            maxpage=maxi,
                            checked=checked,
                            only_title=only_title,
                            matched=matched)    
    # except:
    #     print('search error')

@app.route('/search/pages/0/', methods=['GET'])
def next_page():
    global range_pages
    t_start = time.time()
    current_page = int(request.args.get("current_page"))
    next_result = cut_page(current_page-1)
    response_time = round(time.time()-t_start, 4)
    print("before page range: ", range_pages, current_page)
    if current_page > range_pages[-1]:
        start_page = current_page
        end_page = start_page+max_show_pages if start_page+max_show_pages < maxi else maxi+1
        range_pages = range(start_page, end_page)
    elif current_page < range_pages[0]:
        start_page = current_page-max_show_pages+1 if current_page-max_show_pages+1 > 1 else 1
        end_page = start_page+max_show_pages
        range_pages = range(start_page, end_page)
    print("after page range: ", range_pages)

    return render_template('index.html', query=query,
                        response_time=response_time,
                        total=len(doc_id),
                        range_pages=range_pages,
                        results=next_result,
                        page=current_page,
                        maxpage=maxi,
                        checked=checked,
                        only_title=only_title,
                        matched=True)  


@app.route('/search/pages/1/', methods=['GET'])
def show_content():
    real_doc_id = int(request.args.get('real_id'))
    rst = read_doc_content([real_doc_id])[0]
    rst['text1'] = re.split("\n|Section:::", rst['text'])[1:]
    return render_template('content.html', doc=rst) 


def cut_page(start):
    sub_doc_id = doc_id[start*hits:(start+1)*hits]
    return read_doc_content(sub_doc_id)


def read_doc_content(doc_id_list):
    docs = []
    for id_ in doc_id_list:
        doc = sg.get_docs(id_)
        # with open(input_dir, 'r', encoding="utf8") as f:
        #     f.seek(sg.page_positions[id_], 0)
        #     line = eval(f.readline())
        doc['real_id'] = id_
        docs.append(doc)

    return docs



# -- JINJA CUSTOM FILTERS -- #

@app.template_filter('truncate_title')
def truncate_title(title):
    """
    Truncate title to fit in result format.
    """
    return title if len(title) <= 70 else title[:70] + "..."

@app.context_processor
def utility_processor():
    def highlight(text, query):
        text.replace(query, "<mark>"+query+"</mark>")
        # text_list = text.split(' ')
        # for i, word in enumerate(text_list):
        #     if query in word:
        #         text0 = " ".join(text_list[:i])
        #         text1 = " ".join(text_list[i+1:])
        print(text)
        return text

    return dict(highlight=highlight)

@app.template_filter('truncate_description')
def truncate_description(description):
    """
    Truncate description to fit in result format.
    """
    if len(description) <= 160:
        return description

    cut_desc = ""
    character_counter = 0
    for i, letter in enumerate(description):
        character_counter += 1
        if character_counter > 160:
            if letter == ' ':
                return cut_desc + "..."
            else:
                return cut_desc.rsplit(' ', 1)[0] + "..."
        cut_desc += description[i]
    return cut_desc


@app.template_filter('truncate_url')
def truncate_url(url):
    """
    Truncate url to fit in result format.
    """
    url = parse.unquote(url)
    if len(url) <= 60:
        return url
    url = url[:-1] if url.endswith("/") else url
    url = url.split("//", 1)[1].split("/")
    url = "%s/.../%s" % (url[0], url[-1])
    return url[:60] + "..." if len(url) > 60 else url


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", "-l", action='store_true', help="run local")
    parser.add_argument("--port", "-p", default=5000, type=int, help="runnning port")
    args = parser.parse_args()

    if args.local:
        app.run(host='127.0.0.1', port=args.port)
    else:
        app.run(host='0.0.0.0', port=args.port)
