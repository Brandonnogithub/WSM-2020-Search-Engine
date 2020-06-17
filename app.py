import os
from urllib import parse

from flask import Flask, request, render_template

# init flask app and env variables
app = Flask(__name__)
host = os.getenv("HOST")
port = os.getenv("PORT")


@app.route("/", methods=['GET'])
def search():
    """
    URL : /
    Query engine to find a list of relevant URLs.
    Method : POST or GET (no query)
    Form data :
        - query : the search query
        - hits : the number of hits returned by query
        - start : the start of hits
    Return a template view with the list of relevant URLs.
    """
    # GET data
    query = request.args.get("query", None)
    start = request.args.get("start", 0, type=int)
    hits = request.args.get("hits", 10, type=int)
    if start < 0 or hits < 0:
        return "Error, start or hits cannot be negative numbers"

    if query:
        # query search engine
        # try :
        #     r = requests.post('http://%s:%s/search'%(host, port), data = {
        #         'query':query,
        #         'hits':hits,
        #         'start':start
        #     })
        # except :
        #     return "Error, check your installation"

        # get data and compute range of results pages
        # data = r.json()
        # print(data)
        data = {
            "total": 20,
            "results": [
                {
                    "title": "amd Yes " + str(i + 1),
                    "description": "des sha" * 100,
                    "url": "https://www.baidu.com",
                    "thumbnail": "static/spatial/images/pic07.jpg",
                }
                for i in range(20)
            ],
        }
        i = int(start / hits)
        maxi = 1 + int(data["total"] / hits)
        range_pages = range(i - 5, i + 5 if i + 5 < maxi else maxi) if i >= 6 else range(0, maxi if maxi < 10 else 10)

        # show the list of matching results
        return render_template('spatial/index.html', query=query,
                               # response_time=r.elapsed.total_seconds(),
                               response_time=10.56,
                               total=data["total"],
                               hits=hits,
                               start=start,
                               range_pages=range_pages,
                               results=data["results"],
                               page=i,
                               maxpage=maxi - 1)

    # return homepage (no query)
    return render_template('spatial/index.html')


# -- JINJA CUSTOM FILTERS -- #

@app.template_filter('truncate_title')
def truncate_title(title):
    """
    Truncate title to fit in result format.
    """
    return title if len(title) <= 70 else title[:70] + "..."


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
    app.run(host='127.0.0.1', port=5000)
