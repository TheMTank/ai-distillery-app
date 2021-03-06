"""
Used to create the data for static front end with Chart.js
"""

from collections import Counter
import ast
from pprint import pprint
import json

all_paper_objs = []
all_paper_objs_no_references_or_citations = []

with open('../data/semantic-scholar/ids.txt', 'r') as f:
    ids = [x.strip() for x in f.readlines()]

# with open('../data/semantic-scholar/semantic-scholar.txt', 'r') as f:
#     lines = f.readlines()
#     print(len(lines))
#     #obj = json.loads(lines[0]) # not fully json so we need ast
#
#     for line in lines:
#         obj = ast.literal_eval(line)
#
#         obj_no_references_or_citations = {key:val for key, val in obj.items()
#                                           if key not in ['citations', 'references']}
#
#         # ids['arxiv-search-id'] # todo add
#         # all_paper_objs.append(obj)
#         all_paper_objs_no_references_or_citations.append(obj_no_references_or_citations)
#
# with open('../data/semantic-scholar/semantic-scholar-no-references-or-citations.json', 'w') as fp:
#     json.dump(all_paper_objs_no_references_or_citations, fp)
# with open('../data/semantic-scholar/semantic-scholar.json', 'w') as fp:
#     json.dump(all_paper_objs, fp)

# above was to create the object using ast for the first time and save into full json, later we comment out and just read data
with open('../data/semantic-scholar/semantic-scholar.json', 'r') as fp:
    all_paper_objs = json.load(fp)

all_paper_objs = [paper for paper in all_paper_objs if paper]
print('First paper object:')
pprint(all_paper_objs[0])
print('Num paper objects: {}'.format(len(all_paper_objs)))

def get_all_unique_author_counts(all_paper_objs):
    authors_flattened = [author['name'] for paper in all_paper_objs for author in paper['authors']]
    print('Num authors: {}'.format(len(authors_flattened)))
    print('authors: {}'.format(authors_flattened[0:20]))
    c = Counter(authors_flattened)

    top_100_authors = c.most_common(100)
    for i in top_100_authors:
        print(i)

    labels = [x[0] for x in top_100_authors]
    data = [x[1] for x in top_100_authors]
    print(labels)
    print(data)

def get_most_cited_papers(all_paper_objs):
    paper_citation_info = [(paper['title'], len(paper['citations']), paper['influentialCitationCount']) for paper in all_paper_objs]
    paper_citation_info = sorted(paper_citation_info, key=lambda x: x[1], reverse=True)
    for p_c_i in paper_citation_info[0:100]:
        print(p_c_i)

    top_100 = paper_citation_info[0:100]

    labels = [x[0] for x in top_100]
    data = [x[1] for x in top_100]
    print(labels)
    print(data)

def get_most_cited_papers_over_time(all_paper_objs):
    paper_citation_info = [
        (paper['title'], len(paper['citations']), paper['influentialCitationCount'], paper['year']) for paper in
        all_paper_objs]
    paper_citation_info = sorted(paper_citation_info, key=lambda x: x[1], reverse=True)
    for p_c_i in paper_citation_info[0:100]:
        print(p_c_i)

    top_100 = paper_citation_info[0:100]

    labels = [x[0] for x in top_100]
    data = [x[1] for x in top_100]
    print(labels)
    print(data)

    # todo normalise between 30-70
    x_y_r = [{'x': x[0], 'y': x[1], 'r': x[1] // 40} for x in top_100 if int(x[3]) >= 2014]
    print(x_y_r)

    x_y_r = [{'x': x[3], 'y': x[1], 'r': x[1] // 40} for x in top_100 if int(x[3]) >= 2014]
    print(x_y_r)

    print([x[3] for x in top_100 if int(x[3]) >= 2014])
    print([x[0] for x in top_100 if int(x[3]) >= 2014])
    print([x[1] for x in top_100 if int(x[3]) >= 2014])
    print([x[2] for x in top_100 if int(x[3]) >= 2014])

def get_most_used_topics(all_paper_objs):
    paper_citation_info = [{'title': paper['title'],
                            'citations': len(paper['citations']),
                            'influentialCitationCount': paper['influentialCitationCount'],
                            'year': paper['year'],
                            'topics': paper['topics']} for paper in all_paper_objs]

    all_topics = []
    for pci in paper_citation_info:
        for topic in pci['topics']:
            all_topics.append(topic['topic'])

    c = Counter(all_topics)

    top_100_topics = c.most_common(100)
    for i in top_100_topics:
        print(i)

    labels = [x[0] for x in top_100_topics]
    data = [x[1] for x in top_100_topics]
    print(labels)
    print(data)

    num_topics_altogether = len(c)
    top_all_topics = c.most_common(num_topics_altogether)
    for i in top_all_topics:
        print(i)

    labels = [x[0] for x in top_all_topics]
    data = [x[1] for x in top_all_topics]
    print(labels)
    print(data)

    print('Number of topics: {}'.format(num_topics_altogether))

# get_most_cited_papers(all_paper_objs)
# get_all_unique_author_counts(all_paper_objs)
# get_most_cited_papers_over_time(all_paper_objs)
get_most_used_topics(all_paper_objs)



# expected javascript data (bar chart) in form below
# javascript_data =
# {"labels":["January","February","March","April","May","June","July"]
# new Chart(document.getElementById("chartjs-1"),{"type":"bar","data":{"labels":["January","February","March","April","May","June","July"],
#         "datasets":[{"label":"My First Dataset","data":[65,59,80,81,56,55,40],"fill":false,
# "backgroundColor":["rgba(255, 99, 132, 0.2)","rgba(255, 159, 64, 0.2)","rgba(255, 205, 86, 0.2)","rgba(75, 192, 192, 0.2)","rgba(54, 162, 235, 0.2)","rgba(153, 102, 255, 0.2)","rgba(201, 203, 207, 0.2)"],"borderColor":["rgb(255, 99, 132)","rgb(255, 159, 64)","rgb(255, 205, 86)","rgb(75, 192, 192)","rgb(54, 162, 235)","rgb(153, 102, 255)","rgb(201, 203, 207)"],"borderWidth":1}]},
#                                                 "options":{"scales":{"yAxes":[{"ticks":{"beginAtZero":true}}]}}});

# bubble expected:
# "data":[{"x":20,"y":30,"r":15},{"x":40,"y":10,"r":10}]
