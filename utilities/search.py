from elasticsearch import Elasticsearch

def elastic_search_papers(query, num_results=10):
    client = Elasticsearch()

    # "content": {
    #     "query": "Elasticsearch",
    #     "boost": 3
    # }

    response = client.search(terminate_after=num_results,
        index="arxiv_papers",
        body={
            "query": {
                "bool": {
                    "should": [
                        {"match": {
                            "title": {
                                "query" : query,
                                "boost": 5
                                }
                        }
                        },
                        {"match":  {
                            "full_text": {
                                "query" : query,
                                "boost": 1
                                }
                        }
                        },
                        {"match":  {
                            "abstract": {
                                "query" : query,
                                "boost": 2
                                }
                        }
                        }]
                }
            }
        }
    )
    response_obj = []
    for hit in response['hits']['hits']:
        data = {'paper_id': hit['_id'],
         'title': hit['_source']['title'],
         'abstract': hit['_source']['abstract'],
         'authors': hit['_source']['authors'],
         'date': hit['_source']['date'],
         'distance': round(hit['_score'], 4)
         }
        response_obj.append(data)


    return response_obj