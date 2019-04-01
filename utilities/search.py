from elasticsearch import Elasticsearch

def elastic_search_papers(query, num_results=10, from_result=0):
    client = Elasticsearch()

    response = client.search(
        index="arxiv_papers",
        from_=from_result,
        size=num_results,
        body={
            "query": {
                "bool": {
                    "should": [
                        {"match": {
                            "title": query
                        }},
                        {"match": {
                            "full_text": query
                        }},
                        {"match": {
                            "abstract": query
                        }}]
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
