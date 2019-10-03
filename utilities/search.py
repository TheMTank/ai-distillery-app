from elasticsearch import Elasticsearch

def elastic_search_papers(query, num_results=10, twitter_popularity=False, from_result=0):
    client = Elasticsearch()
    if twitter_popularity == "true":
        response = client.search(
            index="arxiv_papers",
            from_=from_result,
            size=num_results,
            body={
                	"sort": [{
                		"twitter_popularity": {
                			"order": "desc"
                		}
                	}],
                	"query": {
                		"bool": {
                			"should": [{
                					"match": {
                						"title": query
                					}
                				},
                				{
                					"match": {
                						"full_text": query
                					}
                				},
                				{
                					"match": {
                						"abstract": query
                					}
                				}
                			],
                			"must": [{
                				"range": {
                					"twitter_popularity": {
                						"gte": 2
                					}
                				}
                			}]
                		}
                	}
                }
        )
    else:
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
         'distance': hit["_source"]["twitter_popularity"] if twitter_popularity == "true" else round(hit['_score'], 4)
         }
        response_obj.append(data)


    return response_obj
