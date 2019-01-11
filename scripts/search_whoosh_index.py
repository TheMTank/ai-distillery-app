from whoosh.qparser import QueryParser
import whoosh.index as index

ix = index.open_dir("../indexdir_test")


with ix.searcher() as searcher:
    query = QueryParser("full_text", ix.schema).parse("visual speech recognition")
    # query = QueryParser("abstract", ix.schema).parse("trajectory")
    results = searcher.search(query)
    print(len(results))
    if len(results):
        for i in range(5):
            print(results[i])
            print(results[i]['title'])
            print(results[i]['abstract'])
            # print(results[i]['authors'])
            # print(results[i]['date'])
            print()
