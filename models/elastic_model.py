from elasticsearch_dsl import Document, Date, Keyword, Text

class Paper(Document):
    """
    Paper object inside elasticserach
    """
    title = Text(analyzer='snowball', fields={'raw': Keyword()})
    abstract = Text(analyzer='snowball')
    full_text = Text(analyzer='snowball')
    authors = Text(analyzer='snowball')
    date = Date()

    class Index:
        name = 'arxiv_papers'
        settings = {
          "number_of_shards": 2,
        }

    def save(self, ** kwargs):
        return super(Paper, self).save(** kwargs)
