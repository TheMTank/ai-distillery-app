$(function () {
    var tableData, context;

    var getSimilarEmbeddings = function(searchTerm, embeddingType) {
        $.get(embeddingProximityRoute, {'input_str': searchTerm, 'type': embeddingType}, function(data) {
            console.log(data);
            if (data.length == 1) {
                tableData = [{'distance': -1, 'label': data[0]}] //'Word not found in embedding labels'}]
            }
            else {
                tableData = data;
            }
            context = {
                'top_words': tableData,
                'selectedWord': searchTerm
            }
            fillTable(context);
        }, "json")
    }

    var fillTable = function(context) {
        // Grab the template script
        var theTemplateScript = $("#address-template").html();

        // Compile the template
        var theTemplate = Handlebars.compile(theTemplateScript);

        // Pass our data to the template
        var theCompiledHtml = theTemplate(context);

        // Add the compiled html to the page
        $('.content-placeholder').html(theCompiledHtml);
    }

    $(document).bind('keypress', function(e) {
        if(e.keyCode==13) {
            $('#submit-btn').trigger('click');
        }
    });

    $('#submit-btn').click(function (data) {
        var searchTerm = $('#search-box').val();
        var embeddingType = $("#embedding-type").val();
        getSimilarEmbeddings(searchTerm, embeddingType);
    });

    // typeahead setup
    var currentPagesEmbeddingOptions = higherLevelEmbeddingType == 'word' ? ['gensim'] : ['lsa', 'doc2vec'];
    //var typeahead_data_to_get = currentPagesEmbeddingOptions // fasttext
    var typeahead_labels = {}

    for (let i = 0; i < currentPagesEmbeddingOptions.length; i++) {
        $.get( "get-embedding-labels", {'embedding_type': currentPagesEmbeddingOptions[i]}, function( data ) {
          var labels = data
          typeahead_labels[currentPagesEmbeddingOptions[i]] = labels
          console.log(typeahead_labels);
          if (i == 0) { // first one should be default
              currentEmbeddingSelected = currentPagesEmbeddingOptions[i];
              $('#search-box').val(typeahead_labels[currentPagesEmbeddingOptions[i]][Math.floor(Math.random() * typeahead_labels[currentPagesEmbeddingOptions[i]].length)]);
              $("#search-box").autocomplete({
                  //source: typeahead_labels['gensim'], // for all
                  source: function(request, response) {
                        var results = $.ui.autocomplete.filter(typeahead_labels[currentPagesEmbeddingOptions[i]], request.term);

                        response(results.slice(0, 20));
                  },
                  minLength: 2
              });
          }
        });
    }

    // if select embedding changes, get other typeahead and replace source
    $('#embedding-type').change(function() {
        console.log($(this).val())
        currentEmbeddingSelected = $(this).val()
        $( "#search-box" ).autocomplete('option', {'source': function(request, response) {
                        var results = $.ui.autocomplete.filter(typeahead_labels[currentEmbeddingSelected], request.term);

                        response(results.slice(0, 20));
                  }})
    });
});
