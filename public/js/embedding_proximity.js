$(function () {
        var tableData, context;
        var getSimilarWords = function(searchTerm, embeddingType) {
            $.get('/word_embedding_proximity', {'word': searchTerm, 'type': embeddingType}, function(data) {
                console.log(data);
                if (data == 'Word not found') {
                    tableData = [{'distance': -1, 'word': 'Word not found in embedding labels'}]
                }
                else {
                    tableData = data;
                }
                context = {
                    'top_words': tableData,
                    'selectedWord': searchTerm
                }
                fillTable(context);
            })
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

        //getSimilarWords();


        $(document).bind('keypress', function(e) {
            if(e.keyCode==13) {
                $('#submit-btn').trigger('click');
            }
        });

        $('#submit-btn').click(function (data) {
            var searchTerm = $('#search-box').val();
            var embeddingType = $("#embedding-type").val();
            getSimilarWords(searchTerm, embeddingType);
        });

        var typeahead_data_to_get = ['gensim'] // fasttext
        var typeahead_labels = {}

        for (let i = 0; i < typeahead_data_to_get.length; i++) {
            $.get( "get_embedding_labels", function( data ) {
              var labels = data
              typeahead_labels[typeahead_data_to_get[i]] = labels
              console.log(typeahead_labels);
              if (typeahead_data_to_get[i] == 'gensim') {
                  $("#search-box").autocomplete({
                      //source: typeahead_labels['gensim'],
                      source: function(request, response) {
                            var results = $.ui.autocomplete.filter(typeahead_labels['gensim'], request.term);

                            response(results.slice(0, 20));
                      },
                      minLength: 2
                  });
              }
            });
        }

        // todo when two types and select changes https://stackoverflow.com/questions/18441716/jquery-autocomplete-change-source
    });
