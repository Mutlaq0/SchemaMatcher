# SchemaMatcher
 schema matching application - schema matching is a very interesting aspect of data integration.
 
The program has 2 python scripts: Matcher.py, word2vec.py
To use the program:
in matcher.py use the function called Match("first_dataset.csv","second_dataset.csv","exact_match.csv").
to use on our dataset : Match("all_seasons.csv","NBA_Players.csv","correspondence.csv")
NOTE: correspondencies has 'field1' and 'field2' as header, the first line must be : field1,field2
NOTE: I use gensim word2vec model, and polyglot in the system, make sure u have these packages downloaded to your python:

https://pypi.org/project/gensim/

https://polyglot.readthedocs.io/en/latest/Installation.html

http://www.nltk.org/install.html

## The program is not made to use, I wrote it for the full purpose of practice, feel free to use it and improve it.
