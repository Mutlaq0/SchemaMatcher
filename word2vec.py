import gensim, logging
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = [['name','of','player'],
['abbreviated','name','team'],['age','of','player'],['height','player'],
['weight','of','player'],['draft','number'],['gp','games','played'],
['reb','average','rebounds'],['average','assists','ast'],
['wage','salary'],['ppg','points','game'],['apg','assist','per','game'],
['rpg','rebounds','game'],['per','player','effeciency','rating'],['mpg','minutes','played','game'],
['fga','field','goal','attempt'],['fgm','field','goal','made'],['fgp','field','goal','percentage'],
['thp','throw','power'],['ftm','free','throws','mad'],['ftp','free','throws','percentage'],
['blkpg','blockings','game'],['stlpg','steals','game'],['pts','points','goal','percentage'],
['player_name'],['team_abbreviation'],['age'],['player_height'],['player_weight'],['college'],
['country'],['draft_year'],['draft_round'],['draft_number'],['gp'],['pts'],['reb'],['ast'],['net_rating'],
['oreb_pct'],['dreb_pct'],['usg_pct'],['ts_pct'],['ast_pct'],['season'],['team','of', 'player'],['name','of', 'player'],
['experience'],['url'],['position','of', 'player'],['age','of', 'player'],['ht','height'],['wt','weight'],['college'],['salary','wage'],['ppg_last_season','points','game'],
['apg_last_season','assists','game','last', 'season'],['rpg_last_season','rebound','game','last','season'],['per_last_season','player' ,'effeciency' ,'rating' ,'last' ,'season'],
['ppg_careerr','average','points','game', 'career'],['apg_career','average','assists', 'game', 'career'],['rgp_career','average','rebounds','game'],['gp','games','played'],
['mpg_career','minutes','played','game'],['fgm_fga'],['fgp'],['tm_tha'],['thp'],['ftm_fta'],['ftp'],['apg'],['blkpg','blocks','game'],['topg'],['ppg','average','points', 'game']]
# train word2vec on the two 'sentences'

def model_w2v():
    model = gensim.models.Word2Vec(sentences,min_count=1)
    return  model



#print(model.wv.vocab)
def build_voc(sents):
    model = gensim.models.Word2Vec.load('word2vec.model')
    model.build_vocab(sents, update=True)
    model.train(sents, total_examples=model.corpus_count, epochs=model.epochs)
    model.save('word2vec.model')
    return