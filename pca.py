import matplotlib.pyplot as plt
import collections
import json

from sklearn import datasets
from sklearn.decomposition import PCA

from utils import read_conll

conll_path = 'data/english/train.conll'
emb_path= 'output/model_pos_embeddings.json'

pos_emb = []
word_emb = []
verb_emb =[] 

verb_list = []

#collect all verb from training conll dataset
for sentence in read_conll(conll_path):
    for node in sentence:
	if 'VB' in node.pos:
	    verb_list.append(node.norm)

#print 'verb_list length', len(verb_list)
 
#create PCA handler
pca = PCA(n_components=2) 

#read embedding data
with open(emb_path) as f:
    data = json.load(f)
    
    #convert word data to an ordered dict
    data['word'] = collections.OrderedDict(data['word'])
    for word in data['word']:
	word_emb.append(data['word'][word])
    print 'word_raw', len(word_emb), len(word_emb[0])

    X_word = pca.fit(word_emb).transform(word_emb)
    print 'word dim after PCA', len(X_word),len(X_word[0])
    
    X_word_x, X_word_y = [],[]
    for verb in verb_list:
	verb_emb = X_word[data['word'].keys().index(verb)]
	X_word_x.append(verb_emb[0])
	X_word_y.append(verb_emb[1])

    print len(X_word_x)    
    plt.figure()
    plt.plot(X_word_x,X_word_y,'s')


    #process POS data
    plt.figure()
    pos_list = []
    data['pos'] = collections.OrderedDict(data['pos'])
    for pos in data['pos']:
#	print pos, len(data['pos'][pos])   
	pos_emb.append(data['pos'][pos])
        pos_list.append(pos)
    X_pos = pca.fit(pos_emb).transform(pos_emb)
#    print 'X_pos after PCA', len(X_pos), len(X_pos[0])
#    print X_pos
   
    X_pos_x,X_pos_y = [],[]
    for entry in X_pos:
	X_pos_x.append(entry[0])
	X_pos_y.append(entry[1])
    #plot POS data
    plt.plot(X_pos_x,X_pos_y,'s') 
#    plt.annotate(pos_list,X_pos,X_pos)

plt.show()
