HOMEWORK 3
Xiaoqian Ma
xm2164

Part 2
a) 
Method 1: take care number with more clear category, i.e. date, fraction, time mark and etc.
Method 2: Another approach to balance the level of UNKNOWN words in training and testing data is to,
1) calculate the percentage of UNKNONW words in the training data first,
2) replace words in the testing data that are not part of the vocalbulary to UNKNOWN tag only propotionally to the level pertained in the training set. Randomly replace other words that are not in the vocabulary with some known words. 
In this way, the algorithm with get ride of the undeserved performance gain for giving two different unknown words the same matching result. And in this way, it may yeild a more procise result in predictions. 
Method 3: Stem the words to its root form
Get ride of the variations of words and leave the stem form of each word. In this way, the dependency between words is clearer and more robust. Though, this approach may also lose some linguistic information for tense, active or passive and etc. So, this may or may not be a performance booster and is subject to empirical experiments. 


b)
As Eliyahu Kiperwasser describe in the paper, there are some mysterious performance mismatch when adding more embeddings. "when adding external word embeddings the accuracy of the graph-based parser degrades." So,in practice, the choice of features may not be randomly selected without precausious and some semi-supervised learning may help promote the performance.   
The order of processing the four conditions may have different effects on how the algorithm behaves. 

c)
English without POS
11.6135940552
Training loss: (-105.46921634674072, -53.594579219818115)
UAS: 62.84
LAS: 52.18

English with POS
11.7176499367
Training loss: (-124.04089492559433, -31.766519784927368)
UAS: 77.36
LAS: 72.71


Korean with POS
4.77587795258
Training loss: (-35.46677327156067, -19.118006944656372)
UAS: 73.17
LAS: 57.16

Swedish with POS
20.236374855
Training loss: (-1773.8816633224487, -58.28948187828064)
UAS: 82.13
LAS: 72.36
