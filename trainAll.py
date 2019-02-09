import h5py
from operator import add as addop
import os
import nltk
import gensim.models.keyedvectors as word2vec
import json
import numpy as np
import keras
from keras.models import *
from keras.layers import *
from keras.layers.core import *
from sklearn.metrics import *
from sklearn.model_selection import train_test_split as ttsplit
from keras import metrics
from keras.regularizers import L1L2
from keras.callbacks import ModelCheckpoint
from attention_keras import Attention
import pandas as pd
import argparse
from labelTrainSoftmax import extractFeatures as labeling_method
#from vocabTrain import extractFeatures as vocab_method
#from bow import extractFeatures as bag_of_words
#from avg import extractFeatures as w2v_avg
word2VecPath = 'GoogleNews-vectors-negative300.bin'

def readJson(filename):
    print ("Reading [%s]..." % (filename))
    with open(filename) as inputFile:
        jsonData = json.load(inputFile)
    print ("Finished reading [%s]." % (filename))
    return jsonData
def binary_loss(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)
def test(y_hat, y_true):
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    num = [0 for threshold in thresholds]
    for j, (yp, y) in enumerate(zip(y_hat, y_true)):
        ones = [set() for threshold in thresholds]
        if np.argmax(yp) == 0:
            ones = [{0} for threshold in thresholds]
        else:
            ones = [set(np.where(yp >= threshold)[0]) - {0} for threshold in thresholds]
        inc = [int(oneset == set(np.where(y>=0.5)[0])) for oneset in ones]
        num = list(map(addop, num, inc))
    index_max = max(range(len(thresholds)), key=num.__getitem__)
    return (num[index_max] / len(y_hat), thresholds[index_max])
def extractVocab(rowSet):
    word_index = {'RELEVANT':0}
    index_word = {0:'RELEVANT'}
    for i,questionRow in rowSet.iterrows():
        caption = questionRow['caption']

        questionWords = nltk.word_tokenize(questionRow['question'])+nltk.word_tokenize(questionRow['correctquestion'])
        for w in questionWords:
            if (w in w2v) and (w not in word_index) and (w not in excludeWordList):
                word_index[w] = len(word_index)
                index_word[word_index[w]] = w
        captionWords = caption.split(' ')
        for w in captionWords:
            if (w in w2v) and (w not in word_index) and (w not in excludeWordList):
                word_index[w] = len(word_index)
                index_word[word_index[w]] = w

    return word_index, index_word

maxQuestionLength = 8
maxCaptionLength = 16
wordVectorSize = 300
embeddingSize = 250
# numberOfEpochs = 30
subsetCount = 4000
maxLength = 18
totalLength = maxLength * 2
# totalLength = maxLength
n_hidden = 40
excludeWordList = ['is','a','the','what','that','to','who','why', 'What', 'How', 'Who', 'Why']
# experimentType = 'all'
print("Building model...")
modelAttention = load_model("cvqaLabelLSTMSoftmaxAttention.h5", custom_objects={'Attention':Attention, 'binary_loss':binary_loss})
modelNoAttention = load_model("cvqaLabelLSTMSoftmaxNoAttention.h5")
#modelAvg = load_model("cvqaLabelAvg.h5")
#modelBow= load_model("cvqaLabelBow.h5")
print("Loading Word2Vec Dictionary. This may take a long time...")
w2v = word2vec.KeyedVectors.load_word2vec_format(word2VecPath, binary=True)

#print "Loading Captions generated by a Pre-Trained Captioning Model for Images..."
#imageCaptions = readJson(captionFile)

print("Loading Questions...")
dataFile = "outBoth.csv"
dataRows = pd.read_csv(dataFile)

#print 'Vocab size: [%d]' % (len(word_index))

print('Extraction Training Features...')
word_index, index_word = extractVocab(dataRows)
X_questions, X_captions, yLabel, errors = labeling_method(dataRows, totalLength, maxLength)
#X_questionsBagged, yBagged, errors = bag_of_words(dataRows, totalLength, maxLength, word_index)
#X_questionsAvg, yAvg, errors = w2v_avg(dataRows, totalLength, maxLength, word_index)
#X_questionsVocab, X_captionsVocab, yVocab, errors = vocab_method(dataRows, totalLength, maxLength, word_index)
print("Errors:", errors)
#y = np.expand_dims(np.array(y), axis=-1)
yLabel = np.array(yLabel)
#yBagged, yAvg = np.array(yBagged), np.array(yAvg)
X_questions_train, X_questions_test, X_captions_train, X_captions_test, yLabel_train, yLabel_test = ttsplit(X_questions, X_captions, yLabel, test_size=0.25, random_state=1)
#X_questionsBagged_train, X_questionsBagged_test, yBagged_train, yBagged_test = ttsplit(X_questionsBagged, yBagged, test_size=0.25)
#X_questionsAvg_train, X_questionsAvg_test, yAvg_train, yAvg_test = ttsplit(X_questionsAvg, yAvg, test_size=0.25)
expand2 = lambda data : data.reshape(1, data.shape[0], data.shape[1])
expand3 = lambda data : data.reshape(1, data.shape[0], data.shape[1], data.shape[2])
#y_train, y_test = expand2(y_train), expand2(y_test)
#for a in [X_questions_train, X_questions_test, X_captions_train, X_captions_test]:
#    a = expand3(a)
#y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
#y_test  = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)
print("Beginning training...")
#model.fit([X_questions_train, X_captions_train],y_train, batch_size=128, epochs=1000, verbose=2, validation_data=([X_questions_test, X_captions_test], y_test), callbacks=[ModelCheckpoint("cvqaLabel.h5", monitor="loss", verbose=1, save_best_only=True)])
best = [0 for i in range(2)]
best_thresholds = [0.3, 0.15]
testacc = [0 for i in range(2)]
try:
    for epoch in range(1000):
        print("Training Attention accuracy:", 100*test(modelAttention.predict([X_questions_train, X_captions_train]), yLabel_train)[0])
        testacc[0], th = test(modelAttention.predict([X_questions_test, X_captions_test]), yLabel_test)
        print("Validation Attention accuracy:", 100*testacc[0])
        if testacc[0] > best[0]:
            modelAttention.save("cvqaLabelLSTMSoftmaxAttention.h5")
            best[0] = testacc[0]
            best_thresholds[0] = th
            print("Saved attention.")
        print("Training No-Attention accuracy:", 100*test(modelNoAttention.predict([X_questions_train, X_captions_train]), yLabel_train)[0])
        testacc[1], th = test(modelNoAttention.predict([X_questions_test, X_captions_test]), yLabel_test)
        print("Validation No-Attention accuracy:", 100*testacc[1])
        if testacc[1] > best[1]:
            modelNoAttention.save("cvqaLabelLSTMSoftmaxNoAttention.h5")
            best[1] = testacc[1]
            best_thresholds[1] = th
            print("Saved no-attention.")
        modelAttention.fit([X_questions_train, X_captions_train],yLabel_train, batch_size=2500, epochs=1, verbose=1)
        modelNoAttention.fit([X_questions_train, X_captions_train],yLabel_train, batch_size=2500, epochs=1, verbose=1)

except KeyboardInterrupt:
    print(100*best[0])
    print(best_thresholds[0])
    print(100*best[1])
    print(best_thresholds[1])
