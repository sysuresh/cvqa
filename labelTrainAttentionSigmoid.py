from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(42)

import h5py
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
import sys
import argparse
import re
import pickle
def readJson(filename):
    print ("Reading [%s]..." % (filename))
    with open(filename) as inputFile:
        jsonData = json.load(inputFile)
    print ("Finished reading [%s]." % (filename))
    return jsonData
def binary_loss(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)
def test(y_hat, y_true):
    num = 0
    done = False
    for j, (yp, y) in enumerate(zip(y_hat, y_true)):
        ones = set()
        if np.argmax(yp) == 0:
            ones = {0}
        else:
            ones = set(np.where(yp >= 0.2)[0]) - {0}
        inc = (ones == set(np.where(y>=0.5)[0]))
        if not inc and not done:
            print(list(yp))
            print(list(y))
            done = True
        num += inc
    return num / len(y_hat)

def extractFeatures(questionRows, totalLength, maxLength, return_raw_words=False):


    X_captions = []
    x_questions = []
    raw_questions = []
    raw_captions = []
    y = []
    errors = []
    indices = []
    for i, questionRow in questionRows.iterrows():
        imageFilename = questionRow['image']
        caption = questionRow['caption']
        error = questionRow['error'].lower()
        errors = [re.sub(r'([^\w])+', '', w) for w in error.split(",")]
        questionWords = [re.sub(r'([^\w])+', '', w) for w in nltk.word_tokenize(questionRow['question'].lower()) if '?' not in w]
        relQuestionWords = [re.sub(r'([^\w])+', '', w) for w in nltk.word_tokenize(questionRow['correctquestion'].lower()) if '?' not in w]
        captionWords = caption.split(' ')

        rectify = lambda l : [w.lower() for w in l if w in w2v and w.lower() not in excludeWordList]

        newQuestionWords = rectify(questionWords)
        newCaptionWords = rectify(captionWords)
        newRelQuestionWords = rectify(relQuestionWords)
        
        if len(newQuestionWords) != len(newRelQuestionWords):
            errors.append( (questionRow) )
            continue

        questionWords = newQuestionWords
        captionWords = newCaptionWords

        captionFeature = np.zeros((1+maxLength, wordVectorSize))
        questionFeature = np.zeros((1+maxLength, wordVectorSize))
        questionFeatureRelevant = np.zeros((1+maxLength, wordVectorSize))
        
        try:
            for ci,c in enumerate(newCaptionWords):
                captionFeature[1+ci] = w2v[c]

            for ci,c in enumerate(newQuestionWords):
                questionFeature[1+ci] = w2v[c]

            for ci,c in enumerate(newRelQuestionWords):
                questionFeatureRelevant[1+ci] = w2v[c]
        except Exception as e:
            errors.append( (questionRow, e) )
            continue
        X_captions.append(captionFeature)
        x_questions.append(questionFeature)
        y.append([0.0]+[1.0 if w.lower() in errors else 0.0 for w in newQuestionWords]+[0.0 for w in range(maxLength-len(newQuestionWords))])
        X_captions.append(captionFeature)
        x_questions.append(questionFeatureRelevant)
        y.append([1.0]+[0.0 for w in range(maxLength)])
        raw_questions += [newQuestionWords, newRelQuestionWords]
        indices += [i, i]
    if return_raw_words:
        return np.asarray(x_questions),np.asarray(X_captions),y, raw_questions, raw_captions, indices, errors

    return np.asarray(x_questions),np.asarray(X_captions),y, errors

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
if __name__ == "__main__":
    modelArch = open("architecture_cvqaLabelAttention.json", "w")
    word2VecPath = 'GoogleNews-vectors-negative300.bin'
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
    inputQ = Input(shape=((1+maxLength, wordVectorSize)))
    inputC = Input(shape=((1+maxLength, wordVectorSize)))
    lstmQ = Bidirectional(GRU(1+maxLength, return_sequences=True))(inputQ)
    lstmC = Bidirectional(GRU(1+maxLength, return_sequences=True))(inputC)
    encoder = merge([lstmQ, lstmC], mode='concat')

    attention = Attention()(encoder)
    attention = Dense(4*(1+maxLength), activation='relu')(attention)
    attention = Dense(1+maxLength, activation='sigmoid')(attention)
    model = Model(input=[inputQ, inputC], output=attention)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    json.dump(model.to_json(), modelArch)
    modelArch.close()
    if len(sys.argv) < 2 or sys.argv[1] != "new":
        model = load_model("ATTENTIONSIGMOID91.h5", custom_objects={'Attention':Attention, 'binary_loss':binary_loss})
    print("Loading Word2Vec Dictionary. This may take a long time...")
    #w2v = word2vec.KeyedVectors.load_word2vec_format(word2VecPath, binary=True)
    w2v = pickle.load(open("word2vec.bin", "rb"))
    #print "Loading Captions generated by a Pre-Trained Captioning Model for Images..."
    #imageCaptions = readJson(captionFile)

    print("Loading Questions...")
    dataFile = "outBoth.csv"
    dataRows = pd.read_csv(dataFile)

    #print 'Vocab size: [%d]' % (len(word_index))

    print('Extraction Training Features...')
    X_questions, X_captions, y, errors = extractFeatures(dataRows, totalLength, maxLength)
    print("Errors:", errors)
    #y = np.expand_dims(np.array(y), axis=-1)
    y = np.array(y)
    X_questions_train, X_questions_test, X_captions_train, X_captions_test, y_train, y_test = ttsplit(X_questions, X_captions, y, test_size=0.25, random_state=1)
    expand2 = lambda data : data.reshape(1, data.shape[0], data.shape[1])
    expand3 = lambda data : data.reshape(1, data.shape[0], data.shape[1], data.shape[2])
    #y_train, y_test = expand2(y_train), expand2(y_test)
    #for a in [X_questions_train, X_questions_test, X_captions_train, X_captions_test]:
    #    a = expand3(a)
    """
    print(X_questions_train[0].shape)
    print(X_questions_train[0])
    print(X_captions_train[0].shape)
    print(X_captions_train[0])
    print(y_train[0])
    """
    #y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
    #y_test  = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)
    #model.fit([X_questions_train, X_captions_train],y_train, batch_size=128, epochs=1000, verbose=2, validation_data=([X_questions_test, X_captions_test], y_test), callbacks=[ModelCheckpoint("cvqaLabel.h5", monitor="loss", verbose=1, save_best_only=True)])

    best = test(model.predict([X_questions_test, X_captions_test]), y_test)
    print("Validation accuracy:", 100*best)
    print("Training accuracy:", 100*test(model.predict([X_questions_train, X_captions_train]), y_train))
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        try:
            for epoch in range(1000):
                print("Epoch:")
                model.fit([X_questions_train, X_captions_train],y_train, batch_size=2500, epochs=1, verbose=1)
                print("Training accuracy:", 100*test(model.predict([X_questions_train, X_captions_train]), y_train))
                testacc = test(model.predict([X_questions_test, X_captions_test]), y_test)
                print("Validation accuracy:", 100*testacc)
                if testacc > best:
                    model.save("cvqaLabelLSTMSoftmaxAttention.h5")
                    best = testacc
                    print("Saved.")
        except KeyboardInterrupt:
            print("Final accuracy:", 100*best)
else:
    word2VecPath = 'GoogleNews-vectors-negative300.bin'
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
    #w2v = word2vec.KeyedVectors.load_word2vec_format(word2VecPath, binary=True)
    w2v = pickle.load(open("word2vec.bin", "rb"))
