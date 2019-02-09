import h5py
import random
random.seed(1729)
import os
import nltk
import gensim.models.keyedvectors as word2vec
import json
import numpy as np
import tensorflow as tf
sess = tf.Session()
import keras
import keras.backend as K
K.set_session(sess)
from keras.models import *
from keras.layers import *
from sklearn.metrics import *
from sklearn.model_selection import train_test_split as ttsplit
from keras import metrics
from keras.regularizers import L1L2
from keras.callbacks import ModelCheckpoint
import pandas as pd
import argparse
from operator import add as addop
import re
import pickle
def binary_loss(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1) 
def readJson(filename):
    print ("Reading [%s]..." % (filename))
    with open(filename) as inputFile:
        jsonData = json.load(inputFile)
    print ("Finished reading [%s]." % (filename))
    return jsonData
def extractVocab(rowSet):
    word_index = {'RELEVANT':0}
    #index_word = {0:'RELEVANT'}
    count = 1
    for i,questionRow in rowSet.iterrows():
        caption = questionRow['caption']
        rm = lambda s : re.sub(r'([^\w])+', '', s)
        questionWords = [rm(w) for w in nltk.word_tokenize(questionRow['question'])+nltk.word_tokenize(questionRow['correctquestion'])]
        for w in questionWords:
            if (w in w2v) and (w not in word_index) and (w not in excludeWordList) and (w.lower() not in word_index):
                word_index[w.lower()] = count
                #index_word[word_index[w.lower()]] = w.lower()
                count += 1
        captionWords = caption.split(' ')
        for w in captionWords:
            if (w in w2v) and (w not in word_index) and (w not in excludeWordList) and (w.lower() not in word_index):
                word_index[w.lower()] = count
                #index_word[word_index[w.lower()]] = w.lower()
                count += 1

    print(word_index['are'])
    print(len(word_index))
    print(count - 1)

    return word_index


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

if __name__ == "__main__":
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
    X_captions = []
    X_questions = []
    y = []
    errors = []
    """
    for i, questionRow in dataRows.iterrows():
        if not i % 100:
            print(i)
        imageFilename = questionRow['image']
        caption = questionRow['caption']
        error = questionRow['error'].lower()
        errors = [re.sub(r'([^\w])+', '', w) for w in error.split(",")]
        questionWords = [re.sub(r'([^\w])+', '', w) for w in nltk.word_tokenize(questionRow['question'].lower()) if '?' not in w]
        relQuestionWords = [re.sub(r'([^\w])+', '', w) for w in nltk.word_tokenize(questionRow['correctquestion'].lower()) if '?' not in w]
        captionWords = caption.split(' ')

        rectify = lambda l : [w for w in l if w in w2v and w.lower() not in excludeWordList]

        newQuestionWords = rectify(questionWords)
        newCaptionWords = rectify(captionWords)
        newRelQuestionWords = rectify(relQuestionWords)

        questionWords = newQuestionWords
        captionWords = newCaptionWords

        captionFeature = np.zeros((maxLength, wordVectorSize))
        questionFeature = np.zeros((maxLength, wordVectorSize))
        questionFeatureRelevant = np.zeros((maxLength, wordVectorSize))
        labelFeature = np.zeros(len(word_index))
        labelFeatureRelevant = np.zeros(len(word_index))
        labelFeatureRelevant[0] = 1.0
        
        try:
            for ci,c in enumerate(newCaptionWords):
                captionFeature[ci] = w2v[c]

            for ci,c in enumerate(newQuestionWords):
                questionFeature[ci] = w2v[c]

            for ci,c in enumerate(newRelQuestionWords):
                questionFeatureRelevant[ci] = w2v[c]
        except Exception as e:
            errors.append( (questionRow, e) )
        for w in newQuestionWords:
            if w.lower() in errors:
                labelFeature[word_index[w]] = 1.0
        np.savez_compressed('data/cid_'+str(i*2), a=captionFeature);
        np.savez_compressed('data/cid_'+str(1 + i*2), a=captionFeature);
        np.savez_compressed('data/qid_'+str(i*2), a=questionFeature);
        np.savez_compressed('data/qid_'+str(1 + i*2), a=questionFeatureRelevant);
        np.savez_compressed('data/lid_'+str(i*2), a=labelFeature);
        np.savez_compressed('data/lid_'+str(1 + i*2), a=labelFeatureRelevant);
        #X_captions.append(captionFeature)
        #X_questions.append(questionFeature)
        #y.append(labelFeature)
        #X_captions.append(captionFeature)
        #X_questions.append(questionFeatureRelevant)
        #y.append(labelFeatureRelevant)
    """
    #X_questions, X_captions, y, errors = np.asarray(X_questions),np.asarray(X_captions),y, errors
    print("done")

    #pickle.dump(X_questions, open('vocabdataxq.pickle', 'wb'))
    #X_questions = pickle.load(open('vocabdataxq.pickle', 'rb'))#extractVocab(dataRows)
    #print("Errors:", errors)
    #quit()
    #y = np.array(y)
    #modelArch = open("architecture_cvqaLabelVocab.json", "w")
    #json.dump(decoder.to_json(), modelArch)
    idlist = [*range(83576)]
    random.shuffle(idlist)
    idlist_test = idlist[:len(idlist) // 4]
    #idlist_train = idlist[len(idlist)//4:]
    maxLength, wordVectorSize = 18, 300
    encoder_a = Sequential()
    encoder_a.add(Bidirectional(GRU(200, kernel_regularizer=L1L2(l1=0.00,l2=0.0), return_sequences=True), input_shape=(maxLength,wordVectorSize)))

    encoder_b = Sequential()
    encoder_b.add(Bidirectional(GRU(200, kernel_regularizer=L1L2(l1=0.00,l2=0.0), return_sequences=True), input_shape=(maxLength,wordVectorSize)))

    decoder = Sequential()
    decoder.add(Merge([encoder_a, encoder_b], mode='concat'))
    decoder.add(Flatten())
    decoder.add(Dense(10534, activation='softmax')) #len(word_index) = 10534
    model = decoder
    model.load_weights("vocabLSTM.h5")
    X_captions = np.zeros((len(idlist_test), 18, 300))
    X_questions= np.zeros((len(idlist_test), 18, 300))
    y_test = np.zeros((len(idlist_test), 10534))
    for i, ID in enumerate(idlist_test):
        X_captions[i,] = np.load('data/cid_' + str(ID) + '.npz')['a']
        X_questions[i,] = np.load('data/qid_' + str(ID) + '.npz')['a']
        y_test[i,] = np.load('data/lid_'+str(ID)+'.npz')['a']
    print(test(model.predict([X_questions, X_captions]), y_test))  
    #decoder.fit_generator(generator = training_generator, validation_data = validation_generator, epochs=1000, workers=4, use_multiprocessing=True, verbose=1, shuffle=False, callbacks=[ModelCheckpoint('vocabLSTM.h5', monitor='val_loss', verbose=1, save_best_only=True)])
