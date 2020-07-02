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
import keras.backend as K
import pandas as pd
import sys
import argparse
import re
import pickle

import config
from common import readJson, binary_loss

def test(y_hat, y_true):
    num = 0
    done = False
    for j, (yp, y) in enumerate(zip(y_hat, y_true)):
        ones = set()
        if np.argmax(yp) == 0:
            ones = {0}
        else:
            ones = set(np.where(yp >= 0.5)[0]) - {0}
        inc = (ones == set(np.where(y>=0.5)[0]))
        if not inc and not done:
            print(list(yp))
            print(list(y))
            done = True
        num += inc
    return num / len(y_hat)

def extractFeatures(questionRows, return_raw_words=False):


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

        rectify = lambda l : [w.lower() for w in l if w in w2v and w.lower() not in config.excludeWordList]

        newQuestionWords = rectify(questionWords)
        newCaptionWords = rectify(captionWords)
        newRelQuestionWords = rectify(relQuestionWords)
        
        if len(newQuestionWords) != len(newRelQuestionWords):
            errors.append( (questionRow) )
            continue

        questionWords = newQuestionWords
        captionWords = newCaptionWords

        captionFeature = np.zeros((1+config.maxLength, config.wordVectorSize))
        questionFeature = np.zeros((1+config.maxLength, config.wordVectorSize))
        questionFeatureRelevant = np.zeros((1+config.maxLength, config.wordVectorSize))
        
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
        y.append([0.0]+[1.0 if w.lower() in errors else 0.0 for w in newQuestionWords]+[0.0 for w in range(config.maxLength-len(newQuestionWords))])

        X_captions.append(captionFeature)
        x_questions.append(questionFeatureRelevant)
        y.append([1.0]+[0.0 for w in range(config.maxLength)])

        raw_questions += [newQuestionWords, newRelQuestionWords]
        indices += [i, i]

    if return_raw_words:
        return np.asarray(x_questions),np.asarray(X_captions),y, raw_questions, raw_captions, indices, errors

    return np.asarray(x_questions),np.asarray(X_captions),y, errors

if __name__ == "__main__":

    print("Building model...")

    inputQ = Input(shape=((1+config.maxLength, config.wordVectorSize)))
    inputC = Input(shape=((1+config.maxLength, config.wordVectorSize)))
    lstmQ = Bidirectional(GRU(1+config.maxLength, return_sequences=True))(inputQ)
    lstmC = Bidirectional(GRU(1+config.maxLength, return_sequences=True))(inputC)
    encoder = merge([lstmQ, lstmC], mode='concat')

    attention = Attention()(encoder)
    attention = Dense(4*(1+config.maxLength), activation='relu')(attention)
    attention = Dense(1+config.maxLength, activation='sigmoid')(attention)
    model = Model(input=[inputQ, inputC], output=attention)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if len(sys.argv) < 2 or sys.argv[1] != "new":
        model = load_model("model.h5", custom_objects={'Attention':Attention, 'binary_loss':binary_loss})

    print("Loading Word2Vec Dictionary. This may take a long time...")
    w2v = word2vec.KeyedVectors.load_word2vec_format(config.word2VecPath, binary=True)
    #w2v = pickle.load(open("word2vec.bin", "rb"))

    print("Loading Questions...")
    dataFile = "outBoth.csv"
    dataRows = pd.read_csv(dataFile)


    print('Extraction Training Features...')
    X_questions, X_captions, y, errors = extractFeatures(dataRows)
    y = np.array(y)

    X_questions_train, X_questions_test, X_captions_train, X_captions_test, y_train, y_test = ttsplit(X_questions, X_captions, y, test_size=0.25, random_state=1)

    best = test(model.predict([X_questions_test, X_captions_test]), y_test)

    print("Starting validation accuracy:", 100*best)
    print("Starting training accuracy:", 100*test(model.predict([X_questions_train, X_captions_train]), y_train))

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

