import h5py
import os
import nltk
import gensim.models.keyedvectors as word2vec
import json
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import *
from keras.layers import *
from sklearn.metrics import *
from sklearn.model_selection import train_test_split as ttsplit
from keras import metrics
from keras.regularizers import L1L2
from keras.callbacks import ModelCheckpoint
import pandas as pd
import argparse
def mean_abs_diff(y_true, y_pred):
    return sum([abs(k - l) >= 0.5 for i, j in zip(y_true1, y_pred1) for k, l in zip(i, j) ])
def readJson(filename):
    print ("Reading [%s]..." % (filename))
    with open(filename) as inputFile:
        jsonData = json.load(inputFile)
    print ("Finished reading [%s]." % (filename))
    return jsonData

def extractFeatures(questionRows, totalLength, maxLength, word_index):


    X_captions = []
    x_questions = []
    y = []
    errors = []
    for i, questionRow in questionRows.iterrows():
        imageFilename = questionRow['image']
        caption = questionRow['caption']
        error = questionRow['error'].lower()
        errors = error.split(",")
        questionWords = [w for w in nltk.word_tokenize(questionRow['question'].lower()) if '?' not in w]
        relQuestionWords = [w for w in nltk.word_tokenize(questionRow['correctquestion'].lower()) if '?' not in w]
        captionWords = caption.split(' ')

        rectify = lambda l : [w for w in l if w in w2v and w.lower() not in excludeWordList]

        newQuestionWords = rectify(questionWords)
        newCaptionWords = rectify(captionWords)
        newRelQuestionWords = rectify(relQuestionWords)

        questionWords = newQuestionWords
        captionWords = newCaptionWords

        concatFeature = np.zeros((len(newQuestionWords)+len(newCaptionWords), wordVectorSize))
        concatFeatureRelevant = np.zeros((len(newRelQuestionWords)+len(newCaptionWords), wordVectorSize))
        labelFeature = np.zeros(len(word_index))
        labelFeatureRelevant = np.zeros(len(word_index))
        labelFeatureRelevant[0] = 1.0
        
        try:
            for ci,c in enumerate(newQuestionWords+newCaptionWords):
                concatFeature[ci] = w2v[c]

            for ci,c in enumerate(newRelQuestionWords+newCaptionWords):
                concatFeatureRelevant[ci] = w2v[c]
            concatFeature = np.mean(concatFeature, axis=0)
            concatFeatureRelevant = np.mean(concatFeatureRelevant, axis=0)
        except Exception as e:
            errors.append( (questionRow, e) )

        x_questions.append(concatFeature)
        for w in newQuestionWords:
            if w.lower() in errors:
                labelFeature[word_index[w]] = 1.0
        y.append(labelFeature)
        x_questions.append(concatFeatureRelevant)
        y.append(labelFeatureRelevant)

    return np.asarray(x_questions),y, errors

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

def test(y_hat, y_true):
    num = 0
    two = 2
    for j, (yp, y) in enumerate(zip(y_hat, y_true)):
        num += (set(np.where(yp >= 0.01)[0]) == set(np.where(y>=0.5)[0]))
        if two:
            print(np.sort(yp)[-3:])
            two -= 1
    return num / len(y_hat)
    
if __name__ == "__main__":

    word2VecPath = 'GoogleNews-vectors-negative300.bin'
    # captionFile = '/sb-personal/cvqa/data/cvqa/imagecaptions.json'
    # trainFile = '/sb-personal/cvqa/data/cvqa/cvqa-sameQuestionDataset-list5-train.csv'
    # testFile = '/sb-personal/cvqa/data/cvqa/cvqa-sameQuestionDataset-list5-test.csv'

    #trainFile = results.trainFile
    #testFile = results.testFile
    #experimentType = results.experimentType
    #numberOfEpochs = results.numberOfEpochs

    #outputResultsFile = os.path.join(results.outputPath, "outputTestResults-%s.csv" % (experimentType))
    #outputStatsFile = os.path.join(results.outputPath, "outputStats-%s.csv" % (experimentType))

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
    X_questions, y, errors = extractFeatures(dataRows, totalLength, maxLength, word_index)
    print("Errors:", errors)
    y = np.array(y)
    print("Building model...")
    #decoder.add(Dense(200, activation='relu'))
    decoder = Sequential()
    decoder.add(Dense(300, activation='relu', input_shape=(wordVectorSize,)))
    decoder.add(Dense(200, activation='relu'))
    decoder.add(Dense(150, activation='relu'))
    decoder.add(Dense(len(word_index), activation='softmax'))
    decoder.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    modelArch = open("architecture_cvqaLabelAvg.json", "w")
    json.dump(decoder.to_json(), modelArch)
    X_questions_train, X_questions_test, y_train, y_test = ttsplit(X_questions, y, test_size=0.25, random_state=1)
    print("Beginning training...")
    best = 0
    for epoch in range(1000):
        print("Epoch:")
        decoder.fit(X_questions_train,y_train, batch_size=2500, epochs=1, verbose=1)
        print("Training accuracy:", 100*test(decoder.predict(X_questions_train), y_train))
        testacc = test(decoder.predict(X_questions_test), y_test)
        print("Validation accuracy:", 100*testacc)
        if testacc > best:
            decoder.save("cvqaLabelAvg.h5")
            best = testacc
            print("Saved.")
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
    w2v = word2vec.KeyedVectors.load_word2vec_format(word2VecPath, binary=True)

