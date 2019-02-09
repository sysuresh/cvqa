import h5py
import os
import nltk
import random
random.seed(1729)
import pickle
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
from operator import add as addop
import pandas as pd
import argparse
import re
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
        rm = lambda s : re.sub(r'([^\w])+', '', s)
        errors = [rm(w) for w in error.split(",")]
        questionWords = [rm(w) for w in nltk.word_tokenize(questionRow['question'].lower()) if '?' not in w]
        relQuestionWords = [rm(w) for w in nltk.word_tokenize(questionRow['correctquestion'].lower()) if '?' not in w]
        captionWords = caption.split(' ')

        rectify = lambda l : [w for w in l if w in w2v and w.lower() not in excludeWordList]

        newQuestionWords = rectify(questionWords)
        newCaptionWords = rectify(captionWords)
        newRelQuestionWords = rectify(relQuestionWords)

        questionWords = newQuestionWords
        captionWords = newCaptionWords

        concatFeature = np.zeros((len(word_index)))
        concatFeatureRelevant = np.zeros((len(word_index)))
        labelFeature = np.zeros(len(word_index))
        labelFeatureRelevant = np.zeros(len(word_index))
        labelFeatureRelevant[0] = 1.0
        
        try:
            for ci,c in enumerate(newQuestionWords+newCaptionWords):
                concatFeature[word_index[c.lower()]] += 1

            for ci,c in enumerate(newRelQuestionWords+newCaptionWords):
                concatFeatureRelevant[word_index[c.lower()]] += 1

        except Exception as e:
            errors.append( (questionRow, e) )

        #x_questions.append(concatFeature)
        for w in newQuestionWords:
            if w.lower() in errors:
                try:
                    labelFeature[word_index[w.lower()]] = 1.0
                except Exception as e:
                    print(e)
                    print(w)
        #y.append(labelFeature)
        #x_questions.append(concatFeatureRelevant)
        #y.append(labelFeatureRelevant)
        np.savez_compressed('dataBow/qid_'+str(i*2), a=concatFeature);
        np.savez_compressed('dataBow/qid_'+str(1 + i*2), a=concatFeatureRelevant);
        np.savez_compressed('dataBow/lid_'+str(i*2), a=labelFeature);
        np.savez_compressed('dataBow/lid_'+str(1 + i*2), a=labelFeatureRelevant);

    return np.asarray(x_questions),y, errors

def extractVocab(rowSet):
    word_index = {'RELEVANT':0}
    index_word = {0:'RELEVANT'}
    count = 1
    for i,questionRow in rowSet.iterrows():
        caption = questionRow['caption']
        rm = lambda s : re.sub(r'([^\w])+', '', s)
        questionWords = [rm(w) for w in nltk.word_tokenize(questionRow['question'])+nltk.word_tokenize(questionRow['correctquestion'])]
        for w in questionWords:
            if (w in w2v) and (w not in word_index) and (w not in excludeWordList) and (w.lower() not in word_index):
                word_index[w.lower()] = count
                index_word[word_index[w.lower()]] = w.lower()
                count += 1
        captionWords = caption.split(' ')
        for w in captionWords:
            if (w in w2v) and (w not in word_index) and (w not in excludeWordList) and (w.lower() not in word_index):
                word_index[w.lower()] = count
                index_word[word_index[w.lower()]] = w.lower()
                count += 1

    print(word_index['are'])
    print(len(word_index))
    print(count - 1)

    return word_index, index_word

def test(y_hat, y_true):
    thresholds = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2]
    num = [0 for threshold in thresholds]
    done = False
    for j, (yp, y) in enumerate(zip(y_hat, y_true)):
        if not done:
            print(yp[:10])
            done = True
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
    #word2VecPath = 'GoogleNews-vectors-negative300.bin'
    # captionFile = '/sb-personal/cvqa/data/cvqa/imagecaptions.json'
    # trainFile = '/sb-personal/cvqa/data/cvqa/cvqa-sameQuestionDataset-list5-train.csv'
    # testFile = '/sb-personal/cvqa/data/cvqa/cvqa-sameQuestionDataset-list5-test.csv'

    #trainFile = results.trainFile
    #testFile = results.testFile
    #experimentType = results.experimentType
    #numberOfEpochs = results.numberOfEpochs

    #outputResultsFile = os.path.join(results.outputPath, "outputTestResults-%s.csv" % (experimentType))
    #outputStatsFile = os.path.join(results.outputPath, "outputStats-%s.csv" % (experimentType))

    wordVectorSize = 300
    maxLength = 18
    totalLength = maxLength * 2
    excludeWordList = ['is','a','the','what','that','to','who','why']

    #print("Loading Word2Vec Dictionary. This may take a long time...")
    #w2v = pickle.load(open("word2vec.bin", 'rb'))#w2v = word2vec.KeyedVectors.load_word2vec_format(word2VecPath, binary=True)

    #print "Loading Captions generated by a Pre-Trained Captioning Model for Images..."
    #imageCaptions = readJson(captionFile)

    #print("Loading Questions...")
    #dataFile = "outBoth.csv"
    #dataRows = pd.read_csv(dataFile)

    #print 'Vocab size: [%d]' % (len(word_index))

    #print('Extraction Training Features...')
    #word_index, index_word = extractVocab(dataRows)
    #extractFeatures(dataRows, totalLength, maxLength, word_index)#X_questions, y, errors = extractFeatures(dataRows, totalLength, maxLength, word_index)
    #print("done")
    #quit()
    #print("Errors:", errors)
    #y = np.array(y)
    print("Building model...")
    #decoder.add(Dense(200, activation='relu'))
    decoder = Sequential()
    decoder.add(Dense(400, activation='relu', input_shape=(10534,)))
    decoder.add(Dense(500, activation='relu'))
    decoder.add(Dense(10534, activation='softmax'))
    decoder.load_weights("bagOfWords.h5")

    model = decoder
    idlist = [*range(83576)]
    random.shuffle(idlist)
    idlist_test = idlist[:len(idlist) // 4]

    from dataGeneratorBow import DataGenerator
    validation_generator = DataGenerator(idlist_test)

    X_questions= np.zeros((len(idlist_test), 10534))
    y_test = np.zeros((len(idlist_test), 10534))
    for i, ID in enumerate(idlist_test):
        X_questions[i,] = np.load('dataBow/qid_' + str(ID) + '.npz')['a']
        y_test[i,] = np.load('dataBow/lid_'+str(ID)+'.npz')['a']
    print(test(model.predict(X_questions), y_test))  
    #decoder.fit_generator(generator = training_generator, validation_data = validation_generator, epochs=1000, workers=4, use_multiprocessing=True, verbose=1, shuffle=False, callbacks=[ModelCheckpoint('bagOfWords.h5', monitor='val_loss', verbose=1, save_best_only=True)])
    """
    print("Beginning training...")
    best = 0
    threshold = 0

    try:
        for epoch in range(1000):
                print("Epoch:")
                decoder.fit(X_questions_train,y_train, batch_size=2500, epochs=1, verbose=1)
                print("Training accuracy:", 100*test(decoder.predict(X_questions_train), y_train)[0])
                testacc, th = test(decoder.predict(X_questions_test), y_test)
                print("Validation accuracy:", 100*testacc)
                if testacc > best:
                    decoder.save("cvqaLabelBow.h5")
                    best = testacc
                    threshold = th
                    print("Saved.")
    except KeyboardInterrupt:
        print(best)
        print(threshold)
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
    """
