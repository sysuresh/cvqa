import h5py
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

def extractFeatures(questionRows, totalLength, maxLength, word_index):


    X_captions = []
    x_questions = []
    y = []
    errors = []
    for i, questionRow in questionRows.iterrows():
        if not i % 100:
            print(i)
        if i > 41700:
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

        X_captions.append(captionFeature)
        x_questions.append(questionFeature)
        for w in newQuestionWords:
            if w.lower() in errors:
                labelFeature[word_index[w]] = 1.0
        y.append(labelFeature)
        X_captions.append(captionFeature)
        x_questions.append(questionFeatureRelevant)
        y.append(labelFeatureRelevant)

    return np.asarray(x_questions),np.asarray(X_captions),y, errors

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

    print("Loading Questions...")
    dataFile = "outBoth.csv"
    dataRows = pd.read_csv(dataFile)

    print("Loading Word2Vec Dictionary. This may take a long time...")
    w2v = pickle.load(open("word2vec.bin", 'rb'))#word2vec.KeyedVectors.load_word2vec_format(word2VecPath, binary=True)

    #decoder.load_weights("cvqaLabelVocab.h5")


    #print "Loading Captions generated by a Pre-Trained Captioning Model for Images..."
    #imageCaptions = readJson(captionFile)


    #print 'Vocab size: [%d]' % (len(word_index))

    print('Extraction Training Features...')
    word_index = extractVocab(dataRows)#pickle.load(open('word_index.pickle', 'rb'))
    pickle.dump(word_index, open('word_index.pickle', 'wb'))
    print("done")
    #X_questions, X_captions, y, errors = extractFeatures(dataRows, totalLength, maxLength, word_index)

    X_captions = []
    X_questions = []
    y = []
    errors = []
    for i, questionRow in dataRows.iterrows():
        if not i % 100:
            print(i)
        if i > 41700:
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

        X_captions.append(captionFeature)
        X_questions.append(questionFeature)
        for w in newQuestionWords:
            if w.lower() in errors:
                labelFeature[word_index[w]] = 1.0
        y.append(labelFeature)
        X_captions.append(captionFeature)
        X_questions.append(questionFeatureRelevant)
        y.append(labelFeatureRelevant)

    #X_questions, X_captions, y, errors = np.asarray(X_questions),np.asarray(X_captions),y, errors


    pickle.dump(X_questions, open('vocabdataxq.pickle', 'wb'))
    print("Errors:", errors)
    quit()
    y = np.array(y)
    #modelArch = open("architecture_cvqaLabelVocab.json", "w")
    #json.dump(decoder.to_json(), modelArch)
    X_questions_train, X_questions_test, X_captions_train, X_captions_test, y_train, y_test = ttsplit(X_questions, X_captions, y, test_size=0.25)

    print("Building model...")
    encoder_a = Sequential()
    encoder_a.add(Bidirectional(GRU(200, kernel_regularizer=L1L2(l1=0.00,l2=0.0), return_sequences=True), input_shape=(maxLength,wordVectorSize)))

    encoder_b = Sequential()
    encoder_b.add(Bidirectional(GRU(200, kernel_regularizer=L1L2(l1=0.00,l2=0.0), return_sequences=True), input_shape=(maxLength,wordVectorSize)))

    decoder = Sequential()
    decoder.add(Merge([encoder_a, encoder_b], mode='concat'))
    decoder.add(Flatten())
    decoder.add(Dense(10534, activation='softmax')) #len(word_index) = 10534
    decoder.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])


    print("Beginning training...")
    """
    try:
        decoder.fit([X_questions_train, X_captions_train],y_train, batch_size=128, epochs=1000, verbose=2, validation_data=([X_questions_test, X_captions_test], y_test), callbacks=[ModelCheckpoint("cvqaLabel.h5", monitor="val_loss", verbose=2, save_best_only=True)])
    except KeyboardInterrupt:
        print(list(decoder.predict([np.array([X_questions_train[0]]), np.array([X_captions_train[0]])])[0]))
        print(list(y_train[0]))
        print(list(decoder.predict([np.array([X_questions_train[1]]), np.array([X_captions_train[1]])])[0]))
        print(list(y_train[1]))
    """
    best = 0
    threshold = 0
    best, threshold = test(decoder.predict([X_questions_test, X_captions_test]), y_test)
    print("Validation accuracy:", 100*best)

    try:
        for epoch in range(1000):
            print("Epoch:")
            print("Training accuracy:", 100*test(decoder.predict([X_questions_train, X_captions_train]), y_train)[0])
            testacc, th = test(decoder.predict([X_questions_test, X_captions_test]), y_test)
            print("Validation accuracy:", 100*testacc)
            if testacc > best:
                decoder.save("cvqaLabelVocab.h5")
                threshold = th
                best = testacc
                print("Saved.")
            decoder.fit([X_questions_train, X_captions_train],y_train, batch_size=2500, epochs=1, verbose=1)
    except KeyboardInterrupt:
        print(100*best)
        print(threshold)
