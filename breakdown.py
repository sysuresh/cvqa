import keras
from keras import *
from keras.layers import *
from keras.models import *
import gensim.models.keyedvectors as word2vec
from attention_keras import Attention
from labelTrainSoftmax import extractFeatures
from sklearn.model_selection import train_test_split as ttsplit
import pandas as pd
import numpy as np
import re 
import nltk
def test(y_hat, y_true, rows, indices_test, questions, threshold):
    threshes = [0.1, 0.2, 0.3, 0.4, 0.5]
    totalcorrect = [0 for t in threshes]
    #irrelevant
    correctnouns = [0, 0] #correct noun but other inaccuracy, completely correct
    totalnouns = 0

    correctverbs = [0, 0] #correct verb but other inaccuracy, completely correct
    totalverbs = 0

    correctnounverbs = [0, 0, 0, 0] #nouns, verbs, both but other inaccuracy, completely correct
    totalnounverbs = 0

    #relevant
    relevantcorrectnouns = 0
    relevanttotalnouns = 0

    relevantcorrectverbs = 0
    relevanttotalverbs = 0

    relevantcorrectnounverbs = 0
    relevanttotalnounverbs = 0

    rawRows = list(rows.iterrows())
    for yh, yt, i, question in zip(y_hat, y_true, indices_test, questions):
        row = rawRows[i][1]
        error = [re.sub(r'([^\w])+', '', w) for w in row["error"].split(",")]
        question = [re.sub(r'([^\w])+', '', w) for w in question]
        for j in range(len(threshes)):
            totalcorrect[j] += (set(np.where(np.array(yh) >= threshes[j])[0]) == set(np.where(np.array(yt) >= 0.5)[0]))
        if i < third: #nounverb
            if yt[0] > 0.5:
                relevanttotalnounverbs += 1
                if np.argmax(yh) == 0:
                    relevantcorrectnounverbs += 1
            else:
                totalnounverbs += 1
                if np.argmax(yh) == 0:
                    continue
                truenounindex = 1 + question.index(error[0])
                trueverbindex = 1 + question.index(error[1])
                ones = set(np.where(yh >= threshold)[0]) - {0}
                if ones == {truenounindex, trueverbindex}:
                    correctnounverbs[3] += 1
                if truenounindex in ones and trueverbindex in ones:
                    correctnounverbs[2] += 1
                elif truenounindex in ones:
                    correctnounverbs[0] += 1
                elif trueverbindex in ones:
                    correctnounverbs[1] += 1
        elif i <= 2*third: #verb
            if yt[0] > 0.5:
                relevanttotalverbs += 1
                if np.argmax(yh) == 0:
                    relevantcorrectverbs += 1
            else:
                totalverbs += 1
                ones = set(np.where(yh >= threshold)[0]) - {0}
                trueverbindex = 1 + question.index(error[0])
                if ones == {trueverbindex}:
                    correctverbs[1] += 1
                elif trueverbindex in ones:
                    correctverbs[0] += 1

        else: #noun
            if yt[0] > 0.5:
                relevanttotalnouns += 1
                if np.argmax(yh) == 0:
                    relevantcorrectnouns += 1
            else:
                totalnouns += 1
                ones = set(np.where(yh >= threshold)[0]) - {0}
                truenounindex = 1 + question.index(error[0])
                if ones == {truenounindex}:
                    correctnouns[1] += 1
                elif truenounindex in ones:
                    correctnouns[0] += 1

    stats = [len(y_true), correctnouns, totalnouns, correctverbs, totalverbs, correctnounverbs, totalnounverbs, relevantcorrectnouns, relevanttotalnouns, relevantcorrectverbs, relevanttotalverbs, relevantcorrectnounverbs, relevanttotalnounverbs, totalcorrect, threshes]
    return stats

#modelAttention = load_model("cvqaLabelLSTMSoftmaxAttention.h5", custom_objects={'Attention':Attention})
#modelNoAttention = load_model("cvqaLabelLSTMSoftmaxNoAttention.h5")
modelAttention = load_model("cvqaLabelLSTMSigmoidAttention.h5", custom_objects={'Attention':Attention})
"""
maxLength = 18
wordVectorSize = 300
word2VecPath = 'GoogleNews-vectors-negative300.bin'
w2v = word2vec.KeyedVectors.load_word2vec_format(word2VecPath, binary=True)
excludeWordList = ['is','a','the','what','that','to','who','why', 'What', 'How', 'Who', 'Why']
"""
dataRows = pd.read_csv("outBoth.csv")
#word_index, index_word = extractVocab(dataRows)
"""
encoder_a = Sequential()
encoder_a.add(Bidirectional(GRU(200, return_sequences=True), input_shape=(maxLength,wordVectorSize)))

encoder_b = Sequential()
encoder_b.add(Bidirectional(GRU(200, return_sequences=True), input_shape=(maxLength,wordVectorSize)))

decoder = Sequential()
decoder.add(Merge([encoder_a, encoder_b], mode='concat'))
decoder.add(Flatten())
decoder.add(Dense(len(word_index), activation='softmax'))
decoder.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])
decoder.load_weights("cvqaLabelVocab.h5")
modelAttention = decoder
"""
third = len(dataRows) // 3
#X_questions, X_captions, y, raw_questions, indices, errors = extractFeaturesVocab(dataRows, None, 18, word_index)
X_questions, X_captions, y, raw_questions, raw_captions, indices, errors = extractFeatures(dataRows, None, 18, return_raw_words=True)
X_questions_train, X_questions_test, X_captions_train, X_captions_test, y_train, y_test, raw_questions_train, raw_questions_test, indices_train, indices_test  = ttsplit(X_questions, X_captions, y, raw_questions, indices, random_state=1)
print("STATISTICS FOR ATTENTION SIGMOID VALIDATION:")
print(test(modelAttention.predict([X_questions_test, X_captions_test]), y_test, dataRows, indices_test, raw_questions_test, 0.5))
#print(test(modelAttention.predict([X_questions_test, X_captions_test]), y_test, dataRows, indices_test, raw_questions_test, 0.25))
print("STATISTICS FOR NO-ATTENTION VALIDATION:")
#print(test(modelNoAttention.predict([X_questions_test, X_captions_test]), y_test, dataRows, indices_test, raw_questions_test, 0.2))
