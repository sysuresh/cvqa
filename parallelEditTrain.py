import h5py
import os
import nltk
import gensim.models.keyedvectors as word2vec
import json
import numpy as np
import keras
import keras.backend as K
from attention_keras import Attention
from keras.models import *
from keras.layers import *
from sklearn.metrics import *
from sklearn.model_selection import train_test_split as ttsplit
from keras import metrics
from keras.regularizers import L1L2
from keras.callbacks import ModelCheckpoint
import pandas as pd
import argparse
from pattern3.en import singularize, pluralize, conjugate
import random
from multiprocessing import Process, Array
#modelArch = open("architecture_cvqaLabel.json", "w")

#relevantQuestionSentence, captionEmbedding, irrelevantQuestionEmbedding, [(questionEdits, questionEditEmbeddings)], captionTag, questionTag
mD = lambda tense,person,number,aspect='imperfective': {'tense' : tense, 'person' : person, 'number' : number, 'mood' : 'indicative', 'aspect' : aspect, 'negated' : False}
configConjugation = {
        "VB" :  mD('present', 3, 'plural'),
        "VBD" : mD('past', 3, 'singular'),
        "VBG" : mD('present', 3, 'singular', aspect='progressive'),
        "VBN" : mD('past', None, None, aspect='progressive'),
        "VBP" : mD('present', 1, 'singular'),
        "VBZ" : mD('present', 3, 'singular')
        }
def conj(word, tag):
    if tag == 'NNS':
        return pluralize(word)
    if tag in ('NN', 'NNP'):
        return word
    return conjugate(word, **configConjugation[tag])

def mean_abs_diff(y_true, y_pred):
    return sum([abs(k - l) >= 0.5 for i, j in zip(y_true1, y_pred1) for k, l in zip(i, j) ])
def readJson(filename):
    print ("Reading [%s]..." % (filename))
    with open(filename) as inputFile:
        jsonData = json.load(inputFile)
    print ("Finished reading [%s]." % (filename))
    return jsonData
def embed(wordList):
    embedding = np.zeros((1+maxLength, wordVectorSize))
    for i, word in enumerate(wordList):
        if word in w2v:
            embedding[i+1] = w2v[word]
    return embedding
def extractFeatures(questionRows, totalLength, maxLength, questionToTag, captionToNounList, actions):

    y_RelevantQuestions = []
    y_Label = []
    X_RawCaptions = []
    X_Captions = []
    X_IrrelevantQuestions = []
    X_RawQuestionEdits = []
    X_QuestionEdits = []
    
    errors = []
    for i,questionRow in questionRows.iterrows():
        imageFilename = questionRow['image']
        caption = questionRow['caption']
        error = questionRow['error']
        correction = questionRow['correction']

        questionWords = [w for w in nltk.word_tokenize(questionRow['question'].lower()) if '?' not in w]
        relQuestionWords = [w for w in nltk.word_tokenize(questionRow['correctquestion'].lower()) if '?' not in w]
        y_RelevantQuestions.append(questionRow['correctquestion'].lower().replace('?',''))
        captionWords = caption.split(' ')
        X_RawCaptions.append(captionWords)

        rectify = lambda l : [w.lower() for w in l if w in w2v and w.lower() not in excludeWordList]

        newQuestionWords = rectify(questionWords)
        newCaptionWords = rectify(captionWords)
        newRelQuestionWords = rectify(relQuestionWords)

        questionWords = newQuestionWords
        captionWords = newCaptionWords

        captionFeature = np.zeros((1+maxLength, wordVectorSize))
        questionFeature = np.zeros((1+maxLength, wordVectorSize))
        
        try:
            captionFeature, questionFeature = embed(newCaptionWords), embed(newQuestionWords)
        except Exception as e:
            errors.append( (questionRow, e) )
            continue
        X_Captions.append(captionFeature)
        X_IrrelevantQuestions.append(questionFeature)
        y_Label.append([0.0]+[1.0 if w.lower() == error.lower() else 0.0 for w in newQuestionWords])
        replacements = []
        if i < len(questionRows) // 3:
            replacements += [(conj(wNoun, questionToTag[y_RelevantQuestions[-1]].split(",")[0]), conj(wVerb, questionToTag[y_RelevantQuestions[-1]])) for wNoun in captionToNounList[caption] for wVerb in actions]
        elif i <= 2 * len(questionRows) // 3:
            replacements += [conj(w, questionToTag[y_RelevantQuestions[-1]]) for w in captionToNounList[caption]]
        else:
            replacements += [conj(w, questionToTag[y_RelevantQuestions[-1]]) for w in actions]
        replacementQuestionList = [' '.join(newQuestionWords).replace('?','').replace(error.lower(), replacement) if len(error.split(",")) == 1 else ' '.join(newQuestionWords).replace('?', '').replace(error.split(",")[0], replacement[0]).replace(error.split(",")[1], replacement[1]) for replacement in replacements if replacement in w2v]
        X_RawQuestionEdits.append(replacementQuestionList)

    c = lambda x : np.array(x)
    return c(X_IrrelevantQuestions), c(X_Captions), c(y_Label), c(X_QuestionEdits), c(X_RawQuestionEdits), c(y_RelevantQuestions), errors

def extractVocab(rowSet):
    word_index = {'RELEVANT':0}
    index_word = {0:'RELEVANT'}
    for r in rowSet:
        for i,questionRow in r.iterrows():
            imageFilename = questionRow['image']
            caption = imageCaptions[imageFilename]

            questionWords = questionRow['question'].split(' ')
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
def parallel_edit(cap, rawQuestList, relQues, correctEdits, count):
    try: 
        quesList = [embed(nltk.word_tokenize(rawQues)) for rawQues in rawQuesList]
        editPredictions = (np.array(rawQuesList)[np.array([model.predict([np.array([cap]), np.array([ques])])[0][0] for ques in quesList]).argsort()])[::-1]
        for i, editPred in enumerate(editPredictions):
            if i >= len(correctEdits):
                break
            if editPred == relQues:
                correctEdits[i] += 1
        count += 1
        print(count)
    except KeyboardInterrupt:
        break

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
excludeWordList = ['is','a','the','what','that','to','who','why']
# experimentType = 'all'

print("Loading descriptions and tags...")
dataRows = pd.read_csv("questionsThird.csv")
questionToTag = {}
for i, row in dataRows.iterrows():
    questionToTag[row["question"].lower().replace('?','')] = row["type"]
dataRows = pd.read_csv("captions.csv")
captionToNounList = {}
for i, row in dataRows.iterrows():
    captionToNounList[row["caption"]] = eval(row["nouns"])
actions = eval(open("actions.txt").read().strip())
print("Loading Word2Vec Dictionary. This may take a long time...")
w2v = word2vec.KeyedVectors.load_word2vec_format(word2VecPath, binary=True)

#print "Loading Captions generated by a Pre-Trained Captioning Model for Images..."
#imageCaptions = readJson(captionFile)

print("Loading Questions...")
dataFile = "outBoth.csv"
dataRows = pd.read_csv(dataFile)

#print 'Vocab size: [%d]' % (len(word_index))

print('Extraction Training Features...')
X_IrrelevantQuestions, X_Captions, y_Label, X_QuestionEdits, X_RawQuestionEdits, y_RelevantQuestions, errors = extractFeatures(dataRows, totalLength, maxLength, questionToTag, captionToNounList, actions)

print(X_QuestionEdits.shape)
print("Errors:", errors)

print("Loading model...")
modelAttention = load_model("cvqaLabelLSTMSoftmaxAttention.h5", custom_objects={'Attention':Attention})
model = modelNoAttention = load_model("cvqaLabelLSTMSoftmaxNoAttention.h5")

print("Predicting...")
y_predLabel = model.predict([X_IrrelevantQuestions, X_Captions])
indices = [i for i in range(len(y_Label)) if np.argmax(y_Label[i]) == np.argmax(y_predLabel[i])]
#X_QuestionEdits = X_QuestionEdits[indices, :, :]
X_RawQuestionEdits = X_RawQuestionEdits[indices]
X_Captions = X_Captions[indices, :, :] 
indices_Set = {*indices}
y_RelevantQuestions = [ques for i, ques in enumerate(y_RelevantQuestions) if i in indices_Set]

correctEdits = Array('i', [0 for i in range(15)])
count = Value('i', 0)

print("Editing...")
print(len(y_RelevantQuestions))
zippedlists = list(zip(X_Captions, X_RawQuestionEdits, y_RelevantQuestions))
random.shuffle(zippedlists)
for cap, rawQuesList, relQues in zippedlists:
    Process(target=parallel_edit, args=(cap, rawQuesList, relQues, correctEdits, count)).start()

for i in range(1, len(correctEdits)):
    correctEdits[i] += correctEdits[i-1]

print("Final accuracies:")
print('\n'.join([str(rank / count) for rank in correctEdits]))
print("count:", count)
print("total:", len(y_RelevantQuestions))
