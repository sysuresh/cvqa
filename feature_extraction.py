import nltk
import gensim.models.keyedvectors as word2vec
import numpy as np
import pandas as pd
import random
def extractFeatures(questionRows, totalLength, maxLength, w2v, shuffle=False):

    excludeWordList = ['is','a','the','what','that','to','who','why']
    X_captions = []
    x_questions = []
    y = []
    errors = []
    wordVectorSize = 300
    for i, questionRow in questionRows.iterrows():
        imageFilename = questionRow['image']
        caption = questionRow['caption']
        error = questionRow['error'].lower()
        errors = error.split(",")
        questionWords = [w for w in nltk.word_tokenize(questionRow['question'].lower()) if '?' not in w]
        relQuestionWords = [w for w in nltk.word_tokenize(questionRow['correctquestion'].lower()) if '?' not in w]
        captionWords = caption.split(' ')

        rectify = lambda l : [w.lower() for w in l if w.lower() in w2v and w.lower() not in excludeWordList]

        newQuestionWords = rectify(questionWords)
        newCaptionWords = rectify(captionWords)
        newRelQuestionWords = rectify(relQuestionWords)

        if shuffle:
            together = list(zip(newQuestionWords, newRelQuestionWords))
            random.shuffle(together)
            try:
                newQuestionWords, newRelQuestionWords = zip(*together)
            except Exception:
                print(questionRow)
                continue

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
        X_captions.append(captionFeature)
        x_questions.append(questionFeature)
        y.append([0.0]+[1.0 if w.lower() in errors else 0.0 for w in newQuestionWords]+[0.0 for w in range(maxLength-len(newQuestionWords))])
        X_captions.append(captionFeature)
        x_questions.append(questionFeatureRelevant)
        y.append([1.0]+[0.0 for w in range(maxLength)])

    return np.asarray(x_questions),np.asarray(X_captions),y, errors

