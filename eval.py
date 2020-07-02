import h5py
import gensim.models.keyedvectors as word2vec
import json
import numpy as np
import keras
import keras.backend as K
from keras.models import *
from keras.layers import *
from keras.layers.core import *
from keras import metrics
from keras.regularizers import L1L2
from keras.callbacks import ModelCheckpoint
from attention_keras import Attention
import sys
import pickle

import config
from common import readJson, binary_loss

def vectorize(sentence):
    sentence = ''.join([char for char in sentence.lower() if char.isalpha() or char in (' ', "'")])
    sentence = [word for word in sentence.split(' ') if word not in config.excludeWordList and word in w2v]
    
    feature = np.zeros((1 + config.maxLength, config.wordVectorSize)) 
    for index, word in enumerate(sentence):
        feature[1 + index] = w2v[word]
    return feature, sentence

if __name__ == "__main__":

    model = load_model("saved_model.h5", custom_objects={'Attention':Attention, 'binary_loss':binary_loss})

    print("Loading Word2Vec Dictionary. This may take a long time...")
    #w2v = word2vec.KeyedVectors.load_word2vec_format(config.word2VecPath, binary=True)
    w2v = pickle.load(open("word2vec.bin", 'rb'))

    question = sys.argv[1]
    captionFile = sys.argv[2]
    index = int(sys.argv[3])

    imageCaptions = readJson(captionFile)
    caption = None
    for imageCaption in imageCaptions:
        if int(imageCaption["image_id"]) == index:
            caption = imageCaption["caption"]
            break
    if not caption:
        raise IndexError("Image index out of range")

    question_vector, preprocessed_question = vectorize(question)
    caption_vector, _ = vectorize(caption) 
    prediction = model.predict([question_vector[None, :, :], caption_vector[None, :, :]])
    prediction = prediction.flatten()

    if prediction[0] > 0.5:
        print("The question is relevant.")
    else:
        print("Irrelevant words:", ", ".join(np.where(prediction[1 : 1 + len(preprocessed_question)] > 0.5, preprocessed_question, np.array(["" for word in preprocessed_question])).tolist()).strip(", "))
