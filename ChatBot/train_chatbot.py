import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import os

import random
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

print(os.getcwd())
#intents_file = open("intents.json").read()
#intents = json.loads(intents_file)