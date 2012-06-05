import sys
import Perceptron

class FeatureFactory:
    def __init__(self):
        self.list_of_common_names = []

    def compute_sentence_features_eng(self, sentence):
        for instance in sentence.instances:
            print instance.word


    def compute_word_features_eng(self, word, sentence):
        pass    