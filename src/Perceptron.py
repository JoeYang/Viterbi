from DataReader import DataReader
from collections import defaultdict
from FeatureFactory import FeatureFactory
from Viterbi import Viterbi
import sys

class Klass:
    
    def __init__(self, tag):
        self.tag = tag
        self.weights = defaultdict(lambda:0)
        self.correction = defaultdict(lambda:0)
        self.total = defaultdict(lambda:0)
        self.iteration = 0
        self.TP = 0
        self.FP = 0
        self.FN = 0
    
    def adjust(self, features, instruction):
        
        if instruction == '+':
            for key in features:
                self.correction[key] += features[key]
        elif instruction == '-': 
            for key in features:
                self.correction[key] -= features[key]   

    def update(self):
        for key in self.correction:
            self.weights[key] += self.correction[key]
            self.correction[key]= 0
        
        for key in self.weights:
            self.total[key] += self.weights[key]    
        
        self.iteration += 1

    
    def average_weights(self):
        for key in self.total:
            self.weights[key] = float(self.total[key]) / self.iteration
        self.iteration = 0    


class Instance:
    
    def __init__(self, word, POS_tag, Chunk_tag, gold_label):
        self.word = word
        self.POS_tag = POS_tag
        self.Chunk_tag = Chunk_tag
        self.label = gold_label
        self.predicted_label = None
        self.features = defaultdict(lambda:0)
        self.feature_counter = 0
    
    def addFeature(self, fea_val):
        self.features[self.feature_counter] = fea_val
        self.feature_counter += 1
    
class EngInstance(Instance):
    
    def whatever(self):
        pass     

class GerInstance:
    
    def setBaseForm(self, baseform):
        self.baseform = baseform
    
class Sentence:
    
    def __init__(self):
        self.instances = []
        self.full_sentence = ''
    
    def add(self, instance):
        self.instances.append(instance)
        if self.full_sentence != '':
            self.full_sentence += ' '
        self.full_sentence += instance.word 
    
    def size(self):
        return len(self.instances)

class Perceptron:
    
    def __init__(self, language):
        self.total_labels = []
        self.klasses = []
        self.language = language
        self.train_sentences = []
        self.test_sentenses = []
        self.factory = FeatureFactory()
        self.viterbi = Viterbi()

    def read_data(self, train_file, test_file):
        self.read_training_data(train_file)
        self.read_testing_data(test_file)
    
    def read_training_data(self, train_file):
        list_of_training_instances = []
        new_sentence = Sentence()
        for line in train_file:      
            split = line.strip().split()
            if len(split) == 0 and new_sentence.size() != 0:
                if '-DOCSTART-' not in new_sentence.full_sentence:
                    self.train_sentences.append(new_sentence)
                new_sentence = Sentence()
            else:
                instance = EngInstance(split[0], split[1], split[2], split[3])
                list_of_training_instances.append(instance)
                new_sentence.add(instance)
                if split[3] not in self.total_labels:
                    self.total_labels.append(split[3])

        print 'total number of training instances',len(list_of_training_instances), \
                'total number of training sentences', len(self.train_sentences)  

        self.klasses_init()
        self.viterbi.train(self.total_labels, self.train_sentences)

    def klasses_init(self):
        for label in self.total_labels:
            self.klasses.append(Klass(label))

    def tag_klass(self, tag):
        for klass in self.klasses:
            if klass.tag == tag:
                return klass
        return None                

    def read_testing_data(self, test_file):
        list_of_testing_instances = []
        new_sentence = Sentence()
        for line in test_file:      
            split = line.strip().split()
            if len(split) == 0 and new_sentence.size() != 0:
                if '-DOCSTART-' not in new_sentence.full_sentence:
                    self.test_sentenses.append(new_sentence)
                new_sentence = Sentence()
            else:
                instance = EngInstance(split[0], split[1], split[2], split[3])
                list_of_testing_instances.append(instance)
                new_sentence.add(instance)

        print 'total number of testing instances',len(list_of_testing_instances), \
                'total number of testing sentences', len(self.test_sentenses)

    def computeFeatures(self):
        for sentence in self.train_sentences:
            self.factory.compute_sentence_features_eng(sentence)
        for sentence in self.test_sentenses:
            self.factory.compute_sentence_features_eng(sentence)

    def train(self):
        iteration = 0
        total = len(self.train_sentences)
        while iteration < 10:
            error = 0
            for i in range(len(self.train_sentences)):
                sentence = self.train_sentences[i] 
                path = self.classify(sentence)
                for index in range(len(sentence.instances)):
                    instance = sentence.instances[index]
                    if path[index] == instance.label:
                        instance.predicted_label = instance.label
                    else:
                        guess = self.tag_klass(path[index])
                        instance.predicted_label = path[index]
                        gold = self.tag_klass(instance.label)
                        error += 1
                        guess.adjust(instance.features, '-')
                        gold.adjust(instance.features, '+')
                self.factory.features_update(sentence)
                for klass in self.klasses:
                    klass.update()
            iteration += 1
            print 'Iteration %d: number of errors %d' % (iteration, error)
        for klass in self.klasses:
            klass.average_weights()                

    def classify(self, sentence):
        return self.viterbi.viterbi(sentence, self.klasses)
        

    def test(self):
        correct = 0
        wrong = 0
        report_summary = defaultdict(lambda:0)
        
        for i in range(len(self.train_sentences)):
            sentence = self.train_sentences[i] 
            path = self.classify(sentence)
            for index in range(len(sentence.instances)):
                instance = sentence.instances[index]
                instance.predicted_label = path[index]
            self.factory.features_update(sentence)
        
        for sentence in self.test_sentenses:
            path = self.classify(sentence)
            for index in range(len(sentence.instances)):
                instance = sentence.instances[index]
                guess = self.tag_klass(path[index])
                gold = self.tag_klass(instance.label)
                report_summary[(gold.tag, guess.tag)] += 1
                if guess.tag != gold.tag:
                    gold.FN += 1
                    guess.FP += 1
                    wrong += 1
                else:
                    gold.TP += 1
                    if guess.tag != 'O':
                        correct += 1

        for label_1 in self.total_labels:
            print label_1, "&",
        print    
        for label_1 in self.total_labels:
            print label_1, 
            for label_2 in self.total_labels:
                print "&", report_summary[(label_1, label_2)],
            print "\\\\ \\hline"
        print correct, wrong
        for klass in self.klasses:
            try:
                P = float(klass.TP)/(klass.TP + klass.FP) 
            except:
                P = 0
            try:        
                R = float(klass.TP)/(klass.TP + klass.FN) 
            except:
                R = 0
            try:        
                F = 2 * P * R /(P + R) * 100
            except:
                F = 0
            print "%s & %.2f & %.2f & %.2f" % (klass.tag, P * 100, R * 100, F)    
                



        