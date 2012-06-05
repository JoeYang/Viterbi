from DataReader import DataReader
from collections import defaultdict
from FeatureFactory import FeatureFactory

class Klass:
    def __init__(self, tag):
        self.tag = tag
        self.weights = defaultdict(lambda:0)
        self.total = defaultdict(lambda:0)
    def update(self):
        pass
        

class Instance:
    def __init__(self, word, POS_tag, Chunk_tag, gold_label):
        self.word = word
        self.POS_tag = POS_tag
        self.Chunk_tag = Chunk_tag
        self.label = gold_label
        self.features = defaultdict(lambda:0)    

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
        self.full_sentence += instance.word + ' '
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
        
        self.klasses_init()

        print 'total number of training instances',len(list_of_training_instances), \
                'total number of training sentences', len(self.train_sentences)  

    def klasses_init(self):
        for label in self.total_labels:
            self.klasses.append(Klass(label))

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

    def train(self): 
        pass

    def test(self): 
        pass



        