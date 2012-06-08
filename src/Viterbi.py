from collections import defaultdict
import Perceptron
import operator

class Viterbi:

    def __init__(self):
        self.pairs = defaultdict(lambda:0)
        self.prior = defaultdict(lambda:0)
        self.pair_prob = defaultdict(lambda:0)
        self.labels = []
        self.total = 0

    def train(self, list_of_klasses, training_sentences):
        for sentence in training_sentences:
            for instance_1, instance_2 in zip(sentence.instances, sentence.instances[1:]):
                self.pairs[(instance_1.label, instance_2.label)] += 1
                self.prior[instance_1.label] += 1
                self.pair_prob[(instance_1.label, instance_2.label)] = float(self.pairs[(instance_1.label, instance_2.label)]) / self.prior[instance_1.label]
                self.total += 1
        self.labels = list_of_klasses

    def fast(self, sentence, klasses):
        path = []

        instance = sentence.instances[0]
        best_klass = klasses[0]
        best_score = self.feature_klass(instance, best_klass)
        
        for klass in klasses:
            score = self.feature_klass(instance, klass)
            if score > best_score:
                best_score = score
                best_klass = klass
        path.append(best_klass.tag)
        
        for index in range(1, len(sentence.instances)):
            best_klass = klasses[0]
            best_score = self.feature_klass(instance, best_klass)
            prev_tag = path[index-1]
            instance = sentence.instances[index]
            for klass in klasses:
                score = self.feature_klass(instance, klass) * \
                        float(self.pairs[(prev_tag, klass.tag)]) / self.prior[prev_tag]
                if score > best_score:
                    best_score = score
                    best_klass = klass
            path.append(best_klass.tag)

        return path

    def viterbi(self, sentence, klasses):
        path = []
        for index in xrange(len(sentence.instances)): 
            instance = sentence.instances[index]
            path.append(defaultdict(tuple))
            if index == 0:
                for klass in klasses:
                    score = self.feature_klass(instance, klass) * 1
                    path[index][klass.tag] = (score, None)
            else:    
                for curr_klass in klasses:  
                    best_tag = 'O'
                    best_score = path[index-1]['O'][0] + self.feature_klass(instance, curr_klass) * self.pair_prob[('O', 'O')]
                    for prev_klass in klasses:
                        prev_tag = prev_klass.tag
                        score = path[index-1][prev_tag][0] + self.feature_klass(instance, curr_klass) * self.pair_prob[(prev_tag, curr_klass.tag)]
                        if score > best_score:
                            best_score = score
                            best_tag = prev_tag        
                    path[index][curr_klass.tag] = (best_score, best_tag)

        return self.decode(path, len(path))              
                
    def decode(self, path_info, n):
        guesses = []
        index = n-1
        best_klass = 'O'
        (best_score, best_tag) = path_info[n-1][best_klass]

        for klass in path_info[index]:
            (score, tag) = path_info[index][klass]
            if score > best_score:
                best_score = score
                best_klass = klass
                best_tag = tag            

        guesses.insert(0, best_klass)       
        
        while index > 0:
            guesses.insert(0, best_tag)
            index -= 1
            (score, tag) = path_info[index][best_tag]            
            best_tag = tag

        return guesses            

    def feature_klass(self, instance, klass):
        score = 0
        for key in instance.features:
            score += instance.features[key] * klass.weights[key]
        return score    
