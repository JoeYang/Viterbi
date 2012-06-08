import sys
import Perceptron
import re

class FeatureFactory:
    
    def __init__(self):
        self.list_of_loc_names = []
        f = open('location_names.txt', 'r')
        for line in f:
            self.list_of_loc_names.append((line.split()[0]).lower())

        self.list_of_titles = ["Mr", "Mr.", "Miss", "Ms", "Mrs", "Mrs.", \
                                "Dr.", "Dr", "President", "Minister"]

        self.list_of_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        self.list_of_months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', \
                            'august', 'september', 'october', 'november', 'december']
        self.list_of_miscs = []
        self.list_of_org = []
        
        f = open('misc.txt', 'r')
        for line in f:
            self.list_of_miscs.append((line.split()[0]).lower())
        
        f = open('org.txt', 'r')
        for line in f:
            self.list_of_org.append((line.split()[0]).lower())

    def features_update(self, sentence):
        for index in xrange(len(sentence.instances)):
            self.instance_update(index, sentence)

    def instance_update(self, index, sentence):
        
        currInstance = sentence.instances[index]
        if index != 0:
            prev_guess = sentence.instances[index-1].predicted_label
        else:
            prev_guess = None

        if index != sentence.size()-1:
            nest_guess = sentence.instances[index+1].predicted_label
        else:
            nest_guess = None    

        currInstance.features[1] = (prev_guess == 'O')
        currInstance.features[2] = (prev_guess == 'I-PER')   
        currInstance.features[3] = (prev_guess == 'I-ORG')
        currInstance.features[4] = (prev_guess == 'I-LOC')
        currInstance.features[5] = (prev_guess == 'I-MISC')
        currInstance.features[6] = (prev_guess == 'B-PER')
        currInstance.features[7] = (prev_guess == 'B-ORG')
        currInstance.features[8] = (prev_guess == 'B-LOC') 
        currInstance.features[9] = (prev_guess == 'B-MISC')
        
        currInstance.features[10] = (nest_guess == 'O')
        currInstance.features[11] = (nest_guess == 'I-PER')   
        currInstance.features[12] = (nest_guess == 'I-ORG')
        currInstance.features[13] = (nest_guess == 'I-LOC')
        currInstance.features[14] = (nest_guess == 'I-MISC')
        currInstance.features[15] = (nest_guess == 'B-PER')
        currInstance.features[16] = (nest_guess == 'B-ORG')
        currInstance.features[17] = (nest_guess == 'B-LOC') 
        currInstance.features[18] = (nest_guess == 'B-MISC')
        
        

    def compute_sentence_features_eng(self, sentence):
        for index in xrange(len(sentence.instances)):
            self.compute_word_features_eng(index, sentence)


    def compute_word_features_eng(self, index, sentence):
        
        currInstance = sentence.instances[index]
        
        if index != 0:
            prevInstance = sentence.instances[index-1]
        else:
            prevInstance = None
        
        if index != sentence.size()-1:
            nextInstance =  sentence.instances[index+1]
        else:
            nextInstance = None    

        # bias feature
        currInstance.addFeature(1)

        if prevInstance != None:
            prev_word = prevInstance.word
            prev_POS_tag = prevInstance.POS_tag
            prev_chunk = prevInstance.Chunk_tag
            prev_guess = prevInstance.predicted_label
        else:
            prev_word = None
            prev_POS_tag = None
            prev_chunk = None
            prev_guess = None
        
        if nextInstance != None:
            next_word = nextInstance.word
            next_POS_tag = nextInstance.POS_tag
            next_chunk = nextInstance.Chunk_tag
            nest_guess = nextInstance.predicted_label
        else:
            next_word = None
            next_POS_tag = None
            next_chunk = None
            nest_guess = None    

        currInstance.addFeature(prev_guess == 'O')
        currInstance.addFeature(prev_guess == 'I-PER')
        currInstance.addFeature(prev_guess == 'I-ORG')
        currInstance.addFeature(prev_guess == 'I-LOC')
        currInstance.addFeature(prev_guess == 'I-MISC')
        currInstance.addFeature(prev_guess == 'B-PER')
        currInstance.addFeature(prev_guess == 'B-ORG')
        currInstance.addFeature(prev_guess == 'B-LOC') 
        currInstance.addFeature(prev_guess == 'B-MISC')
        
        
        currInstance.addFeature(nest_guess == 'O')
        currInstance.addFeature(nest_guess == 'I-PER')
        currInstance.addFeature(nest_guess == 'I-ORG')
        currInstance.addFeature(nest_guess == 'I-LOC')
        currInstance.addFeature(nest_guess == 'I-MISC')
        currInstance.addFeature(nest_guess == 'B-PER')
        currInstance.addFeature(nest_guess == 'B-ORG')
        currInstance.addFeature(nest_guess == 'B-LOC') 
        currInstance.addFeature(nest_guess == 'B-MISC')
            

        # previous word features        
        currInstance.addFeature(prevInstance == None)
        # next word features  
        currInstance.addFeature(nextInstance == None)

        currInstance.addFeature(prev_word in self.list_of_titles)
        currInstance.addFeature(prev_POS_tag == 'EX')
        currInstance.addFeature(prev_POS_tag == 'IN' and currInstance.POS_tag == 'NNP')
        currInstance.addFeature(prev_POS_tag == 'DT')
        currInstance.addFeature(prev_POS_tag == '(')
        currInstance.addFeature(next_POS_tag == ')')
        currInstance.addFeature(prev_POS_tag == '(' and next_POS_tag == ')')
        currInstance.addFeature(prev_chunk == 'I-PP')

        currInstance.addFeature(next_POS_tag == 'EX')
        currInstance.addFeature(next_POS_tag == 'POS')
        currInstance.addFeature(next_POS_tag == 'RBR')
        currInstance.addFeature(next_POS_tag == 'JJ')
        currInstance.addFeature(next_POS_tag == 'PRP$')
        currInstance.addFeature(next_POS_tag == '$')
        currInstance.addFeature(next_chunk == 'I-ADJP')

        # current word POS Tag features 
        currInstance.addFeature(currInstance.POS_tag == 'NNP')    
        currInstance.addFeature(currInstance.POS_tag == '$')
        currInstance.addFeature(currInstance.POS_tag == '-X-')
        currInstance.addFeature(currInstance.POS_tag == 'EX')
        currInstance.addFeature(currInstance.POS_tag == 'PRP')
        currInstance.addFeature(currInstance.POS_tag == 'POS')
        currInstance.addFeature(currInstance.POS_tag == 'MD')
        currInstance.addFeature(currInstance.POS_tag == 'WP')
        currInstance.addFeature(currInstance.POS_tag == 'TO')
        currInstance.addFeature(currInstance.POS_tag == 'WTD')
        currInstance.addFeature(not currInstance.POS_tag.isalpha())
        
        try:
            currInstance.addFeature(not next_POS_tag.isalpha())
        except:
            currInstance.addFeature(0) 
        
        currInstance.addFeature(currInstance.POS_tag == 'NN' and currInstance.Chunk_tag == "I-NP")
        currInstance.addFeature(currInstance.Chunk_tag == 'I-ADVP')
        currInstance.addFeature(currInstance.Chunk_tag == 'I-PRT')
        currInstance.addFeature(currInstance.Chunk_tag == 'B-VP')
        currInstance.addFeature(currInstance.Chunk_tag == 'B-PP')
        currInstance.addFeature(currInstance.Chunk_tag == 'I-CONJP')
        currInstance.addFeature(currInstance.Chunk_tag == 'B-SBAR')
        currInstance.addFeature(currInstance.Chunk_tag == 'I-PP')
        currInstance.addFeature(currInstance.Chunk_tag == 'B-NP')
        currInstance.addFeature(currInstance.Chunk_tag == 'I-NP')
        currInstance.addFeature(currInstance.Chunk_tag == 'I-SBAR')
        currInstance.addFeature(currInstance.Chunk_tag == 'I-ADJP')
        
        try:
            currInstance.addFeature(currInstance.POS_tag == 'NNP' and next_word.isalpha())    
        except:
            currInstance.addFeature(0)

        # upper case
        word = currInstance.word
        currInstance.addFeature(word[0].isupper() and word[1:].islower())
        currInstance.addFeature(word.islower())
        currInstance.addFeature(word.isupper())
        currInstance.addFeature(word.isalpha())
        if not word.isalpha():
            currInstance.addFeature(1)
        else:
            currInstance.addFeature(0)
        currInstance.addFeature(word.isdigit())
        if '-' in word:
            gagaga = word.split('-')
            count = len(gagaga)
            for k in gagaga:
                if k.isdigit():
                    count -= 1
            currInstance.addFeature(count == 0)
        else:
            currInstance.addFeature(0)
        
        
        # days or months
        currInstance.addFeature(word.lower() in self.list_of_days)
        currInstance.addFeature(word.lower() in self.list_of_months) 
        currInstance.addFeature(word.lower() in self.list_of_loc_names)
        currInstance.addFeature(word.lower() in self.list_of_miscs)
        currInstance.addFeature(word.lower() in self.list_of_org)
        


        # combo
        currInstance.addFeature(next_POS_tag == currInstance.POS_tag)
        currInstance.addFeature(prev_POS_tag == currInstance.POS_tag)
        currInstance.addFeature(next_chunk== currInstance.Chunk_tag)
        currInstance.addFeature(prev_chunk == currInstance.Chunk_tag)
        currInstance.addFeature(next_POS_tag == 'NNS' and currInstance.POS_tag == 'NNP')
        currInstance.addFeature(currInstance.POS_tag == 'NNP' and currInstance.Chunk_tag == 'I-NP' and prev_POS_tag == 'DT')
        currInstance.addFeature(currInstance.POS_tag == 'NNP' and next_chunk == 'I-VP')
        currInstance.addFeature(currInstance.POS_tag == 'NNP' and next_POS_tag == 'CD')
        currInstance.addFeature(currInstance.POS_tag == 'NNP' and currInstance.Chunk_tag == 'I-NP')
        currInstance.addFeature(currInstance.POS_tag == 'NNP' and currInstance.Chunk_tag == 'I-NP' and next_chunk == 'I-VP')
        currInstance.addFeature(currInstance.POS_tag == 'NNP' and currInstance.Chunk_tag == 'I-NP' and prev_chunk == 'I-NP')

        
    
        







