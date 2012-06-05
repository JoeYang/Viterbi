from collections import defaultdict         

class Instance:
    def __init__(self, word, POS_tag, Chunk_tag, gold_label):
        self.word = word
        self.POS_tag = POS_tag
        self.Chunk_tag = Chunk_tag
        self.label = gold_label
        self.features = defaultdict(lambda:0)    
   
