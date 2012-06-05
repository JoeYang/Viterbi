from collections import defaultdict 

class Klass:
    def __init__(self, tag):
        self.tag = tag
        self.weights = defaultdict(lambda:0)
        self.total = defaultdict(lambda:0)
    def update(self):
        pass