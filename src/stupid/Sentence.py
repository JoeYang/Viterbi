
    
class Sentence:
    def __init__(self):
        self.instances = []
        self.full_sentence = ''
    def add(self, instance):
        self.instances.append(instance)
        self.full_sentence += instance.word + ' '
    def size(self):
        return len(self.instances)
