import sys
from Perceptron import Perceptron


def main(argv):
    train_dir = argv[0]
    test_dir = argv[1]

    try:
        train_file = open(train_dir, 'r')
        test_file = open(test_dir, 'r')
    except:
        print "Error: file doesn't open"
        print "Usage: python NER.py trainfile testfile"
        exit(0)

    print 'Perceptron starting...\nTraining File: %s\nTesting File %s' % (train_dir, test_dir)
    
    model = Perceptron(1)
    model.read_data(train_file, test_file)

    train_file.close()
    test_file.close()

    model.computeFeatures()

    model.train()
    model.test()

    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Error: Wrong Number of Arguments"
        print "Usage: python NER.py trainfile testfile"
        exit(0)
    main(sys.argv[1:])    