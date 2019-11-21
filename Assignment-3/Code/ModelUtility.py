import pickle
fileName = lambda cls: 'Model/'+cls+'.pickle'


def saveModel(classifier, classifierName):
  pickle.dump(classifier, open(fileName(classifierName), 'wb'),)

def loadModel(classifierName):
  return pickle.load(open(fileName(classifierName), 'rb'))
