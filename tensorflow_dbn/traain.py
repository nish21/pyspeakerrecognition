import numpy as np
import cPickle as cp

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score

from dbn.tensorflow import SupervisedDBNClassification
# use "from dbn import SupervisedDBNClassification" for computations on CPU with numpy

# Loading dataset
file_train = open('file_tr.npy','rb')
file_test = open('file_tt.npy','rb')
print("Starting to load")
X_train, X_test, Y_train, Y_test = cp.load(file_train), cp.load(file_test), np.load('file_rl.npy'), np.load('file_tl.npy')

print("Loaded this humongous data...now training")
# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[512, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)
classifier.fit(X_train, Y_train)

# Save the model
#classifier.save('model.pkl')

# Restore it
#classifier = SupervisedDBNClassification.load('model.pkl')

# Test
Y_pred = classifier.predict(X_test)
print 'Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred)
