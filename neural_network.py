import numpy as np
from math import exp, log

"""this class implements a 2 layer fully connected neural network"""

class nn():
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    def loss(self, X, y=None, reg=0.0, lbd=2.0):

        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, _ = X.shape
        _, C = W2.shape

        """compute scores for the given batch of inputs"""
        Z = np.dot(X, W1) + b1
        H1 = np.maximum(0, Z)
        H2 = np.dot(H1, W2) + b2
        scores = H2
        
        
        if y is None:
            return scores
        
        """compute average loss of the given batch pf inputs and apply regularization"""
        scores -= scores.max()
        scores = np.exp(scores)
        scores_sum = np.sum(scores, axis=1)
        nums = scores[range(N), y]
        loss = - np.log(nums / scores_sum)
        loss = np.average(loss) + reg * (np.sum(W1 * W1) + np.sum(W2 * W2))


        """compute the gradients"""
        grads = {}

        zs = np.zeros((N, C))
        zs[range(N), y] = 1
        dscores = np.divide(scores, scores_sum.reshape(N, 1)) - zs
        dscores /= N
        dW2 = H1.T.dot(dscores)
        db2 = np.sum(dscores, axis=0)
        dh = dscores.dot(W2.T)
        dh[H1 == 0] = 0
        dW1 = X.T.dot(dh)
        db1 = np.sum(dh, axis=0)
        grads['W2'] = dW2 + 2 * reg * W2
        grads['b2'] = db2
        grads['W1'] = dW1 + 2 * reg * W1
        grads['b1'] = db1


        return loss, grads

    def train(self, X, y, X_val, y_val, learning_rate=1e-3, learning_rate_decay=0.95, reg=5e-6, num_iters=100, batch_size=200, verbose=False):

        num_train, _ = X.shape
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            #X_batch = None
            #y_batch = None

            choices = np.random.choice(num_train, batch_size)
            X_batch = X[choices]
            y_batch = y[choices]

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            self.params['W1'] -= grads['W1'] * learning_rate
            self.params['W2'] -= grads['W2'] * learning_rate

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }
    
    def predict(self, X):

        y_pred = np.argmax(np.dot(np.maximum(0, X.dot(self.params['W1']) + self.params['b1']), self.params['W2']) + self.params['b2'], axis=1)

        return y_pred
