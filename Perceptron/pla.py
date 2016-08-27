import numpy as np
import random
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, N):
        # Random linearly separable data
        xA, yA, xB, yB = [random.uniform(-1, 1) for i in range(4)]
        self.V = np.array([xB*yA-xA*yB, yB-yA, xA-xB])
        self.X = self.generate_points(N)
        self.it = 0
        plt.figure(figsize=(6, 6))

    def generate_points(self, N):
        X = []
        for i in range(N):
            x1, x2 = [random.uniform(-1, 1) for i in range(2)]

            # The heaviside function phi(x) = 1 if z > thresh else -1 cna be re-written as
            # phi(x) = 1 if z > 0 else -1 if we move the introduce x0 and move thresh to the left side of the formula,
            # i.e. phi(x) = w0 * x0 + w1 * x1 + w2 * x2  and here w0 = -thresh
            x = np.array([1, x1, x2])
            s = int(np.sign(self.V.T.dot(x)))
            X.append((x, s))
        return X

    def plot(self, vec=[], mispts=()):

        l = np.linspace(-1, 1)

        plt.hold(False)
        plt.cla()
        # plt.plot(l, a*l+b, 'k-')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.hold(True)

        cols = {1: 'r', -1: 'b'}
        for x, s in self.X:
            plt.plot(x[1], x[2], cols[s]+'o')
        if mispts:
            plt.plot(mispts[0][1], mispts[0][2], 'yx', markersize=10)
        if any(vec):
            aa, bb = -vec[1]/vec[2], -vec[0]/vec[2]
            plt.plot(l, aa*l+bb, 'g-', lw=2)

        plt.title('N = %s, Iteration %s' % (len(self.X), self.it))
        plt.pause(1)

    def find_misclassified_point(self, vec, pts=None):
        mispts = []
        for x,s in self.X:
            if np.sign(vec.T.dot(x)) != s:
                mispts.append((x, s))
        return mispts

    def pla(self, save=False):
        # Initialize the weigths to zeros
        w = np.zeros(3)
        self.plot(vec=w)

        # X, N = self.X, len(self.X)
        self.it = 0

        # Find misclassfied points
        mispts = self.find_misclassified_point(w)
        # Iterate until all points are correctly classified
        while len(mispts) != 0:
            self.it += 1

            # Pick a random misclassified point
            x, s = mispts[random.randrange(0, len(mispts))]

            # Update weights
            w += s*x

            # Plot result
            self.plot(vec=w, mispts=(x, s))
            if save:
                plt.savefig('p_N%s_it%s' % (len(self.X), self.it), dpi=200, bbox_inches='tight')

            # Find misclassfied points
            mispts = self.find_misclassified_point(w)

        self.w = w