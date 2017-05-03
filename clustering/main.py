import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg
import elice_utils
import math

ELICE = False
FILE_PREFIX = ''

def read_data(filename):
    X = []
    # Read the dataset here...
    
    with open(filename) as fp:
        N = int(fp.readline())
        for line_idx in range(N):
            x_i = [float(x) for x in fp.readline().strip().split()]
            X.append(x_i)
    
    # X must be the N * 2 numpy array.
    X = np.array(X)
    return X

def gaussian(mu, sigma, x):
    # Use this function to get the density of multivariate normal distribution
    from scipy.stats import multivariate_normal
    var = multivariate_normal(mean=mu, cov=sigma, allow_singular=True)
    
    # add an extremely small probability to avoid zero probability
    return var.pdf(x) + 10**-20 

def get_initial_random_state(X, K):
    import random
    random.seed(0)
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    
    mu = []
    sigma = []
    pi = []
    
    for k in range(K):
        x1 = random.uniform(min(X1), max(X1))
        x2 = random.uniform(min(X2), max(X2))
        mu.append([x1, x2])
        sigma.append([[1, 0],
                      [0, 1]])
        pi.append(1 / K)
    
    mu = np.array(mu)
    sigma = np.array(sigma)
    pi = np.array(pi)
    return (mu, sigma, pi)

def kmeans(X, theta):
    mu, sigma, pi = theta
    K = len(mu)

    return (mu, sigma, pi)

def expected_complete_LL(X, R, K, theta):
    ll = 0
    
    return ll

def expect(X, theta):
    # unpack
    mu, sigma, pi = theta
    R = []
    
    return np.array(R)

def maximize(X, R, K):
    mu = []
    sigma = []
    pi = []
    
    return (np.array(mu), np.array(sigma), np.array(pi))

def EM(X, K, init_theta):
    LL = 0

    return LL

def find_best_k(X):
    best_LL = None
    best_theta = None
    best_K = None
    best_R = None
    
    return best_K, best_theta, best_LL, best_R

def draw_dataset(X, R):
    # Code from Jooyeon's homework
    filename = FILE_PREFIX + "dataset.svg"

    colors = []
    for r_i in R:
        max_r = max(r_i)
        for k in range(len(r_i)):
            if r_i[k] == max_r:
                colors.append(k + 1)
                break

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_color('#999999')
    plt.gca().spines['left'].set_color('#999999')
    plt.xlabel('x1', fontsize=20, color='#555555'); plt.ylabel('x2', fontsize=20, color='#555555')
    plt.tick_params(axis='x', colors='#777777')
    plt.tick_params(axis='y', colors='#777777')    
    plt.scatter(X[:, 0], X[:, 1], c = colors, edgecolor='none', s=30)

    plt.savefig(filename)
    elice_utils.send_image(filename)

    plt.close()
    
def main():
    X = read_data(FILE_PREFIX + "example.txt")
    best_K, best_theta, best_LL, best_R = find_best_k(X)
    draw_dataset(X, best_R)
    
if __name__ == '__main__':
    global ELICE
    global FILE_PREFIX
    if not ELICE:
        FILE_PREFIX = './data/'
    main()
