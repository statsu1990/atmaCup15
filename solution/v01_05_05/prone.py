"""
以下を改修。乱数シードを指定できるようにした。
https://github.com/VHRanger/nodevectors/blob/master/nodevectors/prone.py

MIT License
Copyright (c) 2018 YOUR NAME
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import csrgraph as cg
import nodevectors as nv
import numpy as np
import scipy
from scipy import sparse, linalg
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd


class ProNE(nv.ProNE):
    def __init__(self, n_components=32, step=10, mu=0.2, theta=0.5, 
                exponent=0.75, verbose=True, random_state=None):
        """
        Fast first order, global method.

        Embeds by doing spectral propagation over an initial SVD embedding.
        This can be seen as augmented spectral propagation.

        Parameters :
        --------------
        step : int >= 1
            Step of recursion in post processing step.
            More means a more refined embedding.
            Generally 5-10 is enough
        mu : float
            Damping factor on optimization post-processing
            You rarely have to change it
        theta : float
            Bessel function parameter in Chebyshev polynomial approximation
            You rarely have to change it
        exponent : float in [0, 1]
            Exponent on negative sampling
            You rarely have to change it
        References:
        --------------
        Reference impl: https://github.com/THUDM/ProNE
        Reference Paper: https://www.ijcai.org/Proceedings/2019/0594.pdf
        """
        self.n_components = n_components
        self.step = step
        self.mu = mu
        self.theta = theta
        self.exponent = exponent
        self.verbose = verbose
        self.random_state = random_state

        super().__init__(n_components=n_components, step=step, mu=mu, theta=theta, 
                exponent=exponent, verbose=verbose)

    def fit_transform(self, graph):
        """
        NOTE: Currently only support str or int as node name for graph
        Parameters
        ----------
        nxGraph : graph data
            Graph to embed
            Can be any graph type that's supported by CSRGraph library
            (NetworkX, numpy 2d array, scipy CSR matrix, CSR matrix components)
        """
        G = cg.csrgraph(graph)
        features_matrix = self.pre_factorization(G.mat,
                                                 self.n_components,
                                                 self.exponent,
                                                 self.random_state)
        vectors = ProNE.chebyshev_gaussian(
            G.mat, features_matrix, self.n_components,
            step=self.step, mu=self.mu, theta=self.theta)
        self.model = dict(zip(G.nodes(), vectors))
        return vectors

    
    def fit(self, graph):
        """
        NOTE: Currently only support str or int as node name for graph
        Parameters
        ----------
        nxGraph : graph data
            Graph to embed
            Can be any graph type that's supported by CSRGraph library
            (NetworkX, numpy 2d array, scipy CSR matrix, CSR matrix components)
        """
        G = cg.csrgraph(graph)
        features_matrix = self.pre_factorization(G.mat,
                                                 self.n_components, 
                                                 self.exponent,
                                                 self.random_state)
        vectors = ProNE.chebyshev_gaussian(
            G.mat, features_matrix, self.n_components,
            step=self.step, mu=self.mu, theta=self.theta)
        self.model = dict(zip(G.nodes(), vectors))

    @staticmethod
    def tsvd_rand(matrix, n_components, random_state=None):
        """
        Sparse randomized tSVD for fast embedding
        """
        l = matrix.shape[0]
        # Is this csc conversion necessary?
        smat = sparse.csc_matrix(matrix)
        U, Sigma, VT = randomized_svd(smat, 
            n_components=n_components, 
            n_iter=5, random_state=random_state)
        U = U * np.sqrt(Sigma)
        U = preprocessing.normalize(U, "l2")
        return U

    @staticmethod
    def pre_factorization(G, n_components, exponent, random_state=None):
        """
        Network Embedding as Sparse Matrix Factorization
        """
        C1 = preprocessing.normalize(G, "l1")
        # Prepare negative samples
        neg = np.array(C1.sum(axis=0))[0] ** exponent
        neg = neg / neg.sum()
        neg = sparse.diags(neg, format="csr")
        neg = G.dot(neg)
        # Set negative elements to 1 -> 0 when log
        C1.data[C1.data <= 0] = 1
        neg.data[neg.data <= 0] = 1
        C1.data = np.log(C1.data)
        neg.data = np.log(neg.data)
        C1 -= neg
        features_matrix = ProNE.tsvd_rand(C1, n_components=n_components, random_state=random_state)
        return features_matrix
