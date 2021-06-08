## Three methods, based on Principal Component Analysis, for computing spatio-temporal patterns of variability in geospatial time series data
## Author: William Gregory
## Last modified: 12/03/2021

import numpy as np
from scipy.spatial.distance import cdist

class PCA:
    def __init__(self,data,weights,latlon=True):
        """
        The input 'data' are expected to be de-trended (zero-mean)
        and in the format x,y,t if an area grid, or lat,lon,t for
        a lat-lon grid.
        """
        if latlon:
            self.weights = np.sqrt(np.cos(np.radians(weights)))
            self.weights[np.isnan(self.weights)] = np.sqrt(np.cos(np.radians(90)))
        else:
            self.weights = np.sqrt(weights)
        self.data = np.multiply(data,self.weights[:,:,np.newaxis])
        self.IDs = np.where(np.nanmax(np.abs(self.data),axis=2)>0)
        self.n = np.shape(self.IDs)[1]
        self.dimX,self.dimY,self.dimT = self.data.shape
        
    def EOFA(self,N=1):
        """
        Empirical Orthogonal Function (EOF) Analysis.
        The Singular Value Decomposition of a given climate data field:
        EOFs are the dominant spatial patterns of variability of the input signal
        Principal Components (PCs) are the amplitude and sign of the modes at each point in time 

        Input variables:
        data: array containing gridded de-trended observations, of shape (x,y,t) or (lat,lon,t)
        weights: array of shape (x,y) to weight grid cells based on area or latitude
        N: number of principal components to return (default is to return the leading PC)
        latlon: Boolean to check if data are on lat/lon grid. False implies area gridded (km^2)
        returns the PCs and EOFs
        """
        
        U,S,V = np.linalg.svd(self.data[self.IDs],full_matrices=False)
        EOFs = np.zeros((self.dimX,self.dimY,N))
        EOF = U/np.atleast_2d(self.weights[self.IDs]).T
        EOFs[self.IDs] = EOF[:,:N]

        return V.T[:,:N],EOFs
    
    def SSA(self,N=1,q=12):
        """
        Singular Spectrum Analysis (SSA), also known as Extended EOF analysis.
        SSA decomposes the signal into spatio-temporal modes of variability

        Input variables:
        data: array containing gridded de-trended observations, of shape (x,y,t) or (lat,lon,t)
        weights: array of shape (x,y) to weight grid cells based on area or latitude
        N: number of modes to return (default is to return the leading PC)
        latlon: Boolean to check if data are on lat/lon grid. False implies area gridded (km^2)
        q: time window that makes up each spatio-temporal pattern (q=1 is equivalent to EOFA)
        returns the q-snapshot spatiotemporal patterns reconstructed back the physical space of the data
        """

        X = self.data[self.IDs]
        X = np.flip(X.T[(np.arange(q))+np.arange(np.max(self.dimT-(q-1),0)).reshape(-1,1)].reshape(self.dimT-q+1,self.n*q).T,0)
        U,S,V = np.linalg.svd(X,full_matrices=False)

        EEOFs = np.zeros((self.dimX,self.dimY,self.dimT,N))
        X_rec = np.zeros((self.n,self.dimT,self.dimT-q+1))
        for k in range(self.dimT-q+1):
            Xk = S[k]*np.dot(np.atleast_2d(U[:,k]).T,np.atleast_2d(V.T[:,k]))
            offset1 = 0
            offset2 = 1
            for t in range(self.dimT):
                if t == 0:
                    X_rec[:,t,k] = Xk[-self.n:,0]
                elif (t > 0) & (t < q-1):
                    x_kj = np.zeros((self.n,t+1))
                    start = self.n*q - (t+1)*self.n
                    for l in range(t+1):
                        x_kj[:,l] = Xk[start:start+self.n,l]
                        start += self.n
                    X_rec[:,t,k] = np.mean(x_kj,1)
                elif (t >= q-1) & (t <= self.dimT-q):
                    x_kj = np.zeros((self.n,q))
                    start = 0
                    for l in range(offset1,q+offset1):
                        x_kj[:,l-offset1] = Xk[start:start+self.n,l]
                        start += self.n
                    offset1 += 1
                    X_rec[:,t,k] = np.mean(x_kj,1)
                elif (t > self.dimT-q) & (t < self.dimT-1):
                    x_kj = np.zeros((self.n,q-offset2))
                    start = 0
                    for l in range(offset1,(q-offset2)+offset1):
                        x_kj[:,l-offset1] = Xk[start:start+self.n,l]
                        start += self.n
                    offset1 += 1
                    offset2 += 1
                    X_rec[:,t,k] = np.mean(x_kj,1)
                elif t == self.dimT-1:
                    X_rec[:,t,k] = Xk[:self.n,-1]

        EEOFs[self.IDs] = (np.flip(X_rec,0)/self.weights[self.IDs][:,np.newaxis,np.newaxis])[:,:,:N]
        
        return EEOFs
    
    def NLSA(self,N=1,q=12,l=None):
        """
        Non-linear Laplacian Spectral Analysis.
        Combines SSA with Laplacian eigenmaps and diffusion maps, creating a
        nonlinear manifold generalisation of SSA

        Input variables:
        data: array containing gridded de-trended observations, of shape (x,y,t)
        weights: array of shape (x,y) to weight grid cells based on area or latitude
        N: number of modes to return (default is to return the leading PC)
        latlon: Boolean to check if data are on lat/lon grid. False implies area gridded (km^2)
        q: time window that makes up each spatio-temporal pattern
        l: number of leading Laplacian eigenfunctions to project the observations on to
        """

        X = self.data[self.IDs]
        X = np.flip(X.T[(np.arange(q))+np.arange(np.max(self.dimT-(q-1),0)).reshape(-1,1)].reshape(self.dimT-q+1,self.n*q).T,0)
        K = np.zeros((self.dimT-q,self.dimT-q))
        for i in range(self.dimT-q):
            xi = np.atleast_2d(X[:,1+i])
            xi_m1 = np.atleast_2d(X[:,1+i-1])
            elli = cdist(xi,xi_m1,'euclidean')
            for j in range(self.dimT-q):
                xj = np.atleast_2d(X[:,1+j])
                xj_m1 = np.atleast_2d(X[:,1+j-1])
                ellj = cdist(xj,xj_m1,'euclidean')
                K[i,j] = np.exp(-cdist(xi,xj,'sqeuclidean')/(elli*ellj))

        Qi,Qj = np.meshgrid(np.sum(K,axis=1),np.sum(K,axis=1))
        K_tilde = K/(Qi*Qj)
        P = K_tilde/np.atleast_2d(np.sum(K_tilde,axis=1)).T #transition (probability) matrix
        L = np.eye(self.dimT-q) - P
        Lambda, phi = np.linalg.eig(L) #Lϕ = λϕ
        Z, mu = np.linalg.eig(P) #μP = μ
        mu = mu[:,np.isclose(Z,1,atol=1e-12)].ravel() #take eigenvector corresponding to where eigenvalue = 1.
        mu = mu / np.sum(mu) #to make the sum of μ equal to 1 (it is a vector of probabilities)
        
        if l is None:
            l = self.dimT-q
        else:
            l = l
            
        A = np.linalg.multi_dot([X[:,1:],np.diag(mu),phi[:,-l:]]) #project X onto leading l Laplacian eigenfunctions
        U,S,V = np.linalg.svd(A,full_matrices=False)

        EEOFs = np.zeros((self.dimX,self.dimY,self.dimT,N)) ; EEOFs[:,:,0,:] = np.nan 
        X_rec = np.zeros((self.n,self.dimT,self.dimT-q)) ; X_rec[:,0,:] = np.nan
        #note that we set the first time stamp (i.e. year 1) to nan, as we have used this to compute the 
        #phase velocities (elli,ellj). 
        for k in range(self.dimT-q):
            Xk = S[k]*np.dot(np.atleast_2d(U[:,k]).T,np.atleast_2d(V.T[:,k]))
            offset1 = 0
            offset2 = 1
            for t in range(1,self.dimT):
                if t == 1:
                    X_rec[:,t,k] = Xk[-self.n:,0]
                elif (t > 1) & (t < q):
                    x_kj = np.zeros((self.n,t+1))
                    start = self.n*q - (t+1)*self.n
                    for l in range(t+1):
                        x_kj[:,l] = Xk[start:start+self.n,l]
                        start += self.n
                    X_rec[:,t,k] = np.mean(x_kj,1)
                elif (t >= q) & (t <= self.dimT-q):
                    x_kj = np.zeros((self.n,q))
                    start = 0
                    for l in range(offset1,q+offset1):
                        x_kj[:,l-offset1] = Xk[start:start+self.n,l]
                        start += self.n
                    offset1 += 1
                    X_rec[:,t,k] = np.mean(x_kj,1)
                elif (t > self.dimT-q) & (t < self.dimT-1):
                    x_kj = np.zeros((self.n,q-offset2))
                    start = 0
                    for l in range(offset1,(q-offset2)+offset1):
                        x_kj[:,l-offset1] = Xk[start:start+self.n,l]
                        start += self.n
                    offset1 += 1
                    offset2 += 1
                    X_rec[:,t,k] = np.mean(x_kj,1)
                elif t == self.dimT-1:
                    X_rec[:,t,k] = Xk[:self.n,-1]

        EEOFs[self.IDs] = (np.flip(X_rec,0)/self.weights[self.IDs][:,np.newaxis,np.newaxis])[:,:,:N]
        
        return EEOFs
