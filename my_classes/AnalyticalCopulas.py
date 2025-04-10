import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal, multivariate_t, t
from scipy.optimize import minimize
from functools import partial

class GaussianCopula():
    def __init__(self):
        self.Name = 'Gaussian Copula'
        self.corr = None
        self.data = None
        self.transformedData = None
        self.IsFitted = False
        pass

    def fitModel(self, data, correlationMeasure='pearson'):
        # Function to fit copula model

        print('Fitting Gaussian Copula model...')     
        self.corr = np.corrcoef(data, rowvar=False)
        self.data = data
        self.transformedData = norm.cdf(data)   
        self.IsFitted = True
        print(f'Estimated correlation coefficient:{self.corr[0,1]:.4f}')
        pass

    def evalCDF(self, u1,u2):
        # Function to evaluate CDF of copula
        if self.IsFitted == False:
            raise ValueError('Model not fitted')
        else:
            mean = [0, 0]
            marginal1 = norm.ppf(u1)
            marginal2 = norm.ppf(u2)
            CopulaFunction = multivariate_normal(mean=mean, cov = self.corr).cdf([marginal1, marginal2])
            return CopulaFunction
            
    def evalPDF(self, u1,u2):
        # Create the bivariate normal distribution pdf
        if self.IsFitted == False:
            raise ValueError('Model not fitted')
        else:
            mean = [0, 0]
            marginal1 = norm.ppf(u1)
            marginal2 = norm.ppf(u2)
            dist = multivariate_normal(mean=mean, cov=self.corr)
            pdf_value = dist.pdf([marginal1, marginal2]) /( norm.pdf(marginal1) * norm.pdf(marginal2))
            return pdf_value

    def sampleCopula(self, n, correlation = 0.0):
        # Function to sample data from copula
        if self.IsFitted == True:
            corr = self.corr
        else:
            # Availability to generate data with different correlation
            corr = [[1, correlation], [correlation, 1]]

        A = np.linalg.cholesky(corr)
        Z = np.random.standard_normal((n, 2))
        X = (A @ Z.T).T
        U = norm.cdf(X)
        return U



class StudentsCopula():
    def __init__(self):
        self.Name = 'Students Copula'
        self.corr = None
        self.degreeFreedom = None
        self.data = None
        self.transformedData = None
        self.IsFitted = False
        pass

    def fitModel(self, data, correlationMeasure='pearson'):
        # Function to fit copula model     
        print('Fitting Students Copula model...')  
        self.corr = np.corrcoef(data,rowvar=False)
        
        self.data = data
        self.transformedData = norm.cdf(data) ## transform data to uniform marginals from normal marginals
        self.degreeFreedom = self._optimizeDegreesFreedom(self.transformedData)[0] ## gives the optimal degrees of freedom
        self.IsFitted = True
        print('Estimated DF: ', self.degreeFreedom, ' Estimated correlation coefficient: ', self.corr[0,1])
        pass

    def _optimizeDegreesFreedom(self, transformedData):
        # Function to optimize the log likelihood of the copula
        print('Optimizing degrees of freedom...')
        epsilon = 1e-10
        bounds = [(1 , None)] # degrees of freedom must be positive

        x0 = [2] ## initial guess
        objectiveFixData = partial(self._logLikelihood, transformedData = transformedData)
        result = minimize(objectiveFixData, x0, method='L-BFGS-B', bounds=bounds,  )
        if result.x == epsilon:
            print('Optimization failed, boundary value encountered')
        return result.x

    def _logLikelihood(self,  df, transformedData):
        likelihoodvals = self._studentsCopulaPDF(transformedData=transformedData, df = df)
        loglikelihood = np.sum(np.log(likelihoodvals))
        return -loglikelihood    

    def _studentsCopulaPDF(self, transformedData, df):
        mean = np.zeros(2)  
        marginal1 = t.ppf(transformedData[:, 0], df)
        marginal2 = t.ppf(transformedData[:, 1], df)
        marginalLikelihood1 = t.pdf(marginal1, df)
        marginalLikelihood2 = t.pdf(marginal2, df)
        marginals = np.column_stack((marginal1, marginal2))
        dist = multivariate_t(loc=mean, shape=self.corr, df=df)
        pdf_values = dist.pdf(marginals) / (marginalLikelihood1  * marginalLikelihood2)   
        return pdf_values  

    def evalPDF(self, u1,u2):
        # Function to evaluate PDF of copula
        if self.IsFitted == False:
            raise ValueError('Model not fitted')
        else:
            mean = np.zeros(2)  
            marginal1 = t.ppf(u1, self.degreeFreedom)
            marginal2 = t.ppf(u2, self.degreeFreedom)
            marginalLikelihood1 = t.pdf(marginal1, self.degreeFreedom)
            marginalLikelihood2 = t.pdf(marginal2, self.degreeFreedom)
            marginals = np.column_stack((marginal1, marginal2))
            dist = multivariate_t(loc=mean, shape=self.corr, df=self.degreeFreedom)
            CopulaFunction = dist.pdf(marginals) / (marginalLikelihood1  * marginalLikelihood2)   
            return CopulaFunction

    def evalCDF(self, u1,u2):
        # Function to evaluate CDF of copula
        if self.IsFitted == False:
            raise ValueError('Model not fitted')
        else:
            mean = np.zeros(2)  
            marginal1 = t.ppf(u1, self.degreeFreedom)
            marginal2 = t.ppf(u2, self.degreeFreedom)
            dist = multivariate_t(loc=mean, shape=self.corr, df=self.degreeFreedom)
            CopulaFunction = dist.cdf([marginal1, marginal2])
            return CopulaFunction
            
    def sampleCopula(self, n, correlation = 0.0, df = 5):
        # Function to sample data from copula also possible to sample from copula with arbitrary correlation and degrees of freedom
        mean = np.zeros(2)
        if self.IsFitted == True:
            sample = multivariate_t.rvs(mean, self.corr, df, size=n)
            transformedSample = t.cdf(sample, self.degreeFreedom)
        else:
            # Availability to generate data with different correlation and degrees of freedom
            corr =[[1, correlation], [correlation, 1]]
            sample = multivariate_t.rvs(mean, corr, df, size=n)
            transformedSample = t.cdf(sample, df)
        return transformedSample
    


class ClaytonCopula():
    def __init__(self):
        self.Name = 'Clayton Copula'
        self.theta = None
        self.data = None
        self.transformedData = None
        self.IsFitted = False
        pass

    def fitModel(self, data, initialGuess = 0.1):
        # Function to fit copula model
        print('Fitting Clayton Copula model...')   
        self.data = data
        self.transformedData = norm.cdf(data) ## transform data to uniform marginals from normal marginals
        self.theta = self._optimizeTheta(self.transformedData, initialGuess)[0] ## gives the optimal theta
        self.IsFitted = True
        epsilon = 1e-10
        if self.theta == epsilon:
            print('Optimization failed, boundary value encountered')
            print('Optimization can be unstable for small values of theta. Please try again with a smaller initial guess')
            self.IsFitted = False
            self.theta = None
        else:
            print('Estimated theta: ', self.theta)
        pass

    def _optimizeTheta(self, transformedData, initialGuess):
        # Function to optimize the log likelihood of the copula
        print('Optimizing theta...')
        epsilon = 1e-10
        bounds = [(0 + epsilon, None)] # theta 
        x0 = [initialGuess] ## Seems to be good to start small rather than large
        objectiveFixData = partial(self._logLikelihood, transformedData = transformedData)
        result = minimize(objectiveFixData, x0, method='L-BFGS-B', bounds=bounds,  )
        return result.x

    def _logLikelihood(self, theta, transformedData):
        likelihoodvals = self._claytonCopulaPDF(transformedData, theta)
        loglikelihood = np.sum(np.log(likelihoodvals))
        return -loglikelihood
    
    def _claytonCopulaPDF(self, transformedData, theta):
        u1,u2 = transformedData[:,0], transformedData[:,1]
        pdf_values = (u1**(-theta) + u2**(-theta) - 1)**(-2 - 1/theta) * u1**(-theta - 1) * u2**(-theta - 1)*(theta + 1)
        return pdf_values

    def evalPDF(self, u1,u2, theta = 1):
        # Function to evaluate PDF of copula
        if self.IsFitted:
            theta = self.theta
        pdf_values = (u1**(-theta) + u2**(-theta) - 1)**(-2 - 1/theta) * u1**(-theta - 1) * u2**(-theta - 1)*(theta + 1)
        return pdf_values

    def evalCDF(self, u1,u2, theta = 1):
        # Function to evaluate CDF of copula
        if self.IsFitted:
            theta = self.theta
        CopulaFunction = (u1**(-theta) + u2**(-theta) - 1)**(-1/theta)

        # else:
        #     CopulaFunction = (u1**(-self.theta) + u2**(-self.theta) - 1)**(-1/self.theta)
        return CopulaFunction
            
    # def sampleCopulaOLD(self, n, theta = 1):
    #     # Function to sample data from copula 
    #     sampledData = np.zeros((n, 2))

    #     if self.IsFitted == True:
    #         theta = self.theta
    #     # Generate nx2 uniform random numbers
    #     U = np.random.uniform(0, 1, (n, 2))
    #     # Fix first variable 
    #     sampledData[:,0] = U[:,0]
    #     # # Find maximum value of the copula function when first variable is fixed
    #     # m = self.evalCDF(u1 =  U[:,0], u2 = np.ones(n), theta = theta)
    #     # solve for the second variable
    #     sampledData[:,1] = ((U[:,0]**(-theta) -1 )* U[:,1]**(-1/theta) +1 )**(-1/theta)   #((m*U[:,1])**(-theta) - U[:,0]**(-theta) + 1  )**(-1/theta)
    #     return sampledData


    def sampleCopula(self, n, theta = 1):
        if self.IsFitted:
            theta = self.theta
        # u1 = np.random.uniform(size=n)
        # u2 = np.random.uniform(size=n)
        u = np.random.uniform(0, 1, (n, 2))
        U1_star = u[:,0]
        S2 = U1_star ** -theta - 1
        U2_star = ((1 + S2) * u[:,1] ** (-theta / (1 + theta)) - S2) ** (-1 / theta)
        # merge the two samples
        U = np.column_stack((U1_star, U2_star))
        return U










