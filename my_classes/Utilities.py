import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

class stockPriceGenerator():
    def __init__(self, marginalDistribution1 = norm, marginalDistribution2 = norm):
        self.M1 = marginalDistribution1
        self.M2 = marginalDistribution2
        # self.stockPrices = None
        # self.stockPriceDF = None
        pass

    def GenerateStockTrajectories(self,copulaRandomnumbers):
        Z = self._generateRandomShocks(copulaRandomNumbers=copulaRandomnumbers) ## generate random shocks
        S = self._eulerMaruyama(s_0 = np.array([100, 100]), Z = Z, mu = np.array([0.03, 0.03]), sigma = np.array([0.2, 0.3]), dt = 1/252) ## simulate stock prices
        return S

    def StockTrajectoriesAsDF(self, S, startDate = '2010-01-01'):
        df = pd.DataFrame(S, columns=['Stock1', 'Stock2'])
        df = self._attachDates(df, startDate=startDate) ## create stock price dataframe
        return df

    def _generateRandomShocks(self, copulaRandomNumbers, marginalDistribution1 = norm, marginalDistribution2=norm):
        # Function to generate random shocks
        Z = np.zeros(copulaRandomNumbers.shape)
        Z[:,0] = marginalDistribution1.ppf(copulaRandomNumbers[:,0])
        Z[:,1] = marginalDistribution2.ppf(copulaRandomNumbers[:,1])
        return Z

    def _eulerMaruyama(self, s_0, Z, mu, sigma, dt):
        # Function to simulate stock prices using Euler Maruyama
        stocks = s_0.shape[0]
        timeSteps = Z.shape[0]
        S = s_0 * np.ones((timeSteps+1, stocks))
        for t in range(1, timeSteps+1):
            S[t, :] =  (S[t-1, :] + S[t-1, :]* mu * dt + S[t-1, :]*sigma * np.sqrt(dt) * Z[t-1, :])
        return S
    
    def _attachDates(self, stockPriceDF, startDate = '2010-01-01'):
        # Function to create stock price DataFrame
        start_date = pd.Timestamp(startDate)
        dates = []
        n = len(stockPriceDF)
        try:
            # Repeat until you get enough dates
            while len(dates) < n:
                # Add 5 weekdays
                weekdays = pd.date_range(start=start_date, periods=5, freq='B')
                dates.extend(weekdays)
                # Add 2 weekend days
                weekends = pd.date_range(start=weekdays[-1] + pd.Timedelta(days=1), periods=2, freq='D')
                dates.extend(weekends)
                # Move start_date forward
                start_date = weekends[-1] + pd.Timedelta(days=1)

            # Trim to match your data length
            dates = dates[:n]

            # Create the dataframe
            #stockPriceDF = pd.DataFrame(S, columns=['Stock1', 'Stock2'])

            ## Append dates to the dataframe
            stockPriceDF['Time'] = dates
            return stockPriceDF

        except (OverflowError, pd.errors.OutOfBoundsDatetime) as e:
            print(f"Date range exceeded pandas limits: {e}")
            return None

        # # Repeat until you get enough dates
        # while len(dates) < len(S):
        #     # Add 5 weekdays
        #     weekdays = pd.date_range(start=start_date, periods=5, freq='B')
        #     dates.extend(weekdays)
        #     # Add 2 weekend days
        #     weekends = pd.date_range(start=weekdays[-1] + pd.Timedelta(days=1), periods=2, freq='D')
        #     dates.extend(weekends)
        #     # Move start_date forward
        #     start_date = weekends[-1] + pd.Timedelta(days=1)

        # # Trim to match your data length
        # dates = dates[:len(S)]

        # # Create the dataframe
        # self.stockPriceDF = pd.DataFrame(S, columns=['Stock1', 'Stock2'])
        # self.stockPriceDF['Time'] = dates
        # return self.stockPriceDF

    def SplitTimeseriesArray(self, S, proportion = 0.6):
        if proportion > 1 or proportion < 0:
            raise ValueError('Proportion must be between 0 and 1')
        n = S.shape[0] 
        splitIndex = int(n * proportion)
        S_fit = S[:splitIndex, :]
        S_test = S[splitIndex:, :]
        return S_fit, S_test

    def SplitTimeseriesDF(self, DF, proportion = 0.6):
        if proportion > 1 or proportion < 0:
            raise ValueError('Proportion must be between 0 and 1')
        n = DF.shape[0]
        splitIndex = int(n * proportion)
        DF_fit = DF.iloc[:splitIndex, :]
        DF_test = DF.iloc[splitIndex-1:, :]
        return DF_fit, DF_test

    def CalculateLogReturns(self, S):
        # Function to calculate log returns
        logReturns = np.log(S[1:,:] / S[:-1,:])
        return logReturns

    def CalculateSimpleReturns(self, S):
        # Function to calculate simple returns
        simpleReturns = ((S[1:,:] - S[:-1,:])/ S[:-1,:])
        return simpleReturns



class PortfolioData():
    def __init__(self, dataDF, Name, proportion = 0.5):
        self.DataGenerator = stockPriceGenerator()
        self.Name = Name
        self.PriceDF = dataDF
        self.FittingDF , self.TestingDF = self.DataGenerator.SplitTimeseriesDF(dataDF, proportion=proportion)
        self.FittingArray, self.TestingArray = self.FittingDF.iloc[:,0:-1].to_numpy(), self.TestingDF.iloc[:,0:-1].to_numpy()
        self.FittingReturns = self.DataGenerator.CalculateLogReturns(self.FittingArray)
        self.TestingReturns = self.DataGenerator.CalculateLogReturns(self.TestingArray)

        self.estimatedStd = np.std(self.FittingReturns, axis=0)
        self.FittingNormalizedReturns = self.FittingReturns / self.estimatedStd
        
        self.FittingSampleDict = {}
        pass

    def fitAndSampleCopulas(self, CopulaList, number = 10000):

        for copula in CopulaList:
            copula.fitModel(self.FittingNormalizedReturns)
            SampledReturns = norm.ppf(copula.sampleCopula(number)) * self.estimatedStd
            self.FittingSampleDict[copula.Name] = SampledReturns
            print('----------------------------------------')
        pass

    def PlotSampledTestComparison(self):
        CopulaPlot = plotCopulaData()
        for copulaName, SampledReturns in self.FittingSampleDict.items():
            CopulaPlot.plotSampleTestComparison(SampledReturns, self.TestingReturns, SampledType = copulaName, TestingType = self.Name)
            #print('Testing returns: ', self.TestingReturns.shape[0], ' Sampled returns: ', SampledReturns.shape[0])
        pass


class CodeRunner():
    def __init__(self, portfolioSpecificationDict, copulaList, n = 10000, proportion = 0.5):
        self.PortfolioSpecifications = portfolioSpecificationDict
        self.CopulaList = copulaList
        self.proportion = proportion
        self.n = n
        self.Portfolios = {}
        self.StockGenerator = stockPriceGenerator()
        pass

    def createPortfolios(self):
        # Function to sample portfolios
        for key, value in self.PortfolioSpecifications.items():
            copula = value['Copula']
            copulaName = copula.Name
            if copulaName == 'Gaussian Copula':
                U = copula.sampleCopula(self.n, correlation=value['correlation'])
            elif copulaName == 'Students Copula':
                U = copula.sampleCopula(self.n, correlation=value['correlation'], df=value['df'])
            elif copulaName == 'Clayton Copula':
                U = copula.sampleCopula(self.n, theta=value['theta'])
            
            S = self.StockGenerator.GenerateStockTrajectories(U)
            df = self.StockGenerator.StockTrajectoriesAsDF(S, startDate='2010-01-01')
            self.Portfolios[key] = PortfolioData(df, key, proportion=self.proportion)    
        pass

    def fitCopulas(self):
        for key, portfolio in self.Portfolios.items():
            print('##############################################')
            print('Portfolio:', key)
            print('##############################################')
            print('----------------------------------------------')
            portfolio.fitAndSampleCopulas(self.CopulaList, number=int(np.floor(self.n/2)))

    def displayResults(self):
        for key, portfolio in self.Portfolios.items():
            print('----------------------------------------------')
            print('Portfolio:', key)
            print('----------------------------------------------')
            portfolio.PlotSampledTestComparison()
            
    def insertRealData(self, dataDF):
        # Function to insert real data portfolio into code runner dictionary
        pass


class plotCopulaData():
    def __init__(self):
        pass

    def plotProbabilitySpaceData(self, data, title):
            df_returnSpace = pd.DataFrame({
                 "U1": data[:,0].flatten(),  
                 "U2": data[:,1].flatten()})
            sns.jointplot(
                data=df_returnSpace, x="U1", y="U2", kind="scatter",
                marginal_kws=dict(bins=30, fill=True),
                joint_kws={"s": 10, "edgecolor": "none"} )      
            plt.suptitle(f"Probability space data for {title} ", y=1.02)
            plt.show()

    def plotReturnSpaceData(self, data, title):
            df_returnSpace = pd.DataFrame({
                 "X1": data[:,0].flatten(),  
                 "X2": data[:,1].flatten()})
            sns.jointplot(
                data=df_returnSpace, x="X1", y="X2", kind="scatter",
                marginal_kws=dict(bins=30, fill=True),
                joint_kws={"s": 10, "edgecolor": "none"} )      
            plt.suptitle(f"Return space data for {title} ", y=1.02)
            plt.show()          

    def plotCopulaContour(self, Copula, title):
        u1 = np.linspace(0.01, 0.99, 100)
        u2 = np.linspace(0.01, 0.99, 100)

        U1, U2 = np.meshgrid(u1, u2)
        Z = np.zeros_like(U1)
        for i in range(100):
            for j in range(100):
                Z[i,j] = Copula.evalCDF(U1[i,j], U2[i,j])

        fig, ax = plt.subplots(figsize=(5, 5))
        contour = ax.contourf(U1, U2, Z, cmap='viridis', levels=50)
        ax.set_aspect('equal')
        cbar = plt.colorbar(contour, ax=ax, shrink=0.5)  
        ax.set_title(f'Contour plot of Copula CDF for {title}')
        ax.set_xlabel('U1')
        ax.set_ylabel('U2')
        fig.tight_layout()
        plt.show()
        pass
    
    def plotCopulaSurface(self, Copula, title):
        u1 = np.linspace(0.01, 0.99, 100)
        u2 = np.linspace(0.01, 0.99, 100)

        U1, U2 = np.meshgrid(u1, u2)
        Z = np.zeros_like(U1)
        for i in range(100):
            for j in range(100):
                Z[i,j] = Copula.evalCDF(U1[i,j], U2[i,j])

        fig = plt.figure(figsize=(5, 5))
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.plot_surface(U1, U2, Z, cmap='viridis')
        ax1.set_title(f'Copula CDF for: {title}')
        ax1.set_xlabel('U1')
        ax1.set_ylabel('U2')
        ax1.set_zlabel('C(U1, U2)')
        ax1.view_init(elev=15, azim=280, roll=0)
        plt.show()
        pass

    def plotCopulaPDFContour(self, Copula, title):
        u1 = np.linspace(0.01, 0.99, 100)
        u2 = np.linspace(0.01, 0.99, 100)

        U1, U2 = np.meshgrid(u1, u2)
        Z = np.zeros_like(U1)
        for i in range(100):
            for j in range(100):
                Z[i,j] = Copula.evalPDF(U1[i,j], U2[i,j])

        fig, ax = plt.subplots(figsize=(5, 5))
        contour = ax.contourf(U1, U2, Z, cmap='viridis', levels=50)
        ax.set_aspect('equal')
        cbar = plt.colorbar(contour, ax=ax, shrink=0.5)  
        ax.set_title(f'Contour plot of Copula PDF for: {title}')
        ax.set_xlabel('U1')
        ax.set_ylabel('U2')
        fig.tight_layout()
        plt.show()
        pass
    
    def plotCopulaPDFSurface(self, Copula, title):
        u1 = np.linspace(0.01, 0.99, 100)
        u2 = np.linspace(0.01, 0.99, 100)

        U1, U2 = np.meshgrid(u1, u2)
        Z = np.zeros_like(U1)
        for i in range(100):
            for j in range(100):
                Z[i,j] = Copula.evalPDF(U1[i,j], U2[i,j])

        fig = plt.figure(figsize=(5, 5))
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.plot_surface(U1, U2, Z, cmap='viridis')
        ax1.set_title(f'Copula PDF for: {title}')
        ax1.set_xlabel('U1')
        ax1.set_ylabel('U2')
        ax1.set_zlabel('C(U1, U2)')
        ax1.view_init(elev=15, azim=280, roll=0)
        plt.show()
        pass

    def plotStockTrajectory(self, PriceDF, title = 'Stock Price Trajectories'):
        fig = plt.figure(figsize=(8, 4))
        PriceDF.plot(x='Time', title = title ,ylabel='Price', xlabel='Time');
        pass

    def plotStockPriceSections(self, FittingDF, TestingDF):
        fig, ax = plt.subplots(figsize=(10, 5))
        # Plot Stock1 from fit and test
        FittingDF.plot(x='Time', y='Stock1', ax=ax, color='C0', label='Stock1 - Fitting Data')
        TestingDF.plot(x='Time', y='Stock1', ax=ax, color='#104e8c', label='Stock1 - Testing Data')

        # Plot Stock2 from fit and test
        FittingDF.plot(x='Time', y='Stock2', ax=ax, color='orange', label='Stock2 - Fitting Data')
        TestingDF.plot(x='Time', y='Stock2', ax=ax, color='C1', label='Stock2 - Testing Data')

        transition_time = FittingDF['Time'].iloc[-1]
        ax.axvline(x=transition_time, color='black', linestyle='--', linewidth=2, label='Fit/Test divider')
        ax.set_title('Stock Price Trajectory division')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=9, ncol=2)
        plt.show()
        pass

    def plotSampleTestComparison(self, SampledReturnsArray, TestingReturnsArray, title = 'Return coparison', SampledType = 'Unnamed', TestingType = 'Unnamed'):
        df_returnSpace_fitting = pd.DataFrame({
                "X1": SampledReturnsArray[:, 0].flatten(),  
                "X2": SampledReturnsArray[:, 1].flatten()})
        df_returnSpace_testing = pd.DataFrame({
                "X1": TestingReturnsArray[:, 0].flatten(),  
                "X2": TestingReturnsArray[:, 1].flatten()})

        g = sns.jointplot(
            data=df_returnSpace_fitting, x="X1", y="X2", kind="scatter",
            marginal_kws=dict(bins=30, fill=True, alpha=0.75, stat="density"),  # Make it semi-transparent
            joint_kws={"s": 10, "edgecolor": "none"},
            color="blue",  # Color for fitting data
            label="Sampled Returns"
        )
        sns.scatterplot(
            data=df_returnSpace_testing, x="X1", y="X2",
            color="red", s=10, edgecolor="none", ax=g.ax_joint, label="Testing Returns"
        )
        sns.histplot(df_returnSpace_testing["X1"], bins=30, color="red", alpha=0.6, ax=g.ax_marg_x, stat="density")
        sns.histplot(y=df_returnSpace_testing["X2"], bins=30, color="red", alpha=0.6, ax=g.ax_marg_y, stat="density")   
        g.ax_joint.legend(
        handles=g.ax_joint.collections + g.ax_joint.get_lines(), 
        labels=['Sampling Returns', 'Testing Returns'],
        loc="upper right")
        plt.suptitle(f"Return Space comparison for {TestingType} fitted by {SampledType}", y=1.02)
        plt.show()
        pass
        
    def plotFittingTestComparison(self, SampledReturnsArray, TestingReturnsArray, title = 'Return coparison'):
        df_returnSpace_fitting = pd.DataFrame({
                "X1": SampledReturnsArray[:, 0].flatten(),  
                "X2": SampledReturnsArray[:, 1].flatten()})
        df_returnSpace_testing = pd.DataFrame({
                "X1": TestingReturnsArray[:, 0].flatten(),  
                "X2": TestingReturnsArray[:, 1].flatten()})

        g = sns.jointplot(
            data=df_returnSpace_fitting, x="X1", y="X2", kind="scatter",
            marginal_kws=dict(bins=30, fill=True, alpha=0.75, stat="density"),  # Make it semi-transparent
            joint_kws={"s": 10, "edgecolor": "none"},
            color="blue",  # Color for fitting data
            label="Sampled Returns"
        )
        sns.scatterplot(
            data=df_returnSpace_testing, x="X1", y="X2",
            color="red", s=10, edgecolor="none", ax=g.ax_joint, label="Testing Returns"
        )
        sns.histplot(df_returnSpace_testing["X1"], bins=30, color="red", alpha=0.6, ax=g.ax_marg_x, stat="density")
        sns.histplot(y=df_returnSpace_testing["X2"], bins=30, color="red", alpha=0.6, ax=g.ax_marg_y, stat="density")   
        g.ax_joint.legend(
        handles=g.ax_joint.collections + g.ax_joint.get_lines(), 
        labels=['Fitting Returns', 'Testing Returns'],
        loc="upper right")
        
        plt.suptitle(f"Return Space Data for {title}", y=1.02)
        plt.show()
        pass


