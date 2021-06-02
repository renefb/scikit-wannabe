import numpy as np


class LinearRegression():
    
    # properties set in self.fit() method:
    # self.__reg_coef
    # self.__r2
    
    def fit(self, X, y):
        Xcopy = X.copy() # saving a X copy for calling score method at the end 
        X = np.array(X)  # assert X is numpy.array
        y = np.array(y)  # assert y is numpy.array
        n_rows = y.shape[0]
        # if X has not the correct shape, transpose it
        if X.shape[0] != n_rows:
            X = X.T
        # add ones to calculate intercept
        X = np.column_stack((np.ones(n_rows), X))
        # reg_coefs = (X.T * X)^(-1) * X.T * y
        XT_dot_X = np.dot(X.T, X)
        XT_dot_y = np.dot(X.T, y)
        coefs = np.dot(np.linalg.inv(XT_dot_X), XT_dot_y)
        self.__reg_coefs = np.around(coefs, 3)
        r2 = self.score(Xcopy, y)
        self.__r2 = round(r2, 3)
              
    
    def predict(self, X):
        # if X has not the correct shape, transpose it
        if (X.shape[1] + 1) != self.__reg_coefs.shape[0]:
            X = X.T
        # add ones to calculate intercept
        X = np.column_stack((np.ones(X.shape[0]), X))
        # return prediction
        return np.dot(X, self.__reg_coefs)
    
    
    def score(self, X, y):
        pred = self.predict(X)
        u = ((y - pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return round((1 - u/v), 3)
    
    
    def summary(self):
        print('       Results:')
        print('========================')
        print('Score:\t\t', self.__r2)
        print('------------------------')
        print('                 Coef')
        print('Intercept\t', self.intercept_)
        for i in range(len(self.coef_)):
            print(f'       w{i}\t', self.coef_[i])
        print('========================')
    
    
    @property
    def intercept_(self):
        return self.__reg_coefs[0]
    
    
    @property
    def coef_(self):
        return self.__reg_coefs[1:]