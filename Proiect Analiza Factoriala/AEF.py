'''
clasa care incapsuileaza o implementare de AEF
'''
import numpy as np
import ACP as acp
import scipy.stats as sts


class AEF:

    def __init__(self, matrice):  # parametru asteptat este un numpy.ndarray
        self.X = matrice

        # intantiere model ACP
        acpModel = acp.ACP(self.X)
        self.Xstd = acpModel.getXstd()
        self.Corr = acpModel.getCorr()
        self.alpha = acpModel.getValProp()
        self.Scoruri = acpModel.getScoruri()
        self.CalObs = acpModel.getCalObs()


    def getXstd(self):
        return self.Xstd

    def getScoruri(self):
        return self.Scoruri

    def getValProp(self):
        return self.alpha

    def getCalObs(self):
        return self.CalObs

    def calculTestBartlett(self, loadings, epsilon):
        n = self.X.shape[0]
        m, q = np.shape(loadings)
        print('n, m, q: ', n, m, q)
        V = self.Corr
        psi = np.diag(epsilon)
        # print(psi)
        Vestim = loadings @ np.transpose(loadings) + psi
        # estimarea matricei identitate
        I = np.linalg.inv(Vestim) @ V
        detI = np.linalg.det(I)
        if detI > 0:
            urmaI = np.trace(I)
            chi2Calc = (n - 1 - (2*m + 4*q -5)/6) * (urmaI - np.log(detI) - m)
            r = ((m - q)**2 - m - q) / 2
            chi2Tab = 1 - sts.chi2.cdf(chi2Calc, r)
        else:
            chi2Calc, chi2Tab = np.nan, np.nan

        return chi2Calc, chi2Tab


