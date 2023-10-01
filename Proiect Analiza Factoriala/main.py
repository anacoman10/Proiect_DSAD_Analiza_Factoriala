import numpy as np
import pandas as pd
import AEF as aef
import factor_analyzer as fa
import grafice as g
from functii import *
from sklearn.preprocessing import StandardScaler

tabel = pd.read_csv('winequality-red.csv', index_col=0)

tabel.dropna(inplace=True)
obsNume = tabel.index
print("Observatii: ",obsNume)
varNume = tabel.columns
matrice_numerica = tabel.values

aefModel = aef.AEF(matrice_numerica)
Xstd = aefModel.getXstd()

scalare = StandardScaler()
Xstd = scalare.fit_transform(matrice_numerica)
Xstd_df = pd.DataFrame(data=Xstd, index=obsNume, columns=varNume)
Xstd_df.to_csv('Xstd.csv')

sfericitateBartlett = fa.calculate_bartlett_sphericity(Xstd_df)  # preia o matrice standardizata ca DataFrame

if sfericitateBartlett[0] > sfericitateBartlett[1]:
    print(sfericitateBartlett)
    print('Exista cel putin un factor comun!')
else:
    print('Nu exista factori!')
    exit(-1)

# calcul indici Kaiser-Meyer-Olkin pentru fiecare variabila initiala
kmo = fa.calculate_kmo(Xstd_df)
vector = kmo[0]
print('Indicele KMO general este ',kmo[1])
# adaugare noua axa la un ndarray unidimensional
matrice = vector[:, np.newaxis]
matrice_df = pd.DataFrame(data=matrice, index=varNume, columns=['Indici KMO'])
matrice_df.to_csv('IndiciKMO.csv')
g.corelograma(matrice=matrice_df, dec=3, titlu='Indicii Kaiser-Meyer-Olkin')
g.afisare()

# testare valoare KMO global
if kmo[1] > 0.5:
    print('Exista cel putin un factor comun!')
else:
    print('Nu exista factori!')
    exit(-2)

# determinarea numarului de factori semnificativi
numarFactoriSemnificativi = len(varNume)

faModelFit = fa.FactorAnalyzer(n_factors=numarFactoriSemnificativi,rotation='varimax')
faModelFit.fit(Xstd)

# extragerea valorilor proprii din Factor Analyzer
valPropFA = faModelFit.get_eigenvalues()
g.componentePrincipale(valoriProprii=valPropFA[0], titlu='Varianta explicata de componentele principale (Factor Analyzer)')
g.afisare()
print('Valorile proprii sunt: ',valPropFA[0])
count = 0
for i in valPropFA[0] :
    if i > 1 : count+=1
numarFactoriSemnificativi = count

faModelFit = fa.FactorAnalyzer(n_factors=numarFactoriSemnificativi,rotation='varimax')
faModelFit.fit(Xstd)
loadings = faModelFit.loadings_

factoriSemnificativi = ['F'+str(k+1) for k in range(numarFactoriSemnificativi)]
loadings_df = pd.DataFrame(data=loadings, index=varNume,columns=factoriSemnificativi)
# creare corelograma factori comuni
g.corelograma(matrice=loadings_df, dec=3, titlu='Corelograma factorilor de corelatie')
g.afisare()

# crearea cercului corelatiilor
g.cerculCorelatiilor(matrice=loadings_df, titlu='Distributia variabilor observate in spatiul factorilor F1 si F2')
g.afisare()

# extragerea valorilor proprii din ACP
valPropACP = aefModel.getValProp()
g.componentePrincipale(valoriProprii=valPropACP, titlu='Varianta explicata de componentele principale (ACP)')
g.afisare()

# crearea corelograma scoruri din ACP
scoruri = aefModel.getScoruri()
scoruri = scoruri[:24,:]
scoruri_df = pd.DataFrame(data=scoruri, index=obsNume[:24], columns=('C'+str(j+1) for j in range(varNume.shape[0])))
# salvare scoruri in fisier CSV
scoruri_df.to_csv('Scoruri.csv')
g.corelograma(matrice=scoruri_df, dec=2, titlu='Corelograma scorurilor din ACP')
g.afisare()

# extragere scoruri din Factor Analyzer si creare corelograma scoruri
scoruriFA = faModelFit.transform(Xstd_df)
scoruriFA = scoruriFA[:24,:]
scoruriFA_df = pd.DataFrame(data=scoruriFA, index=obsNume[:24], columns=factoriSemnificativi)
g.corelograma(matrice=scoruriFA_df, dec=2, titlu='Corelograma scorurilor din Factor Analyzer')
g.afisare()

# creare corelograma a calitatii reprezentarii observatiilor pe axele componentelor principale din ACP
calObs = aefModel.getCalObs()
calObs = calObs[:24,:]
calObs_df = pd.DataFrame(data=calObs, index=obsNume[:24], columns=('C'+str(j+1) for j in range(varNume.shape[0])))
g.corelograma(matrice=calObs_df, titlu='Corelograma calitatii reprezentarii observatiilor pe axele componentelor principale din ACP')
g.afisare()

# calcul comunalitati
Rxf2 = loadings * loadings
comunalitati = np.cumsum(Rxf2, axis=1)  # sume cumulative pe linii
comunalitati_df = pd.DataFrame(data=comunalitati, index=varNume, columns=factoriSemnificativi)
g.corelograma(matrice=comunalitati_df, titlu='Corelograma comunalitatilor')
g.afisare()

