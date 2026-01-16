import numpy as np
from ramses.mod import Modelo
from ramses.util import *
from scipy.stats import multivariate_normal

class MixturaGauss(Modelo):
    def __init__(self, pathMod=None, lisMod=None, n_gaussianas=3):
        self.n_gaussianas = n_gaussianas
        if pathMod != None:
            self.leeMod(pathMod)
        elif lisMod != None:
            self.unidades = leeLis(lisMod)
        else:
            raise("Hay que indicar el fichero del modelo o la lista de unidades")
    
    def leeMod(self, pathMod):
        with open(pathMod, 'rb') as fpMod:
            self.medias = np.load(fpMod, allow_pickle=True).item()
            self.covs = np.load(fpMod, allow_pickle=True).item()
            self.pesos = np.load(fpMod, allow_pickle=True).item()
            self.n_gaussianas = np.load(fpMod, allow_pickle=True).item()
        self.unidades = self.medias.keys()
        
        # Crear objetos gaussianos
        self.gaussianas = {}
        for unidad in self.unidades:
            self.gaussianas[unidad] = []
            for i in range(self.n_gaussianas[unidad]):
                self.gaussianas[unidad].append(
                    multivariate_normal(mean=self.medias[unidad][i], 
                                      cov=self.covs[unidad][i], 
                                      allow_singular=True)
                )

    def escMod(self, pathMod):
        chkPathName(pathMod)
        with open(pathMod, 'wb') as fpMod:
            np.save(fpMod, self.medias)
            np.save(fpMod, self.covs)
            np.save(fpMod, self.pesos)
            np.save(fpMod, self.n_gaussianas)

    def inicMod(self):
        self.datos_ent = {unidad: [] for unidad in self.unidades}

    def __add__(self, prm_unidad):
        prm, unidad = prm_unidad
        self.datos_ent[unidad].append(prm)
        return self
    
    def calcMod(self):
        self.medias = {}
        self.covs = {}
        self.pesos = {}
        self.gaussianas = {}
        
        for unidad in self.unidades:
            datos = np.array(self.datos_ent[unidad])
            N = len(datos)
            
            # Inicializar medias con muestras aleatorias
            indices = np.random.choice(N, size=self.n_gaussianas, replace=False)
            medias_k = datos[indices]
            
            # Covarianza diagonal
            cov_global = np.diag(np.var(datos, axis=0))
            covs_k = np.array([cov_global for _ in range(self.n_gaussianas)])
            pesos_k = np.ones(self.n_gaussianas) / self.n_gaussianas
            
            # EM Algorithm
            for iter in range(5):
                # E-step
                gamma = np.zeros((N, self.n_gaussianas))
                for k in range(self.n_gaussianas):
                    rv = multivariate_normal(mean=medias_k[k], cov=covs_k[k], allow_singular=True)
                    gamma[:, k] = pesos_k[k] * np.exp(rv.logpdf(datos))
                
                gamma_sum = gamma.sum(axis=1, keepdims=True) + 1e-10
                gamma = gamma / gamma_sum
                
                # M-step
                Nk = gamma.sum(axis=0)
                pesos_k = Nk / N
                
                for k in range(self.n_gaussianas):
                    medias_k[k] = (gamma[:, k:k+1] * datos).sum(axis=0) / (Nk[k] + 1e-10)
                    diff = datos - medias_k[k]
                    covs_k[k] = np.diag(((gamma[:, k:k+1] * diff**2).sum(axis=0) / (Nk[k] + 1e-10)))
            
            self.medias[unidad] = medias_k
            self.covs[unidad] = covs_k
            self.pesos[unidad] = pesos_k
            
            # Crear gaussianas
            self.gaussianas[unidad] = []
            for k in range(self.n_gaussianas):
                self.gaussianas[unidad].append(
                    multivariate_normal(mean=medias_k[k], cov=covs_k[k], allow_singular=True)
                )

    def __call__(self, prm):
        logpdf_max = -np.inf
        for unidad in self.unidades:
            # Log-suma de las gaussianas ponderadas
            logpdfs = []
            for k in range(self.n_gaussianas):
                logpdfs.append(np.log(self.pesos[unidad][k] + 1e-10) + 
                             self.gaussianas[unidad][k].logpdf(prm))
            
            logpdf = np.logaddexp.reduce(logpdfs)
            
            if logpdf > logpdf_max:
                logpdf_max = logpdf
                reconocida = unidad
        
        return reconocida
