import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shapely as shp
import geopandas as gpd
import math
import random
import scipy.stats as stats
import itertools

from functions import *

'''PARAMETERS'''
niters = 100

random.seed(12345)

#Generate sensitivity data
distsd = 3*stats.uniform.rvs(size = 10)
dbhs = 40*stats.uniform.rvs(size = 10)

combs = list(itertools.product(distsd, dbhs))
distm = []
dbhscm = []
for i in range(0, len(combs)):
    distm.append(combs[i][0])
    dbhscm.append(combs[i][1])

synthData = pd.DataFrame({'DISTSD_M' : distm, 
                          'DBH_CM' : dbhscm})

for index, row in synthData.iterrows():
    site = index
    distMu = 4
    distSigma = row.DISTSD_M #set for now to test sensitivity of mean, can change later to test standard deviation sensitivity
    dbhMu = row.DBH_CM
    dbhSigma = 1
    pCon = 1 #change to test different values
    n = 10 #change to test different values, but should be uniform across sites

    pcov = []
    pover = []
    total = []
    run = []

    for i in range(0, niters):
        pcoverage, poverlap, tot = simulateSite(n, distMu, distSigma, dbhMu, dbhSigma, pCon)
        pcov.append(pcoverage)
        pover.append(poverlap)
        total.append(tot)
        run.append(i)

    #Save to dataframe and csv
    dat = pd.DataFrame({'pCon' : pCon,
                        'distMu' : distMu, 
                        'distSigma' : distSigma, 
                        'dbhMu' :dbhMu, 
                        'dbhSigma' : dbhSigma, 
                        'iteration' : run, 
                        'pOverlap' : pover, 
                        'pCoverage' : pcov,
                        'totalArea' : total})
    
    dat.to_csv('./modeloutput/sensitivity-tests/sensitivitysimulation-sddist-meandbh-' + str(site) + '-' + 'pCon' + str(pCon) + '.csv')





