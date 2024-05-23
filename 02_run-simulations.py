import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shapely as shp
import geopandas as gpd
import math
import random
import scipy.stats as stats

from functions import *

'''PARAMETERS'''
niters = 100

veg_data = pd.read_csv('./data/compiled_data_formodel.csv')

random.seed(12345)

for index, row in veg_data.iterrows():
    site = row.Stake_ID
    syear = row.SYear
    pCon = row.prop_Coniferous
    Co = row.Co
    distMu = row.DIST_M
    distSigma = row.DIST_M_SD
    dbhMu = row.DBH_CM
    dbhSigma = row.DBH_CM_SD
    siteSD = row.maxDepth
    siteLAI = row['OLS.Prediction.Ring.5']
    n = row.n

    print(site + '\n')
    print('Trees: ' + str(n) + '\n')
    print('Distance inputs: ' + str(distMu) + ', ' + str(distSigma) + '\n')
    print('DBH inputs: ' + str(dbhMu) + ', ' + str(dbhSigma) + '\n')

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
    dat = pd.DataFrame({'SITE': site, 
                        'SYEAR' : syear,
                        'maxSnowDepth' : siteSD, 
                        'LAI' : siteLAI,
                        'pCon' : pCon,
                        'pCo' : Co/n, 
                        'distMu' : distMu, 
                        'distSigma' : distSigma, 
                        'dbhMu' :dbhMu, 
                        'dbhSigma' : dbhSigma, 
                        'iteration' : run, 
                        'pOverlap' : pover, 
                        'pCoverage' : pcov,
                        'totalArea' : total})
    
    dat.to_csv('./modeloutput/test-simulation-noDecidCanopy-site-' + str(site) + '-' + str(syear) + '.csv')





