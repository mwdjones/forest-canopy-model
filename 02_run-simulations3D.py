import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
import random
import scipy.stats as stats
import pymesh

from functions import *

#Open file for writing output
f = open("forestModel-output.txt", "w")

#'''PARAMETERS'''
niters = 10

f.write('Loading data')

veg_data = pd.read_csv('./data/compiled_data_formodel.csv')

#random.seed(12345)

f.write('Data opened, model run starting')

for index, row in veg_data.iterrows():
    site = row.Stake_ID

    if(site == 'S200'):
        continue 
    
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

    f.write(site + '\n')
    f.write('Trees: ' + str(n) + '\n')
    f.write('Distance inputs: ' + str(distMu) + ', ' + str(distSigma) + '\n')
    f.write('DBH inputs: ' + str(dbhMu) + ', ' + str(dbhSigma) + '\n')

    pcov = []
    pover = []
    total = []
    run = []

    for i in range(0, niters):
        pcoverage, poverlap, tot = simulateSite3D(n, distMu, distSigma, dbhMu, dbhSigma, pCon)
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
                        'pCo' : Co, 
                        'distMu' : distMu, 
                        'distSigma' : distSigma, 
                        'dbhMu' :dbhMu, 
                        'dbhSigma' : dbhSigma, 
                        'iteration' : run, 
                        'pOverlap' : pover, 
                        'pCoverage' : pcov,
                        'totalArea' : total})
    
    dat.to_csv('./modeloutput/test-simulation-noDecidCanopy-3D-site-' + str(site) + '-' + str(syear) + '.csv')

f.close()



