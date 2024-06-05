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

#Veg data
veg_data = pd.read_csv('./data/compiled_data_formodel.csv')
veg_data = veg_data[veg_data.watershed == 'S6']

#load data
#Columns: SITE,COMMON_NAME,SPECIES,DIST_FT,DIST_M,DBH_IN,DBH_CM,CC,NORTHING,EASTING
treeDataS2 = pd.read_csv('./data/S2overstory_2023_compiled.csv', 
                        dtype = {'SITE': str, 'COMMON_NAME': str, 'SPECIES': str,
                                 'DIST_FT': float, 'DIST_M': float, 'DBH_IN':float, 'DBH_CM': float, 
                                 'CC': str, 'NORTHING': float, 'EASTING':float})
treeDataS6 = pd.read_csv('./data/S6overstory_2023_compiled.csv', 
                        dtype = {'SITE': str, 'COMMON_NAME': str, 'SPECIES': str,
                                 'DIST_FT': float, 'DIST_M': float, 'DBH_IN':float, 'DBH_CM': float, 
                                 'CC': str, 'NORTHING': float, 'EASTING':float})

#concatenate
treeData = pd.concat([treeDataS2, treeDataS6]).reset_index(drop = True)

#Add conif column
treeData['Con'] = [True if sp in ('Abies balsamea', 'Picea mariana', 'Larix laricina', 'Picea glauca', 'Pinus resinosa', 'Pinus Strobus') else False for sp in treeData.SPECIES]

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

    pcov = []
    pover = []
    total = []
    run = []

    #Pull site trees
    trees = treeData[treeData.SITE == site].reset_index(drop = True)

    for i in range(0, niters):
        pcoverage, poverlap, tot = simulateTrees(trees)
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
    
    dat.to_csv('./modeloutput/test-simulation-noDecidCanopy-allometric-trees-site-' + str(site) + '-' + str(syear) + '.csv')





