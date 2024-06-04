import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
#import shapely as shp
#import geopandas as gpd
import math
#from matplotlib import offsetbox
#from matplotlib.gridspec import GridSpec
import numpy as np
#import xarray as xr
#from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks
from scipy.stats import kendalltau
import scipy.stats as stats
import pymesh
import itertools

from math import floor, log10

'''Parameters'''
ELEVATION = 1384 #feet
LATITUDE = 47.5152344351
LONGITUDE = -93.4706821209
HI = ((ELEVATION - 887) / 100) * 1.0 + (LATITUDE - 39.54) * 4.0 + (-82.52 - LONGITUDE) * 1.25

'''Functions'''

#Generally useful
def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)


def jitter(values,j):
  return values + np.random.normal(0,j,values.shape)

def positive_cumsum(x):
    y = np.zeros(len(x))

    for i in range(1, len(x)-1):
        if(y[i-1] + x[i-1] < 0):
            y[i] = 0
        else:
            y[i] = y[i-1] + x[i-1]

    return y


#Model Functions
def rotate_origin_only(x, y, radians):
    """Only rotate a point around the origin (0, 0)."""
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    return xx, yy

'''
def simulateSite(n, distMu, distSigma, dbhMu, dbhSigma, pCon, plots = False):
    #Step 1: Generate site
    site = shp.Point((0, 0)).buffer(8)

    #Generate distances
    dists = np.random.normal(loc = distMu, scale = distSigma, size = n)

    #Check all are in correct range
    while ((dists > 8).any()) | ((dists < 0).any()):
        dists = np.random.normal(loc = distMu, scale = distSigma, size = n)

    #Generate DBHs (in cm)
    dbhs = np.random.normal(loc = dbhMu, scale = dbhSigma, size = n)

    #Check all are in correct range
    while(dbhs < 0).any():
        dbhs = np.random.normal(loc = dbhMu, scale = dbhSigma, size = n)

    #Generate conif crown widths 
    conifCC = (stats.gamma.rvs(2.84604046158689, loc=-1.8990921243783796, scale=14.411867120014431, size=n))/100
    while(conifCC < 0).any():
        conifCC = (stats.gamma.rvs(2.84604046158689, loc=-1.8990921243783796, scale=14.411867120014431, size=n))/100

    #Based on pCon, assign crown width for conif and decid trees
    nCon = int(n*pCon)
    crown = np.zeros(n)
    #take first nCon trees:
    if nCon == n:
        #For sites with all conifers, just take all simulated crown data
        crown = conifCC
    elif (nCon > 0) & (nCon < n):
        #CONIFEROUS - take first nCon values
        crown[0:nCon+1] = conifCC[0:nCon+1]
        #DECIDUOUS - calculate crown diameter (using allometric equation for american elm)
        crown[nCon+1:] = dbhs[nCon+1:]/100
        #crown[nCon+1:] = (1.92 + 18.30*(dbhs[nCon+1:]/100))/2
    else:
        #try just using DBH instead of overestimated canopy 
        crown = dbhs/100
        #crown = (1.92 + 18.30*(dbhs/100))/2

    #Generate locations
    trees = []
    start = site.centroid
    for i in range(0, len(dists)):
        #assume on x axis and rotate a random number of degrees
        deg = np.random.uniform(low = 0, high = 2*math.pi)
        xx, yy = rotate_origin_only(dists[i], 0, deg)

        #create point
        trees.append(shp.Point((xx, yy)).buffer(crown[i]))


    #Step 1.5: Calculate total tree canopy
    areas = [tree.area for tree in trees]
    tot = np.cumsum(areas)[-1]

    #Step 2: Calculate total plot coverage
    #Use shapely to calculate the total union and the intersection over the union
    uni = shp.ops.unary_union(trees)
    uni_clipped = gpd.GeoSeries(uni).clip(site)
    pcoverage = float(uni_clipped.area/site.area)

    #Step 3: Calculate total overlap
    #Collect intersections recursively
    temp = trees[0]
    intersects = []
    for tree in trees[1:]:
        #add intersection of unioned shape and new tree
        intersects.append(shp.intersection(tree, temp))
        #compute union
        temp = shp.union(tree, temp)

    #Merge
    intersect = shp.ops.unary_union(intersects)
    int_clipped = gpd.GeoSeries(intersect).clip(site)
    
    if(len(int_clipped.area) > 0):
        poverlap = float(int_clipped.area/site.area)
    else:
        poverlap = 0

    #Step 4: Plot
    if(plots):
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize = (5,  10), 
                                            sharex = True, 
                                            sharey = True, 
                                            layout = 'tight')

        #Plot union
        gpd.GeoSeries(site).plot(ax = ax1, edgecolor = 'silver', facecolor = 'white')
        for tree in trees:
            gpd.GeoSeries(tree).plot(ax = ax1, color = 'green', alpha = 0.2)
            gpd.GeoSeries(tree.centroid).plot(ax = ax1, color = 'green')
        gpd.GeoSeries(uni_clipped).plot(ax = ax1, edgecolor = 'black', alpha = 0.2)

        #Plot intersection
        gpd.GeoSeries(site).plot(ax = ax2, edgecolor = 'silver', facecolor = 'white')
        for tree in trees:
            gpd.GeoSeries(tree).plot(ax = ax2, color = 'green', alpha = 0.2)
            gpd.GeoSeries(tree.centroid).plot(ax = ax2, color = 'green')

        #clip and plot union
        gpd.GeoSeries(int_clipped).plot(ax = ax2, edgecolor = 'black', alpha = 0.2)

        ax1.set_xlim(-8, 8)
        ax1.set_ylim(-8, 8)
        ax1.set_title('Total Plot Coverage: ' + str(np.round(pcoverage, 2)), loc = 'left', size = 'small')
        ax2.set_xlim(-8, 8)
        ax2.set_ylim(-8, 8)
        ax2.set_title('Total Plot Overlap: ' + str(np.round(poverlap, 2)), loc = 'left', size = 'small')

    return pcoverage, poverlap, tot
'''

def createTree(dbh, dist, con, ba, hi):
    if con:
        #Simulate coniferous trees as Black Spruce
        #Calculate height
        height_ft = 4.5 + 2136.9468*math.exp(-6.2688*math.pow(dbh, -0.2161)) #feet
        height = height_ft*0.3048 #transform to m
        #Calulate crown ratio
        cr = 10*(5.540/(1 + 0.0072*ba)) + (4.200*(1 - math.exp(-0.0530*dbh)))
        #Calculate crown
        crown_ft = -0.8566 + 0.9693*dbh + 0.0573*cr   
        crown = (crown_ft*0.3048)/2
    else:
        #Simulate deciduous trees as Red Maple
        #Calculate height
        height_ft = 4.5 * math.exp(4.3379 + ((-3.8214)/(dbh + 1))) #feet
        height = height_ft*0.3048 #transform to m
        #Calulate crown ratio
        #cr = 10*(4.340/(1 + 0.0046*ba)) + (1.820*(1 - math.exp(-0.2740*dbh)))
        #Calculate crown
        #crown_ft = 2.7562 + 1.4212*dbh - 0.0143*math.pow(dbh, 2) + 0.0993*cr - 0.0276*hi 
        #crown = crown_ft*0.3048
        #Simply use BA as crown width
        crown = (dbh*0.0254)/2 #in to m then to radius
    
    #Rotate location
    #assume on x axis and rotate a random number of degrees
    deg = np.random.uniform(low = 0, high = 2*math.pi)
    xx, yy = rotate_origin_only(dist, 0, deg)

    #create point
    return pymesh.generate_cylinder((xx, yy, 0), (xx, yy, height), crown, 0, 12) 


def simulateSite3D(n, distMu, distSigma, dbhMu, dbhSigma, pCon, plots = False):
    #Step 1: Generate site
    site = pymesh.generate_cylinder((0, 0, 0), (0, 0, 25), 8, 8, 32)

    #Generate distances
    dists = np.random.normal(loc = distMu, scale = distSigma, size = n)
    #Check all are in correct range
    while ((dists > 8).any()) | ((dists < 0).any()):
        dists = np.random.normal(loc = distMu, scale = distSigma, size = n)

    #Generate DBHs (in cm)
    dbhs = np.random.normal(loc = dbhMu, scale = dbhSigma, size = n)
    #Check all are in correct range
    while(dbhs < 0).any():
        dbhs = np.random.normal(loc = dbhMu, scale = dbhSigma, size = n)
    #Convert to in
    dbhs_in = dbhs*0.3937
    
    #Calculate Basal Area for site (m2/hectare)
    BA = sum(math.pi*((dbhs/2)**2))*(1/(math.pi*((26.2)**2)))*(1/0.00000929)*(0.001)
    
    #Based on pCon, assign conifs
    nCon = int(n*pCon)
    con = np.concatenate((np.zeros(nCon), np.ones(n - nCon)))

    #Generate locations
    trees = []
    for i in range(0, len(dists)):
        trees.append(createTree(dbhs_in[i], dists[i], con[i], BA, HI))

    #Step 1.5: Calculate total tree canopy
    volumes = [tree.volume for tree in trees]
    tot = np.cumsum(volumes)[-1]

    #Step 2: Calculate total plot coverage
    #Use shapely to calculate the total union and the intersection over the union
    uni = trees[0]
    for tree in trees[1:]:
        uni = pymesh.boolean(uni, tree, 'union')
    
    uni_clipped = pymesh.boolean(uni, site, 'intersection')
    pcoverage = float(uni_clipped.volume/site.volume)

    #Step 3: Calculate total overlap
    #Collect intersections recursively
    #intersects = []
    #for i in range(len(trees)):
    #    for j in range(i):
    #        intersects.append(pymesh.boolean(trees[i], trees[j], 'intersection'))

    #merge
    #ints = intersects[0]
    #for i in range(1, len(intersects)):
    #    ints = pymesh.boolean(ints, intersects[i], 'union')
    treeComb = itertools.combinations(trees, 2)
    intersects = [pymesh.boolean(i, j, 'intersection') for (i, j) in treeComb]

    #merge
    ints = pymesh.CSGTree({"union": [{"mesh": mesh} for mesh in intersects]})    
    
    #Clip
    int_clipped = pymesh.boolean(ints.mesh, site, 'intersection')
    poverlap = float(int_clipped.volume/site.volume)

    #Step 4: Plot
    if(plots):
        fig = plt.figure(figsize = (12,10))
        ax = plt.axes(projection='3d')

        #plot trees
        for tree in trees:
            l = len(tree.vertices)
            x = [tree.vertices[i][0] for i in range(0, l)]
            y = [tree.vertices[i][1] for i in range(0, l)]
            z = [tree.vertices[i][2] for i in range(0, l)]
            ax.scatter(x,y,z, color = 'black', alpha = 0.25)
            #ax.plot_trisurf(x, y, z, color = 'black', alpha = 0.25)

        #plot intersection
        l = len(ints.vertices)
        x_int = [ints.vertices[i][0] for i in range(0, l)]
        y_int = [ints.vertices[i][1] for i in range(0, l)]
        z_int = [ints.vertices[i][2] for i in range(0, l)]
        ax.scatter(x_int, y_int, z_int)
        ax.plot_trisurf(x_int, y_int, z_int, alpha = 0.25)
        
        ax.set_ylim(-8, 8)
        ax.set_xlim(-8, 8)
    #ax.set_zlim(0, 25)

    return pcoverage, poverlap, tot

'''
#Model Assumption Validation Functions
def calculate_residuals(model, features, label):
    """
    Creates predictions on the features with the model and calculates residuals
    """
    predictions = model.predict(features)
    df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])
    
    return df_results


def normal_errors_assumption(model, features, label, p_value_thresh=0.05):
    """
    Normality: Assumes that the error terms are normally distributed. If they are not,
    nonlinear transformations of variables may solve this.
               
    This assumption being violated primarily causes issues with the confidence intervals
    """
    from statsmodels.stats.diagnostic import normal_ad
    print('Assumption 2: The error terms are normally distributed', '\n')
    
    # Calculating residuals for the Anderson-Darling test
    df_results = calculate_residuals(model, features, label)
    
    print('Using the Anderson-Darling test for normal distribution')

    # Performing the test on the residuals
    p_value = normal_ad(df_results['Residuals'])[1]
    print('p-value from the test - below 0.05 generally means non-normal:', p_value)
    
    # Reporting the normality of the residuals
    if p_value < p_value_thresh:
        print('Residuals are not normally distributed')
    else:
        print('Residuals are normally distributed')
    
    # Plotting the residuals distribution
    plt.subplots(figsize=(12, 6))
    plt.title('Distribution of Residuals')
    sns.distplot(df_results['Residuals'])
    plt.show()
    
    print()
    if p_value > p_value_thresh:
        print('Assumption satisfied')
    else:
        print('Assumption not satisfied')
        print()
        print('Confidence intervals will likely be affected')
        print('Try performing nonlinear transformations on variables')


def autocorrelation_assumption(model, features, label):
    """
    Autocorrelation: Assumes that there is no autocorrelation in the residuals. If there is
                     autocorrelation, then there is a pattern that is not explained due to
                     the current value being dependent on the previous value.
                     This may be resolved by adding a lag variable of either the dependent
                     variable or some of the predictors.
    """
    from statsmodels.stats.stattools import durbin_watson
    print('Assumption 4: No Autocorrelation', '\n')
    
    # Calculating residuals for the Durbin Watson-tests
    df_results = calculate_residuals(model, features, label)

    print('\nPerforming Durbin-Watson Test')
    print('Values of 1.5 < d < 2.5 generally show that there is no autocorrelation in the data')
    print('0 to 2< is positive autocorrelation')
    print('>2 to 4 is negative autocorrelation')
    print('-------------------------------------')
    durbinWatson = durbin_watson(df_results['Residuals'])
    print('Durbin-Watson:', durbinWatson)
    if durbinWatson < 1.5:
        print('Signs of positive autocorrelation', '\n')
        print('Assumption not satisfied')
    elif durbinWatson > 2.5:
        print('Signs of negative autocorrelation', '\n')
        print('Assumption not satisfied')
    else:
        print('Little to no autocorrelation', '\n')
        print('Assumption satisfied')

def homoscedasticity_assumption(model, features, label):
    """
    Homoscedasticity: Assumes that the errors exhibit constant variance
    """
    print('Assumption 5: Homoscedasticity of Error Terms', '\n')
    
    print('Residuals should have relative constant variance')
        
    # Calculating residuals for the plot
    df_results = calculate_residuals(model, features, label)

    # Plotting the residuals
    plt.subplots(figsize=(12, 6))
    ax = plt.subplot(111)  # To remove spines
    plt.scatter(x=df_results.index, y=df_results.Residuals, alpha=0.5)
    plt.plot(np.repeat(0, df_results.index.max()), color='darkorange', linestyle='--')
    ax.spines['right'].set_visible(False)  # Removing the right spine
    ax.spines['top'].set_visible(False)  # Removing the top spine
    plt.title('Residuals')
    plt.show()  
'''