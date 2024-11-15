    
###############################################################################
### Import Data ###############################################################
###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.close('all')

from statsmodels.graphics.gofplots import qqplot # for QQ Plot
from scipy.stats import anderson # for Anderson Darling

df = pd.read_excel("G:/SYNC/School/VT/CLASSES/Panel Data (6614)/Final Project/LIBOR/Libor-data-2022.xlsx")

# To handle unit root
df['r1Lag1'] = df['r1'].shift(1)
df['r2Lag1'] = df['r2'].shift(1)
df['r3Lag1'] = df['r3'].shift(1)
df['r4Lag1'] = df['r4'].shift(1)

### Format Variables ###

# To get the variables in DIFF
df['diffr1'] = df['r1']-df['r1'].shift(1)
df['diffr2'] = df['r2']-df['r2'].shift(1)
df['diffr3'] = df['r3']-df['r3'].shift(1)
df['diffr4'] = df['r4']-df['r4'].shift(1)
df['diffr5'] = df['r5']-df['r5'].shift(1)
df['diffr7'] = df['r7']-df['r7'].shift(1)
df['diffr10'] = df['r10']-df['r10'].shift(1)
df['diffr30'] = df['r30']-df['r30'].shift(1)

###############################################################################
### Make Variables for MS Testing #############################################
###############################################################################


###                     Variables to test dependence [4]                    ###
###############################################################################

df['diffr1Lag1'] = df['diffr1'].shift(1)
df['diffr1Lag2'] = df['diffr1'].shift(2)
df['diffr1Lag3'] = df['diffr1'].shift(3)
df['diffr1Lag4'] = df['diffr1'].shift(4)

df['diffr2Lag1'] = df['diffr2'].shift(1)
df['diffr2Lag2'] = df['diffr2'].shift(2)
df['diffr2Lag3'] = df['diffr2'].shift(3)
df['diffr2Lag4'] = df['diffr2'].shift(4)

df['diffr3Lag1'] = df['diffr3'].shift(1)
df['diffr3Lag2'] = df['diffr3'].shift(2)
df['diffr3Lag3'] = df['diffr3'].shift(3)
df['diffr3Lag4'] = df['diffr3'].shift(4)

df['diffr4Lag1'] = df['diffr4'].shift(1)
df['diffr4Lag2'] = df['diffr4'].shift(2)
df['diffr4Lag3'] = df['diffr4'].shift(3)
df['diffr4Lag4'] = df['diffr4'].shift(4)



df['diffr1Lag1SQ'] = df['diffr1Lag1']**2
df['diffr2Lag1SQ'] = df['diffr2Lag1']**2
df['diffr3Lag1SQ'] = df['diffr3Lag1']**2
df['diffr4Lag1SQ'] = df['diffr4Lag1']**2



###                  Variables to test time invariance [5]                  ###
###############################################################################

### Generate t0 ###

n = len(df.index)

df['trend'] = 0

for i in df.index:
    df['trend'][i] = (2*df['t'][i]-n-1)/(n-1)

df['trend2'] = df['trend']**2
df['trend3'] = df['trend']**3
df['trend4'] = df['trend']**4

### Generate Seasonal Dummies ###
    # n/a

"""    
###                                 t-Plots                                 ###
###############################################################################

fig, [[ax1,ax2],[ax3,ax4],[ax5,ax6],[ax7,ax8]] = plt.subplots(4,2,constrained_layout=True)
fig.suptitle('All Variables over Time')
###r1###
ax1.plot(df['t'], df['diffr1'], linewidth=1)
ax1.set(ylabel='r1')

###r2###
ax2.plot(df['t'], df['diffr2'], linewidth=1)
ax2.set(ylabel="r2")

###r3###
ax3.plot(df['t'], df['diffr3'], linewidth=1)
ax3.set(ylabel='r3')

###r4###
ax4.plot(df['t'], df['diffr4'], linewidth=1)
ax4.set(ylabel='r4')

###r5###
ax5.plot(df['t'], df['diffr5'], linewidth=1)
ax5.set(ylabel='r5')

###r7###
ax6.plot(df['t'], df['diffr7'], linewidth=1)
ax6.set(ylabel='r7')

###r10###
ax7.plot(df['t'], df['diffr10'], linewidth=1)
ax7.set(ylabel='r10')

###r30###
ax8.plot(df['t'], df['diffr30'], linewidth=1)
ax8.set(ylabel='r30')

#fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.025),
#          ncol=5, facecolor="white", fancybox=True, shadow=True)
fig.tight_layout()
"""

################ Generate Residuals for Auxiliary Regressions ################# 
import statsmodels.api as sm

# Starts with base model, each new line adds variables based on M-S testing results
    # for example: if trend found, then add trend into the regression 
        # (which updates the residuals you MS test with!)

### Base EQ
X = pd.DataFrame(index=range(len(df)),columns=range(0))

#X['constant'] = 1
X['diffr1'] = df['diffr1']
X['diffr2'] = df['diffr2']
X['diffr3'] = df['diffr3']
X['diffr4'] = df['diffr4']
#X['r1'] = df['r1']
#X['r2'] = df['r2']
#X['r3'] = df['r3']
#X['r4'] = df['r4']

# To import into R, to run StVAR
X=X.tail(-1)
X.to_excel('G:/SYNC/School/VT/CLASSES/Panel Data (6614)/Final Project/LIBOR/panelDataSet.xlsx', index=False) 

# To import into R, to include unit roots in StVAR
panelrLags = pd.DataFrame(index=range(len(df)),columns=range(0))
panelrLags['r1Lag1'] = df['r1Lag1']
panelrLags['r2Lag1'] = df['r2Lag1']
panelrLags['r3Lag1'] = df['r3Lag1']
panelrLags['r4Lag1'] = df['r4Lag1']
panelrLags=panelrLags.tail(-1)
panelrLags.to_excel('G:/SYNC/School/VT/CLASSES/Panel Data (6614)/Final Project/LIBOR/panelrLags.xlsx', index=False) 

import sys
user=input("Are you ready to import the Student's t residuals? (Y/N): ").lower()
if(user!="y"):
    sys.exit()
"""The purpose of the block above are to ensure the user has generated the 
Student's T residuals via the R file (since only R has the corresponding package)"""



###############################################################################
###############################################################################
###############################################################################
### NOW PUT EXPORTED FILE INTO R > EXPORT R RESULTS & IMPORT HERE
###############################################################################
###############################################################################
###############################################################################



resid_r1_St = pd.read_csv("G:/SYNC/School/VT/CLASSES/Panel Data (6614)/Final Project/LIBOR/Resid_r1 (PD).txt")
resid_r2_St = pd.read_csv("G:/SYNC/School/VT/CLASSES/Panel Data (6614)/Final Project/LIBOR/Resid_r2 (PD).txt")
resid_r3_St = pd.read_csv("G:/SYNC/School/VT/CLASSES/Panel Data (6614)/Final Project/LIBOR/Resid_r3 (PD).txt")
resid_r4_St = pd.read_csv("G:/SYNC/School/VT/CLASSES/Panel Data (6614)/Final Project/LIBOR/Resid_r4 (PD).txt")

yHat_r1_St = pd.read_csv("G:/SYNC/School/VT/CLASSES/Panel Data (6614)/Final Project/LIBOR/yHat_r1 (PD).txt")
yHat_r2_St = pd.read_csv("G:/SYNC/School/VT/CLASSES/Panel Data (6614)/Final Project/LIBOR/yHat_r2 (PD).txt")
yHat_r3_St = pd.read_csv("G:/SYNC/School/VT/CLASSES/Panel Data (6614)/Final Project/LIBOR/yHat_r3 (PD).txt")
yHat_r4_St = pd.read_csv("G:/SYNC/School/VT/CLASSES/Panel Data (6614)/Final Project/LIBOR/yHat_r4 (PD).txt")

sigSqHat_St = pd.read_csv("G:/SYNC/School/VT/CLASSES/Panel Data (6614)/Final Project/LIBOR/SigSqHat (PD).txt")

sq_resid_r1_St = resid_r1_St**2
sq_resid_r2_St = resid_r2_St**2
sq_resid_r3_St = resid_r3_St**2
sq_resid_r4_St = resid_r4_St**2



############################################################
### Below MS tests each part of the VAR EQ (for each r1234)
############################################################

print()
print("Which variable of the VAR do you want to test? (Only one at a time)")
user=input("(r1, r2, r3 or r4): ").lower()

if(user=="r1"):
    ###############################################################################
    ###############################################################################
    ### r1 (Graphs & MS Testing) ##################################################
    ###############################################################################
    ###############################################################################
    
    ###                     Graph Residuals of Regressions over t               ###
    ###############################################################################
    
    ###### CHANGE if lags added/removed ######
    tempt = df['t'].tail(-4)
    
    fig, [ax1,ax2] = plt.subplots(2,1,constrained_layout=True)
    fig.suptitle('Students t Residuals', fontsize=14)
        
    ax1.plot(tempt, resid_r1_St, linewidth=1)
    ax1.set(ylabel='u')
    ax1.set(xlabel='t')
    ax2.plot(tempt, sq_resid_r1_St, linewidth=1)
    ax2.set(ylabel='u^2')
    ax2.set(xlabel='t')
    
    fig.tight_layout()
    
    ###         Compare histogram of residual to Student's t distribution       ###
    ###############################################################################
    
    # https://docs.scipy.org/doc/scipy/reference/stats.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html#scipy.stats.t
    # Google "python how to plot student's t distribution with specific degrees of freedom"
    
    import seaborn as sns
    from scipy import stats
    
    ###### CHANGE if lags added/removed ######
    tempt = df['t'].tail(-4)
    
    resid_hist=resid_r1_St.squeeze()
    
    fig, ax = plt.subplots()
    # plot the residuals
    sns.histplot(x=resid_hist, ax=ax, stat="density", linewidth=0, kde=True, label='Residual')
    ax.set(title="Distribution of Residuals (After MS Testing & Respecification)", xlabel="residual")
    # plot corresponding normal curve
    xmin, xmax = plt.xlim() # the maximum x values from the histogram above
    x = np.linspace(xmin, xmax, 1000) # generate some x values
    
    ### COMPARE TO DISTRIBUTIONS ###
    # Show Student's t distribution
    #v, mu, std = stats.t.fit(resid_hist)
        # v = degrees of freedom
    #p = stats.t.pdf(x, df=3, loc=mu, scale=std) # calculate the y values for the Students t curve
    #sns.lineplot(x=x, y=p, color="orange", ax=ax,label="Students t (v=3)")

    # Show Student's t distribution
    v, mu, std = stats.t.fit(resid_hist)
        # v = degrees of freedom
    p = stats.t.pdf(x, df=2, loc=mu, scale=std) # calculate the y values for the Students t curve
    sns.lineplot(x=x, y=p, color="green", ax=ax,label="Students t (v=2)")
    
    v, mu, std = stats.t.fit(resid_hist)
        # v = degrees of freedom
    p = stats.t.pdf(x, df=4, loc=mu, scale=std) # calculate the y values for the Students t curve
    sns.lineplot(x=x, y=p, color="red", ax=ax,label="Students t (v=4)")
    
    # Show Normal distribution
    muNorm, stdNorm = stats.norm.fit(resid_hist)
    q = stats.norm.pdf(x, muNorm, stdNorm) # calculate the y values for the normal curve
    sns.lineplot(x=x, y=q, color="black", ax=ax,label="Normal")
    
    plt.show()
    
    ########################### Auxiliary Regressions #############################

    
    
    ###                     Auxilliary regression - 1st Moment                  ###
    ###############################################################################
    
    ### ALL variables to test
    X1 = pd.DataFrame(index=range(len(df)),columns=range(0))
    X1['constant'] = 1
    
    X1['trend'] = df['trend']
    X1['trend2'] = df['trend2']
    #X1['trend3'] = df['trend3']
    #X1['trend4'] = df['trend4']
    
    X1['uLag1'] = resid_r1_St.shift(1)
    #
    X1['uLag2'] = resid_r1_St.shift(2)
    #
    X1['uLag3'] = resid_r1_St.shift(3)
    #X1['uLag4'] = resid_r1_St.shift(4)
    
    X1['yHat'] = yHat_r1_St
    X1['yHatSQ'] = yHat_r1_St**2
    
    X1['diffr1Lag1'] = df['diffr1'].shift(1)
    X1['diffr1Lag2'] = df['diffr1'].shift(2)
    X1['diffr1Lag3'] = df['diffr1'].shift(3)
    #X1['diffr1Lag4'] = df['diffr1'].shift(4)
    
    X1['diffr2Lag1'] = df['diffr2'].shift(1)
    X1['diffr2Lag2'] = df['diffr2'].shift(2)
    X1['diffr2Lag3'] = df['diffr2'].shift(3)
    #X1['diffr2Lag4'] = df['diffr2'].shift(4)
    
    X1['diffr3Lag1'] = df['diffr3'].shift(1)
    X1['diffr3Lag2'] = df['diffr3'].shift(2)
    X1['diffr3Lag3'] = df['diffr3'].shift(3)
    #X1['diffr3Lag4'] = df['diffr3'].shift(4)
    
    X1['diffr4Lag1'] = df['diffr4'].shift(1)
    X1['diffr4Lag2'] = df['diffr4'].shift(2)
    X1['diffr4Lag3'] = df['diffr4'].shift(3)
    #X1['diffr4Lag4'] = df['diffr4'].shift(4)
    
    #X1['diffr1Lag1SQ'] = df['diffr1Lag1']**2
    #X1['diffr2Lag1SQ'] = df['diffr2Lag1']**2
    #X1['diffr3Lag1SQ'] = df['diffr3Lag1']**2
    #X1['diffr4Lag1SQ'] = df['diffr4Lag1']**2
    
    X1['r1Lag1'] = df['r1Lag1']
    X1['r2Lag1'] = df['r2Lag1']
    X1['r3Lag1'] = df['r3Lag1']
    X1['r4Lag1'] = df['r4Lag1']
    
    ###### CHANGE if lags added/removed ######
    X1=X1.tail(-8) # to drop FIRST 8 rows
    X1=X1.head(-4) # to drop LAST 4 rows
    resid_r1_St=resid_r1_St.tail(-8) # to drop FIRST 8 rows
    
    auxreg1 = sm.OLS(resid_r1_St, X1).fit()
    moment1Result = auxreg1.summary()
    print(moment1Result)
    
    auxresid1 = auxreg1.resid
    
    
    
    ###                     Auxilliary regression - 2nd Moment                  ###
    ###############################################################################
    
    ### ALL variables to test
    X2 = pd.DataFrame(index=range(len(df)),columns=range(0))
    X2['constant'] = 1
    
    #X2['trend'] = df['trend']
    #X2['trend2'] = df['trend2']
    X2['trend3'] = df['trend3']
    X2['trend4'] = df['trend4']
    
    X2['sigSq'] = sigSqHat_St  
    X2['yHatSQ'] = yHat_r1_St**2
    
    X2['sigSqLag1'] = sigSqHat_St.shift(1)
    #X2['sigSqLag2'] = sigSqHat_St.shift(2)
    #X2['sigSqLag3'] = sigSqHat_St.shift(3)
    #X2['sigSqLag4'] = sigSqHat_St.shift(4)
    
    X1['r1Lag1SQ'] = df['r1Lag1']**2
    
    ###### CHANGE if lags added/removed ######
    X2=X2.tail(-8) # to drop FIRST 4 rows
    X2=X2.head(-4) # to drop LAST 4 rows
    sq_resid_r1_St=sq_resid_r1_St.tail(-8) # to drop FIRST 4 rows
    
    auxreg2 = sm.OLS(sq_resid_r1_St, X2).fit()
    moment2Result = auxreg2.summary()
    print(moment2Result)
    
    auxresid2 = auxreg2.resid



elif(user=="r2"):
    ###############################################################################
    ###############################################################################
    ### r2 (Graphs & MS Testing) ##################################################
    ###############################################################################
    ###############################################################################
    
    ###                     Graph Residuals of Regressions over t               ###
    ###############################################################################
    
    ###### CHANGE if lags added/removed ######
    tempt = df['t'].tail(-4)
    
    fig, [ax1,ax2] = plt.subplots(2,1,constrained_layout=True)
    fig.suptitle('Students t Residuals', fontsize=14)
        
    ax1.plot(tempt, resid_r2_St, linewidth=1)
    ax1.set(ylabel='u')
    ax1.set(xlabel='t')
    ax2.plot(tempt, sq_resid_r2_St, linewidth=1)
    ax2.set(ylabel='u^2')
    ax2.set(xlabel='t')
    
    fig.tight_layout()
    
    ###         Compare histogram of residual to Student's t distribution       ###
    ###############################################################################
    
    # https://docs.scipy.org/doc/scipy/reference/stats.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html#scipy.stats.t
    # Google "python how to plot student's t distribution with specific degrees of freedom"
    
    import seaborn as sns
    from scipy import stats
    
    ###### CHANGE if lags added/removed ######
    tempt = df['t'].tail(-4)
    
    resid_hist=resid_r2_St.squeeze()
    
    fig, ax = plt.subplots()
    # plot the residuals
    sns.histplot(x=resid_hist, ax=ax, stat="density", linewidth=0, kde=True, label='Residual')
    ax.set(title="Distribution of Residuals (After MS Testing & Respecification)", xlabel="residual")
    # plot corresponding normal curve
    xmin, xmax = plt.xlim() # the maximum x values from the histogram above
    x = np.linspace(xmin, xmax, 1000) # generate some x values
    
    ### COMPARE TO DISTRIBUTIONS ###
    # Show Student's t distribution
    #v, mu, std = stats.t.fit(resid_hist)
        # v = degrees of freedom
    #p = stats.t.pdf(x, df=3, loc=mu, scale=std) # calculate the y values for the Students t curve
    #sns.lineplot(x=x, y=p, color="orange", ax=ax,label="Students t (v=3)")
    
    # Show Student's t distribution
    v, mu, std = stats.t.fit(resid_hist)
        # v = degrees of freedom
    p = stats.t.pdf(x, df=2, loc=mu, scale=std) # calculate the y values for the Students t curve
    sns.lineplot(x=x, y=p, color="green", ax=ax,label="Students t (v=2)")
    
    v, mu, std = stats.t.fit(resid_hist)
        # v = degrees of freedom
    p = stats.t.pdf(x, df=4, loc=mu, scale=std) # calculate the y values for the Students t curve
    sns.lineplot(x=x, y=p, color="red", ax=ax,label="Students t (v=4)")
    
    # Show Normal distribution
    muNorm, stdNorm = stats.norm.fit(resid_hist)
    q = stats.norm.pdf(x, muNorm, stdNorm) # calculate the y values for the normal curve
    sns.lineplot(x=x, y=q, color="black", ax=ax,label="Normal")
    
    plt.show()
    
    ########################### Auxiliary Regressions #############################
    
    
    
    ###                     Auxilliary regression - 1st Moment                  ###
    ###############################################################################
    
    ### ALL variables to test
    X1 = pd.DataFrame(index=range(len(df)),columns=range(0))
    X1['constant'] = 1
    
    X1['trend'] = df['trend']
    X1['trend2'] = df['trend2']
    #X1['trend3'] = df['trend3']
    #X1['trend4'] = df['trend4']
    
    X1['uLag1'] = resid_r2_St.shift(1)
    #
    X1['uLag2'] = resid_r2_St.shift(2)
    #
    X1['uLag3'] = resid_r2_St.shift(3)
    #X1['uLag4'] = resid_r2_St.shift(4)
    
    X1['yHat'] = yHat_r2_St
    X1['yHatSQ'] = yHat_r2_St**2
    
    X1['diffr1Lag1'] = df['diffr1'].shift(1)
    X1['diffr1Lag2'] = df['diffr1'].shift(2)
    X1['diffr1Lag3'] = df['diffr1'].shift(3)
    #X1['diffr1Lag4'] = df['diffr1'].shift(4)
    
    X1['diffr2Lag1'] = df['diffr2'].shift(1)
    X1['diffr2Lag2'] = df['diffr2'].shift(2)
    X1['diffr2Lag3'] = df['diffr2'].shift(3)
    #X1['diffr2Lag4'] = df['diffr2'].shift(4)
    
    X1['diffr3Lag1'] = df['diffr3'].shift(1)
    X1['diffr3Lag2'] = df['diffr3'].shift(2)
    X1['diffr3Lag3'] = df['diffr3'].shift(3)
    #X1['diffr3Lag4'] = df['diffr3'].shift(4)
    
    X1['diffr4Lag1'] = df['diffr4'].shift(1)
    X1['diffr4Lag2'] = df['diffr4'].shift(2)
    X1['diffr4Lag3'] = df['diffr4'].shift(3)
    #X1['diffr4Lag4'] = df['diffr4'].shift(4)
    
    #X1['diffr1Lag1SQ'] = df['diffr1Lag1']**2
    #X1['diffr2Lag1SQ'] = df['diffr2Lag1']**2
    #X1['diffr3Lag1SQ'] = df['diffr3Lag1']**2
    #X1['diffr4Lag1SQ'] = df['diffr4Lag1']**2
    
    X1['r1Lag1'] = df['r1Lag1']
    X1['r2Lag1'] = df['r2Lag1']
    X1['r3Lag1'] = df['r3Lag1']
    X1['r4Lag1'] = df['r4Lag1']
    
    ###### CHANGE if lags added/removed ######
    X1=X1.tail(-8) # to drop FIRST 8 rows
    X1=X1.head(-4) # to drop LAST 4 rows
    resid_r2_St=resid_r2_St.tail(-8) # to drop FIRST 8 rows
    
    auxreg1 = sm.OLS(resid_r2_St, X1).fit()
    moment1Result = auxreg1.summary()
    print(moment1Result)
    
    auxresid1 = auxreg1.resid
    
    
    
    ###                     Auxilliary regression - 2nd Moment                  ###
    ###############################################################################
    
    ### ALL variables to test
    X2 = pd.DataFrame(index=range(len(df)),columns=range(0))
    X2['constant'] = 1
    
    #X2['trend'] = df['trend']
    #X2['trend2'] = df['trend2']
    X2['trend3'] = df['trend3']
    X2['trend4'] = df['trend4']
    
    X2['sigSq'] = sigSqHat_St  
    X2['yHatSQ'] = yHat_r1_St**2
    
    X2['sigSqLag1'] = sigSqHat_St.shift(1)
    #X2['sigSqLag2'] = sigSqHat_St.shift(2)
    #X2['sigSqLag3'] = sigSqHat_St.shift(3)
    #X2['sigSqLag4'] = sigSqHat_St.shift(4)
    
    ###### CHANGE if lags added/removed ######
    X2=X2.tail(-8) # to drop FIRST 4 rows
    X2=X2.head(-4) # to drop LAST 4 rows
    sq_resid_r2_St=sq_resid_r2_St.tail(-8) # to drop FIRST 4 rows
    
    auxreg2 = sm.OLS(sq_resid_r2_St, X2).fit()
    moment2Result = auxreg2.summary()
    print(moment2Result)
    
    auxresid2 = auxreg2.resid



elif(user=="r3"):
    ###############################################################################
    ###############################################################################
    ### r3 (Graphs & MS Testing) ##################################################
    ###############################################################################
    ###############################################################################
    
    ###                     Graph Residuals of Regressions over t               ###
    ###############################################################################
    
    ###### CHANGE if lags added/removed ######
    tempt = df['t'].tail(-4)
    
    fig, [ax1,ax2] = plt.subplots(2,1,constrained_layout=True)
    fig.suptitle('Students t Residuals', fontsize=14)
        
    ax1.plot(tempt, resid_r3_St, linewidth=1)
    ax1.set(ylabel='u')
    ax1.set(xlabel='t')
    ax2.plot(tempt, sq_resid_r3_St, linewidth=1)
    ax2.set(ylabel='u^2')
    ax2.set(xlabel='t')
    
    fig.tight_layout()
    
    ###         Compare histogram of residual to Student's t distribution       ###
    ###############################################################################
    
    # https://docs.scipy.org/doc/scipy/reference/stats.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html#scipy.stats.t
    # Google "python how to plot student's t distribution with specific degrees of freedom"
    
    import seaborn as sns
    from scipy import stats
    
    ###### CHANGE if lags added/removed ######
    tempt = df['t'].tail(-4)
    
    resid_hist=resid_r3_St.squeeze()
    
    fig, ax = plt.subplots()
    # plot the residuals
    sns.histplot(x=resid_hist, ax=ax, stat="density", linewidth=0, kde=True, label='Residual')
    ax.set(title="Distribution of Residuals (After MS Testing & Respecification)", xlabel="residual")
    # plot corresponding normal curve
    xmin, xmax = plt.xlim() # the maximum x values from the histogram above
    x = np.linspace(xmin, xmax, 1000) # generate some x values
    
    ### COMPARE TO DISTRIBUTIONS ###
    # Show Student's t distribution
    #v, mu, std = stats.t.fit(resid_hist)
        # v = degrees of freedom
    #p = stats.t.pdf(x, df=3, loc=mu, scale=std) # calculate the y values for the Students t curve
    #sns.lineplot(x=x, y=p, color="orange", ax=ax,label="Students t (v=3)")
    
    # Show Student's t distribution
    v, mu, std = stats.t.fit(resid_hist)
        # v = degrees of freedom
    p = stats.t.pdf(x, df=2, loc=mu, scale=std) # calculate the y values for the Students t curve
    sns.lineplot(x=x, y=p, color="green", ax=ax,label="Students t (v=2)")
    
    v, mu, std = stats.t.fit(resid_hist)
        # v = degrees of freedom
    p = stats.t.pdf(x, df=4, loc=mu, scale=std) # calculate the y values for the Students t curve
    sns.lineplot(x=x, y=p, color="red", ax=ax,label="Students t (v=4)")
    
    # Show Normal distribution
    muNorm, stdNorm = stats.norm.fit(resid_hist)
    q = stats.norm.pdf(x, muNorm, stdNorm) # calculate the y values for the normal curve
    sns.lineplot(x=x, y=q, color="black", ax=ax,label="Normal")
    
    plt.show()
    
    ########################### Auxiliary Regressions #############################
    
    
    
    ###                     Auxilliary regression - 1st Moment                  ###
    ###############################################################################
    
    ### ALL variables to test
    X1 = pd.DataFrame(index=range(len(df)),columns=range(0))
    X1['constant'] = 1
    
    X1['trend'] = df['trend']
    X1['trend2'] = df['trend2']
    #X1['trend3'] = df['trend3']
    #X1['trend4'] = df['trend4']
    
    X1['uLag1'] = resid_r3_St.shift(1)
    X1['uLag2'] = resid_r3_St.shift(2)
    X1['uLag3'] = resid_r3_St.shift(3)
    #X1['uLag4'] = resid_r3_St.shift(4)
    
    X1['yHat'] = yHat_r3_St
    X1['yHatSQ'] = yHat_r3_St**2
    
    X1['diffr1Lag1'] = df['diffr1'].shift(1)
    X1['diffr1Lag2'] = df['diffr1'].shift(2)
    X1['diffr1Lag3'] = df['diffr1'].shift(3)
    #X1['diffr1Lag4'] = df['diffr1'].shift(4)
    
    X1['diffr2Lag1'] = df['diffr2'].shift(1)
    X1['diffr2Lag2'] = df['diffr2'].shift(2)
    X1['diffr2Lag3'] = df['diffr2'].shift(3)
    #X1['diffr2Lag4'] = df['diffr2'].shift(4)
    
    X1['diffr3Lag1'] = df['diffr3'].shift(1)
    X1['diffr3Lag2'] = df['diffr3'].shift(2)
    X1['diffr3Lag3'] = df['diffr3'].shift(3)
    #X1['diffr3Lag4'] = df['diffr3'].shift(4)
    
    X1['diffr4Lag1'] = df['diffr4'].shift(1)
    X1['diffr4Lag2'] = df['diffr4'].shift(2)
    X1['diffr4Lag3'] = df['diffr4'].shift(3)
    #X1['diffr4Lag4'] = df['diffr4'].shift(4)
    
    #X1['diffr1Lag1SQ'] = df['diffr1Lag1']**2
    #X1['diffr2Lag1SQ'] = df['diffr2Lag1']**2
    #X1['diffr3Lag1SQ'] = df['diffr3Lag1']**2
    #X1['diffr4Lag1SQ'] = df['diffr4Lag1']**2
    
    X1['r1Lag1'] = df['r1Lag1']
    X1['r2Lag1'] = df['r2Lag1']
    X1['r3Lag1'] = df['r3Lag1']
    X1['r4Lag1'] = df['r4Lag1']    
        
    ###### CHANGE if lags added/removed ######
    X1=X1.tail(-8) # to drop FIRST 8 rows
    X1=X1.head(-4) # to drop LAST 4 rows
    resid_r3_St=resid_r3_St.tail(-8) # to drop FIRST 8 rows
    
    auxreg1 = sm.OLS(resid_r3_St, X1).fit()
    moment1Result = auxreg1.summary()
    print(moment1Result)
    
    auxresid1 = auxreg1.resid
    
    
    
    ###                     Auxilliary regression - 2nd Moment                  ###
    ###############################################################################
    
    ### ALL variables to test
    X2 = pd.DataFrame(index=range(len(df)),columns=range(0))
    X2['constant'] = 1
    
    #X2['trend'] = df['trend']
    #X2['trend2'] = df['trend2']
    X2['trend3'] = df['trend3']
    X2['trend4'] = df['trend4']
    
    X2['sigSq'] = sigSqHat_St  
    X2['yHatSQ'] = yHat_r1_St**2
    
    X2['sigSqLag1'] = sigSqHat_St.shift(1)
    #X2['sigSqLag2'] = sigSqHat_St.shift(2)
    #X2['sigSqLag3'] = sigSqHat_St.shift(3)
    #X2['sigSqLag4'] = sigSqHat_St.shift(4)
    
    ###### CHANGE if lags added/removed ######
    X2=X2.tail(-8) # to drop FIRST 4 rows
    X2=X2.head(-4) # to drop LAST 4 rows
    sq_resid_r3_St=sq_resid_r3_St.tail(-8) # to drop FIRST 4 rows
    
    auxreg2 = sm.OLS(sq_resid_r3_St, X2).fit()
    moment2Result = auxreg2.summary()
    print(moment2Result)
    
    auxresid2 = auxreg2.resid



elif(user=="r4"):
    ###############################################################################
    ###############################################################################
    ### r4 (Graphs & MS Testing) ##################################################
    ###############################################################################
    ###############################################################################
    
    ###                     Graph Residuals of Regressions over t               ###
    ###############################################################################
    
    ###### CHANGE if lags added/removed ######
    tempt = df['t'].tail(-4)
    
    fig, [ax1,ax2] = plt.subplots(2,1,constrained_layout=True)
    fig.suptitle('Students t Residuals', fontsize=14)
        
    ax1.plot(tempt, resid_r4_St, linewidth=1)
    ax1.set(ylabel='u')
    ax1.set(xlabel='t')
    ax2.plot(tempt, sq_resid_r4_St, linewidth=1)
    ax2.set(ylabel='u^2')
    ax2.set(xlabel='t')
    
    fig.tight_layout()
    
    ###         Compare histogram of residual to Student's t distribution       ###
    ###############################################################################
    
    # https://docs.scipy.org/doc/scipy/reference/stats.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html#scipy.stats.t
    # Google "python how to plot student's t distribution with specific degrees of freedom"
    
    import seaborn as sns
    from scipy import stats
    
    ###### CHANGE if lags added/removed ######
    tempt = df['t'].tail(-4)
    
    resid_hist=resid_r4_St.squeeze()
    
    fig, ax = plt.subplots()
    # plot the residuals
    sns.histplot(x=resid_hist, ax=ax, stat="density", linewidth=0, kde=True, label='Residual')
    ax.set(title="Distribution of Residuals (After MS Testing & Respecification)", xlabel="residual")
    # plot corresponding normal curve
    xmin, xmax = plt.xlim() # the maximum x values from the histogram above
    x = np.linspace(xmin, xmax, 1000) # generate some x values
    
    ### COMPARE TO DISTRIBUTIONS ###
    # Show Student's t distribution
    #v, mu, std = stats.t.fit(resid_hist)
        # v = degrees of freedom
    #p = stats.t.pdf(x, df=3, loc=mu, scale=std) # calculate the y values for the Students t curve
    #sns.lineplot(x=x, y=p, color="orange", ax=ax,label="Students t (v=3)")
    
    # Show Student's t distribution
    v, mu, std = stats.t.fit(resid_hist)
        # v = degrees of freedom
    p = stats.t.pdf(x, df=2, loc=mu, scale=std) # calculate the y values for the Students t curve
    sns.lineplot(x=x, y=p, color="green", ax=ax,label="Students t (v=2)")

    v, mu, std = stats.t.fit(resid_hist)
        # v = degrees of freedom
    p = stats.t.pdf(x, df=4, loc=mu, scale=std) # calculate the y values for the Students t curve
    sns.lineplot(x=x, y=p, color="red", ax=ax,label="Students t (v=4)")
    
    # Show Normal distribution
    muNorm, stdNorm = stats.norm.fit(resid_hist)
    q = stats.norm.pdf(x, muNorm, stdNorm) # calculate the y values for the normal curve
    sns.lineplot(x=x, y=q, color="black", ax=ax,label="Normal")
    
    plt.show()
    
    ########################### Auxiliary Regressions #############################
    
    
    
    ###                     Auxilliary regression - 1st Moment                  ###
    ###############################################################################
    
    ### ALL variables to test
    X1 = pd.DataFrame(index=range(len(df)),columns=range(0))
    X1['constant'] = 1
    
    X1['trend'] = df['trend']
    X1['trend2'] = df['trend2']
    #X1['trend3'] = df['trend3']
    #X1['trend4'] = df['trend4']
    
    X1['uLag1'] = resid_r4_St.shift(1)
    X1['uLag2'] = resid_r4_St.shift(2)
    X1['uLag3'] = resid_r4_St.shift(3)
    #X1['uLag4'] = resid_r4_St.shift(4)
    
    X1['yHat'] = yHat_r4_St
    X1['yHatSQ'] = yHat_r4_St**2
    
    X1['diffr1Lag1'] = df['diffr1'].shift(1)
    X1['diffr1Lag2'] = df['diffr1'].shift(2)
    X1['diffr1Lag3'] = df['diffr1'].shift(3)
    #X1['diffr1Lag4'] = df['diffr1'].shift(4)
    
    X1['diffr2Lag1'] = df['diffr2'].shift(1)
    X1['diffr2Lag2'] = df['diffr2'].shift(2)
    X1['diffr2Lag3'] = df['diffr2'].shift(3)
    #X1['diffr2Lag4'] = df['diffr2'].shift(4)
    
    X1['diffr3Lag1'] = df['diffr3'].shift(1)
    X1['diffr3Lag2'] = df['diffr3'].shift(2)
    X1['diffr3Lag3'] = df['diffr3'].shift(3)
    #X1['diffr3Lag4'] = df['diffr3'].shift(4)
    
    X1['diffr4Lag1'] = df['diffr4'].shift(1)
    X1['diffr4Lag2'] = df['diffr4'].shift(2)
    X1['diffr4Lag3'] = df['diffr4'].shift(3)
    #X1['diffr4Lag4'] = df['diffr4'].shift(4)
    
    #X1['diffr1Lag1SQ'] = df['diffr1Lag1']**2
    #X1['diffr2Lag1SQ'] = df['diffr2Lag1']**2
    #X1['diffr3Lag1SQ'] = df['diffr3Lag1']**2
    #X1['diffr4Lag1SQ'] = df['diffr4Lag1']**2
    
    X1['r1Lag1'] = df['r1Lag1']
    X1['r2Lag1'] = df['r2Lag1']
    X1['r3Lag1'] = df['r3Lag1']
    X1['r4Lag1'] = df['r4Lag1']
        
    ###### CHANGE if lags added/removed ######
    X1=X1.tail(-8) # to drop FIRST 8 rows
    X1=X1.head(-4) # to drop LAST 4 rows
    resid_r4_St=resid_r4_St.tail(-8) # to drop FIRST 8 rows
    
    auxreg1 = sm.OLS(resid_r4_St, X1).fit()
    moment1Result = auxreg1.summary()
    print(moment1Result)
    
    auxresid1 = auxreg1.resid
    
    
    
    ###                     Auxilliary regression - 2nd Moment                  ###
    ###############################################################################
    
    ### ALL variables to test
    X2 = pd.DataFrame(index=range(len(df)),columns=range(0))
    X2['constant'] = 1
    
    #X2['trend'] = df['trend']
    #X2['trend2'] = df['trend2']
    X2['trend3'] = df['trend3']
    X2['trend4'] = df['trend4']
    
    X2['sigSq'] = sigSqHat_St  
    X2['yHatSQ'] = yHat_r1_St**2
    
    X2['sigSqLag1'] = sigSqHat_St.shift(1)
    #X2['sigSqLag2'] = sigSqHat_St.shift(2)
    #X2['sigSqLag3'] = sigSqHat_St.shift(3)
    #X2['sigSqLag4'] = sigSqHat_St.shift(4)
    
    ###### CHANGE if lags added/removed ######
    X2=X2.tail(-8) # to drop FIRST 4 rows
    X2=X2.head(-4) # to drop LAST 4 rows
    sq_resid_r4_St=sq_resid_r4_St.tail(-8) # to drop FIRST 4 rows
    
    auxreg2 = sm.OLS(sq_resid_r4_St, X2).fit()
    moment2Result = auxreg2.summary()
    print(moment2Result)
    
    auxresid2 = auxreg2.resid



else:
    print("INVALID Selction!!!")



##########################################################
### IF YOU WANT TO COMPARE DISTRIBUTION TO STUDENT'S T ###
##########################################################

import seaborn as sns
from scipy import stats

###### CHANGE if lags added/removed ######
tempt = df['t'].tail(-4)

resid_hist=resid_r1_St.squeeze()

fig, ax = plt.subplots()
# plot the residuals
sns.histplot(x=resid_hist, ax=ax, stat="density", linewidth=0, kde=True, label='Residual')
ax.set(title="Distribution of Residuals (After MS Testing & Respecification)", xlabel="residual")
# plot corresponding normal curve
xmin, xmax = plt.xlim() # the maximum x values from the histogram above
x = np.linspace(xmin, xmax, 1000) # generate some x values

### COMPARE TO DISTRIBUTIONS ###
# Show Normal distribution
muNorm, stdNorm = stats.norm.fit(resid_hist)
q = stats.norm.pdf(x, muNorm, stdNorm) # calculate the y values for the normal curve
sns.lineplot(x=x, y=q, color="black", ax=ax,label="Normal")

# Show Student's t distribution (v=3)
v, mu, std = stats.t.fit(resid_hist)
    # v = degrees of freedom
p = stats.t.pdf(x, df=3, loc=mu, scale=std) # calculate the y values for the Students t curve
sns.lineplot(x=x, y=p, color="orange", ax=ax,label="Students t (v=3)")

"""
# Show Student's t distribution (v=4)
v, mu, std = stats.t.fit(resid_hist)
    # v = degrees of freedom
p = stats.t.pdf(x, df=4, loc=mu, scale=std) # calculate the y values for the Students t curve
sns.lineplot(x=x, y=p, color="brown", ax=ax,label="Students t (v=4)")

# Show Student's t distribution (v=5)
v, mu, std = stats.t.fit(resid_hist)
    # v = degrees of freedom
p = stats.t.pdf(x, df=5, loc=mu, scale=std) # calculate the y values for the Students t curve
sns.lineplot(x=x, y=p, color="green", ax=ax,label="Students t (v=5)")

# Show Student's t distribution (v=6)
v, mu, std = stats.t.fit(resid_hist)
    # v = degrees of freedom
p = stats.t.pdf(x, df=6, loc=mu, scale=std) # calculate the y values for the Students t curve
sns.lineplot(x=x, y=p, color="red", ax=ax,label="Students t (v=6)")
"""

plt.show()
