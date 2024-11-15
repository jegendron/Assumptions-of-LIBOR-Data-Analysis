
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

df = pd.read_excel("G:/SYNC/School/VT/CLASSES/STUDENT/SPANOS/Final Project/LIBOR/Libor-data-2022.xlsx")

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

###### IF Testing log-linear model
#df['e']=np.log(df['e'])

###                         Variables to test [2-3]                         ###
###############################################################################

#df['depVarSQ'] = df['depVar']**2 
###### IF testing log-linear model
# df['depVarSQ'] = np.log(df['depVar'])**2 

#df['x1SQ'] = df['x1']**2



###                          Variables to test [4]                          ###
###############################################################################

df['diffr1Lag1'] = df['diffr1'].shift(1)
df['diffr1Lag2'] = df['diffr1'].shift(2)
df['diffr1Lag3'] = df['diffr1'].shift(3)
df['diffr1Lag4'] = df['diffr1'].shift(4)

df['diffr1Lag1SQ'] = df['diffr1Lag1']**2
df['diffr1Lag2SQ'] = df['diffr1Lag2']**2
df['diffr1Lag3SQ'] = df['diffr1Lag3']**2
df['diffr1Lag4SQ'] = df['diffr1Lag4']**2



###                          Variables to test [5]                          ###
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
###############################################################################
### t-Plots ###################################################################
###############################################################################

fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2,constrained_layout=True)
#fig, [[ax1,ax2],[ax3,ax4],[ax5,ax6],[ax7,ax8]] = plt.subplots(4,2,constrained_layout=True)
fig.suptitle('All Variables over Time (Differenced)')
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
#ax5.plot(df['t'], df['diffr5'], linewidth=1)
#ax5.set(ylabel='r5')

###r7###
#ax6.plot(df['t'], df['diffr7'], linewidth=1)
#ax6.set(ylabel='r7')

###r10###
#ax7.plot(df['t'], df['diffr10'], linewidth=1)
#ax7.set(ylabel='r10')

###r30###
#ax8.plot(df['t'], df['diffr30'], linewidth=1)
#ax8.set(ylabel='r30')

#fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.025),
#          ncol=5, facecolor="white", fancybox=True, shadow=True)
fig.tight_layout()



fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2,constrained_layout=True)
#fig, [[ax1,ax2],[ax3,ax4],[ax5,ax6],[ax7,ax8]] = plt.subplots(4,2,constrained_layout=True)
fig.suptitle('All Variables over Time')
###r1###
ax1.plot(df['t'], df['r1'], linewidth=1)
ax1.set(ylabel='r1')

###r2###
ax2.plot(df['t'], df['r2'], linewidth=1)
ax2.set(ylabel="r2")

###r3###
ax3.plot(df['t'], df['r3'], linewidth=1)
ax3.set(ylabel='r3')

###r4###
ax4.plot(df['t'], df['r4'], linewidth=1)
ax4.set(ylabel='r4')

###r5###
#ax5.plot(df['t'], df['r5'], linewidth=1)
#ax5.set(ylabel='r5')

###r7###
#ax6.plot(df['t'], df['r7'], linewidth=1)
#ax6.set(ylabel='r7')

###r10###
#ax7.plot(df['t'], df['r10'], linewidth=1)
#ax7.set(ylabel='r10')

###r30###
#ax8.plot(df['t'], df['r30'], linewidth=1)
#ax8.set(ylabel='r30')

#fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.025),
#          ncol=5, facecolor="white", fancybox=True, shadow=True)
fig.tight_layout()
"""



###############################################################################
### M-S Testing ###############################################################
###############################################################################



################ Generate Residuals for Auxiliary Regressions ################# 
import statsmodels.api as sm

# Starts with base model, each new line adds variables based on M-S testing results
    # for example: if trend found, then add trend into the regression 
        # (which updates the residuals you MS test with!)

### Base EQ
X = pd.DataFrame(index=range(len(df)),columns=range(0))

X['constant'] = 1
#
X['mu'] = np.mean(df['diffr1'])

### FINAL Model
#X['trend'] = df['trend']
#X['trend2'] = df['trend2']
#X['diffr1Lag1'] = df['diffr1Lag1']
#X['diffr1Lag2'] = df['diffr1Lag2']
#X['diffr1Lag3'] = df['diffr1Lag3']  
#X['r1Lag1'] = df['r1Lag1']



###### CHANGE if lags added/removed ######
    # Now drop (1) rows B/C of the lag, without changing the dataframe
#X=X.tail(-1)
    #BUT change to below to account for lag4 terms in aux regs
#
X=X.tail(-5)

y = df['diffr1']
###### IF Testing log-linear model
# y = np.log(df['e'])

###### CHANGE if lags added/removed ######
#y=y.tail(-1)
    #BUT change to below to account for lag4 terms in aux regs
#
y=y.tail(-5)

# == reg X on y
reg1 = sm.OLS(y, X).fit()
initialResult = reg1.summary()
#print(initialResult)

resid = reg1.resid
sq_resid = resid**2

###                     Graph Residuals of Regressions over t               ###
###############################################################################

###### CHANGE if lags added/removed ######
tempt = df['t'].tail(-5)

fig, [ax1,ax2] = plt.subplots(2,1,constrained_layout=True)
fig.suptitle('Normal Residuals', fontsize=14)
    
ax1.plot(tempt, resid, linewidth=1)
ax1.set(ylabel='u')
ax1.set(xlabel='t')
ax2.plot(tempt, sq_resid, linewidth=1)
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
tempt = df['t'].tail(-5)

resid_hist=resid.squeeze()

fig, ax = plt.subplots()
# plot the residuals
sns.histplot(x=resid_hist, ax=ax, stat="density", linewidth=0, kde=True, label='Residual')
ax.set(title="Normal Residuals (After MS Testing & Respecification)", xlabel="residual")
# plot corresponding normal curve
xmin, xmax = plt.xlim() # the maximum x values from the histogram above
x = np.linspace(xmin, xmax, 1000) # generate some x values

### COMPARE TO DISTRIBUTIONS ###
# Show Student's t distribution
v, mu, std = stats.t.fit(resid_hist)
    # v = degrees of freedom
    
# If your distribution shape looks off, you can manually adjust the values of v
p = stats.t.pdf(x, df=3, loc=mu, scale=std) # calculate the y values for the Students t curve
sns.lineplot(x=x, y=p, color="orange", ax=ax,label="Students t (v=3)")

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
#
X1['mu'] = X['mu']

X1['trend'] = df['trend']
X1['trend2'] = df['trend2']
#
X1['trend3'] = df['trend3']
#
X1['trend4'] = df['trend4']

X1['diffr1Lag1'] = df['diffr1Lag1']
X1['diffr1Lag2'] = df['diffr1Lag2']
X1['diffr1Lag3'] = df['diffr1Lag3']
#
X1['diffr1Lag4'] = df['diffr1Lag4']

X1['r1Lag1'] = df['r1Lag1']



###### CHANGE if lags added/removed ######
X1=X1.tail(-5) # to drop FIRST 5 rows

auxreg1 = sm.OLS(resid, X1).fit()
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
#
X2['trend3'] = df['trend3']
#
X2['trend4'] = df['trend4']

X2['diffr1Lag1SQ'] = df['diffr1Lag1SQ']
X2['diffr1Lag2SQ'] = df['diffr1Lag2SQ']
X2['diffr1Lag3SQ'] = df['diffr1Lag3SQ']
#X2['diffr1Lag4SQ'] = df['diffr1Lag4SQ']

X2['r1Lag1SQ'] = df['r1Lag1']**2



###### CHANGE if lags added/removed ######
X2=X2.tail(-5) # to drop FIRST 5 rows

auxreg2 = sm.OLS(sq_resid, X2).fit()
moment2Result = auxreg2.summary()
print(moment2Result)

auxresid2 = auxreg2.resid



"""
###              Graph Residuals of Auxiliary regressions over t            ###
###############################################################################

###### CHANGE if lags added/removed ######

#tempt = df['t'].tail(-1) #drops 1st obs so can plot residuals
tempt = df['t'].tail(-4)

fig, [ax1,ax2] = plt.subplots(2,1,constrained_layout=True)
fig.suptitle('Aux Regs', fontsize=14)
    
ax1.plot(tempt, auxresid1, linewidth=1)
ax1.set(ylabel='u')
ax1.set(xlabel='t')
ax2.plot(tempt, auxresid2, linewidth=1)
ax2.set(ylabel='u^2')
ax2.set(xlabel='t')

fig.tight_layout()



############################### Test Normality ################################

userInput = input("Did you already test for Linearity, Homoskedasticity, Dependence, and t-invariance AND respecify generating new residuals? (Y/N):")
if(userInput!="Y" ):
    print("Please go back before you continue")

import seaborn as sns
from scipy import stats



qqplot(resid, line='s')
plt.show()

# Anderson Darling Normality test
result = anderson(resid)
print('Statistic: %.3f' % result.statistic)
p = 0
for i in range(len(result.critical_values)):
 sl, cv = result.significance_level[i], result.critical_values[i]
 if result.statistic < result.critical_values[i]:
     print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl," cv))
 else:
     print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))

"""
