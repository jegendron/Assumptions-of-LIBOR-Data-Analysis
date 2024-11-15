# https://www.rdocumentation.org/packages/StVAR/versions/1.1/topics/StVAR

############################################################################
### Import Needed Libraries ################################################
############################################################################

### Package dependencies
#install.packages('matlab')
#install.packages('ADGofTest')
#install.packages('MCMCpack')

### Download ARCHIVED package from CRAN archive
#url <- "https://cran.r-project.org/src/contrib/Archive/StVAR/StVAR_1.1.tar.gz"
#pkgFile <- "StVAR_1.1.tar.gz"
#download.file(url = url, destfile = pkgFile)
#install.packages(pkgs=pkgFile, type="source", repos=NULL)

library(StVAR)

### For importing from Excel
# install.packages("readxl")
library(readxl)

############################################################################
### Set Up Data ############################################################
############################################################################

## Random number seed
set.seed(7504)

n <- 1256

## Creating trend variable.
t <- seq(1,n,1)

### Import Data (from Python)
initialPanelData <- read_excel("G:\\SYNC\\School\\VT\\CLASSES\\Panel Data (6614)\\Final Project\\LIBOR\\panelDataSet.xlsx")
  # !!! Here just import 4 columns for 4 variables (r1, r2, r3 & r4) !!!

rLags <- read_excel("G:\\SYNC\\School\\VT\\CLASSES\\Panel Data (6614)\\Final Project\\LIBOR\\panelrLags.xlsx")
r1Lag1 <- rLags[,1]
r2Lag1 <- rLags[,2]
r3Lag1 <- rLags[,3]
r4Lag1 <- rLags[,4]

### All of below include r_t-1 to account for the unit root ONLY as a covariate
trend12 <- read_excel("G:\\SYNC\\School\\VT\\CLASSES\\Panel Data (6614)\\Final Project\\LIBOR\\trendData.xlsx")
trend12 <- data.matrix(cbind(1,trend12,rLags))

### EXTRA CODE ###
#trend12 <- data.matrix(cbind(1,trend12))
#trend12 <- data.matrix(cbind(1,trend12,r1Lag1))
#trend12 <- data.matrix(cbind(1,poly(t,2,raw=TRUE),rLags))
  ### Show BOTH trend & trend^2

#trend1 <- trend12[,-3]
  ### JUST show trend



############################################################################
### StVAR Model ############################################################
############################################################################

data <- initialPanelData
data <- data.matrix(data)

# Estimating the model
  # v = # of variables - 1 (i.e. n-1) 

stvar <- StVAR(data,lag=3,Trend=trend12,v=4,maxiter=2000,hes=TRUE)
  # v = 3 results in NaNs when >= 200 iters
  # v = 2 is good

  ### WARNING: Estimated Run Time ~= 15-20 minutes (for data of 1256 by 4 with 6 covariates)

  ### WARNING: Estimated Run Time ~= 5-10 minutes (for data of 1256 by 4 with 2 covariates)

stvar['beta']
stvar['coef']



############################################################################
### EXPORT (to analyze further in Python) ##################################
############################################################################

exportResids_r1 <- cbind(stvar$res[,1]) # residual
exportResids_r2 <- cbind(stvar$res[,2]) # residual
exportResids_r3 <- cbind(stvar$res[,3]) # residual
exportResids_r4 <- cbind(stvar$res[,4]) # residual

exportYhat_r1 <- cbind(stvar$fitted[,1]) # yHat
exportYhat_r2 <- cbind(stvar$fitted[,2]) # yHat
exportYhat_r3 <- cbind(stvar$fitted[,3]) # yHat
exportYhat_r4 <- cbind(stvar$fitted[,4]) # yHat

exportSigSqHat <- cbind(stvar$cvar)    # sigmaSqHat

# Resulting files in Documents folder
# 
write.table(exportResids_r1,file="G:\\SYNC\\School\\VT\\CLASSES\\Panel Data (6614)\\Final Project\\LIBOR\\Resid_r1 (PD).txt",row.names=FALSE)
# 
write.table(exportResids_r2,file="G:\\SYNC\\School\\VT\\CLASSES\\Panel Data (6614)\\Final Project\\LIBOR\\Resid_r2 (PD).txt",row.names=FALSE)
# 
write.table(exportResids_r3,file="G:\\SYNC\\School\\VT\\CLASSES\\Panel Data (6614)\\Final Project\\LIBOR\\Resid_r3 (PD).txt",row.names=FALSE)
# 
write.table(exportResids_r4,file="G:\\SYNC\\School\\VT\\CLASSES\\Panel Data (6614)\\Final Project\\LIBOR\\Resid_r4 (PD).txt",row.names=FALSE)

# 
write.table(exportYhat_r1,file="G:\\SYNC\\School\\VT\\CLASSES\\Panel Data (6614)\\Final Project\\LIBOR\\yHat_r1 (PD).txt",row.names=FALSE)
# 
write.table(exportYhat_r2,file="G:\\SYNC\\School\\VT\\CLASSES\\Panel Data (6614)\\Final Project\\LIBOR\\yHat_r2 (PD).txt",row.names=FALSE)
# 
write.table(exportYhat_r3,file="G:\\SYNC\\School\\VT\\CLASSES\\Panel Data (6614)\\Final Project\\LIBOR\\yHat_r3 (PD).txt",row.names=FALSE)
#
write.table(exportYhat_r4,file="G:\\SYNC\\School\\VT\\CLASSES\\Panel Data (6614)\\Final Project\\LIBOR\\yHat_r4 (PD).txt",row.names=FALSE)
 
# 
write.table(exportSigSqHat,file="G:\\SYNC\\School\\VT\\CLASSES\\Panel Data (6614)\\Final Project\\LIBOR\\SigSqHat (PD).txt",row.names=FALSE)



############################################################################
### EXTRA CODE: Plots in R #################################################
############################################################################

###### Plotting the variable y, its estimated trend and the fitted value ######
# lag <- 3
# 
# d <- seq(1,n-lag,1)
# y <- data.matrix(initialPanelData)
# y <- y[-(1:lag),]
# #x <- data.matrix(x)
# #x <- x[-(1:lag),]
# 
# Y <-  cbind(y,stvar$fit[,1],stvar$trend[,1])
# color <- c("black","blue","black") 
# legend <- c("data","trend","fitted values")
# cvar <- cbind(stvar$cvar)
# 
# par(mfcol=c(3,1))
# ## t-plot of y, trend & yHat
# matplot(d,Y,xlab="t",type='l',lty=c(1,2,3),lwd=c(1,1,3),col=color,ylab=" ",xaxt="n")
#   #axis.Date(1,at=seq(as.Date("2014/1/1"), as.Date("2016/1/1"),"months"),labels=TRUE)
# legend("bottomleft",legend=legend,lty=c(1,2,3),lwd=c(1,1,3), col=color,cex=.85)
# 
# ## Histogram of y
# hist(stvar$res[,1],main="Residuals",xlab="",ylab="Density",prob=TRUE,breaks="FD") #,ylim=c(0,12)
# 
# ## t-plot of fitted variance
# matplot(d,cvar,xlab="t",type='l',lty=2,lwd=1,ylab="fitted variance",xaxt="n")
#   #axis.Date(1,at=seq(as.Date("2014/1/1"),as.Date("2016/1/1"),"months"),labels=TRUE)
