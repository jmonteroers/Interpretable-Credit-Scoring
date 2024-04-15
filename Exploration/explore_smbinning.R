# In this script, I replicate the results for the bins from smbinning 
# using the underlying partykit::ctree library

library(smbinning)

# Load library and its dataset
library(smbinning) # Load package and its data
pop=smbsimdf1 # Set population
train=subset(pop,rnd<=0.7) # Training sample
# Binning application for a numeric variable
result=smbinning(df=train,y="fgood",x="dep") # Run and save result
# Generate a dataset with binned characteristic
pop=smbinning.gen(pop,result,"g1dep")
# Check new field counts
table(pop$g1dep)

# obtained bands
print(result$bands)

# replicate bands using partykit::ctree
# borrowing code from github mirror
# https://github.com/cran/smbinning/blob/master/R/smbinning.R
library(partykit)
N <- nrow(train)
ctree_settings <- ctree_control(minbucket = ceiling(round(p*N)))
ctree_fit <- partykit::ctree(
  fgood ~ dep, 
  data = train, 
  na.action = na.exclude, 
  control = ctree_settings)

# extract bands
cutvct=data.frame(matrix(ncol=0,nrow=0)) # Shell
n=length(ctree_fit) # Number of nodes
for (i in 1:n) {
  cutvct=rbind(cutvct,ctree_fit[i]$node$split$breaks)
}