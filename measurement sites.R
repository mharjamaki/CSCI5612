library(viridis)
library(arules)
library(TSP)
library(data.table)
#library(ggplot2)
#library(Matrix)
library(tcltk)
library(dplyr)
library(devtools)
library(purrr)
library(tidyr)
library(arulesViz)
library(RColorBrewer)


# set working directory 
setwd("C:/Users/mikyl/OneDrive/Documents/CSCI/AssociationRuleMining")

sites <- read.transactions("measurement sites.csv",
                                rm.duplicates = FALSE, 
                                format = "basket",  
                                sep=",",  
                                cols=1) # The data set has a transaction ID
inspect(sites)


# apply apriori algorithm to get rules
siterules = arules::apriori(sites, parameter = list(support=.15, 
                                                       confidence=.40, minlen=2))
inspect(siterules)


# Plot item frequency 
arules::itemFrequencyPlot(sites, topN = 14,
                          col = brewer.pal(8, 'Pastel2'),
                          main = 'Frequency of Each Measurement',
                          type = "absolute", # or relative
                          ylab = "Item Frequency")


# Sort rules by a measure such as conf, sup, or lift
SortedRules <- sort(siterules, by="lift", decreasing=FALSE)
inspect(SortedRules[1:15])
(summary(SortedRules))

# Selecting or targeting specific rules  RHS
# PMRules <- apriori(data=sites,parameter = list(supp=.001, conf=.01, minlen=2),
#                      appearance = list(default="lhs", rhs="PM2.5"),
#                      control=list(verbose=FALSE))
# PMRules <- sort(PMRules, decreasing=TRUE, by="confidence")
# inspect(PMRules[1:4])
# 
# # Selecting rules with LHS specified
# LPMRules <- apriori(data=sites,parameter = list(supp=.001, conf=.01, minlen=2),
#                       appearance = list(default="rhs", lhs="PM10"),
#                       control=list(verbose=FALSE))
# LPMRules <- sort(LPMRules, decreasing=TRUE, by="support")
# inspect(LPMRules[1:4])

## Visualize
## tcltk

subrules <- head(sort(SortedRules, by="lift"),20)
plot(subrules)

plot(subrules, method="graph", engine="interactive")

