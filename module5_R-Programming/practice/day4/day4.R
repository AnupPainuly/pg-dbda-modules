#install.packages('tidyverse')
setwd("/home/darkstar/Documents/pg-dbda/module5_R-Programming/Practice/day1/Datasets")
cars93 <- read.csv("Cars93.csv")
library(ggplot2)

ggplot(data=cars93, aes(x=Price, y=MPG.city,col=Type))+
  geom_point()+labs(x="Price of the vehicle",y="Miles per gallon (city)")

#smoothing the regression line with method as linear model and setting confidence band distribution to se=false
ggplot(data=cars93, aes(x=Price, y=MPG.city,col=AirBags))+
geom_point()+geom_smooth(method = "lm",se=F)+
labs(x="Price of the vehicle",y="Miles per gallon (city)")

ggplot(data=cars93, aes(x=Price, y=MPG.city,col=AirBags))+
geom_point()+facet_grid(~AirBags)+geom_smooth(method = "lm",se=F)+
labs(x="Price of the vehicle",y="Miles per gallon (city)")+theme(panel.grid.major = element_blank())

