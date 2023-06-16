A <- c(23, 56, 89, 100, 34)
barplot(A)
setwd("C:/Training/Academy/R Course (C-DAC)/Datasets")
cars93 <- read.csv("Cars93.csv")
table(cars93$Type)
table(cars93$Price.Category)
barplot(table(cars93$Price.Category))
barplot(table(cars93$Type),
        main = "Types of Cars",
        col="mistyrose2")
colors()

hist(cars93$Price)

plot(cars93$Price, cars93$MPG.city,
     xlab = "Price",ylab = "Milage in City",
     main = "Price Vs MPG")

gasoline <- read.csv("Gasoline.csv")

boxplot(cars93$Price)
boxplot(cars93$Price ~ cars93$Type)

library(ggplot2)
ggplot(data = cars93, aes(x=Price,y=MPG.city))+
  geom_point()+
  labs(x="Price",y="Milage in City")

ggplot(data = cars93, 
       aes(x=Price,y=MPG.city,color=Type))+
  geom_point()+
  labs(x="Price",y="Milage in City")

ggplot(data = cars93, 
       aes(x=Price,y=MPG.city,color=AirBags))+
  geom_point()+
  labs(x="Price",y="Milage in City")

ggplot(data = cars93, 
       aes(x=Price,y=MPG.city,color=AirBags))+
  geom_point()+facet_grid(~AirBags)
  labs(x="Price",y="Milage in City")

ggplot(data = cars93, 
         aes(x=Price,y=MPG.city))+
    geom_point()+geom_smooth(method = "lm", se=F)+
    labs(x="Price",y="Milage in City") 

ggplot(data = cars93, 
       aes(x=Price,y=MPG.city,size=EngineSize))+
  geom_point(color='blue', alpha=0.4)


######## Boxplot #############
ggplot(data = cars93,aes(y=Price))+
  geom_boxplot()

ggplot(data = cars93,aes(x=Type,y=Price))+
  geom_boxplot()

ggplot(data = cars93,
       aes(x=Type,y=Price, color=Type))+
  geom_boxplot()

ggplot(data = cars93,
       aes(x=Type,y=Price, fill=Type))+
  geom_boxplot()

########## Histogram ##################
ggplot(data = cars93, 
       aes(x=Price))+
  geom_histogram(bins = 20, fill="pink",
                 color="red")

ggplot(data = cars93, 
       aes(x=Price))+
  geom_histogram(binwidth = 10, fill="pink",
                 color="red")

######### Bar Plot ##############
ggplot(data = cars93, aes(x=Type)) + 
  geom_bar(fill="violetred2")

ggplot(data = cars93, aes(x=AirBags)) + 
  geom_bar(fill="steelblue4")

table(cars93$AirBags, cars93$Type)

ggplot(data = cars93, aes(x=Type, fill=AirBags)) + 
  geom_bar(position = 'dodge')

library(dplyr)
summ_cars <- cars93 %>% 
  group_by(AirBags) %>% 
  summarise(avg_price=mean(Price, na.rm = T))

ggplot(data = summ_cars,
       aes(x=AirBags,y=avg_price,fill=AirBags))+
  geom_bar(stat = 'identity')

### Theme

ggplot(data = summ_cars,
       aes(x=AirBags,y=avg_price,fill=AirBags))+
  geom_bar(stat = 'identity')+theme_bw()

