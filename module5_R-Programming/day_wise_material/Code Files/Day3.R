setwd("C:/Training/Academy/R Course (C-DAC)/Datasets")
cars93 <- read.csv("Cars93.csv")
a <- c(NA, 34, 78, 12)
mean(a, na.rm = T)

library(dplyr)
cars93_1 = rename(cars93, Minimum=Min.Price)

cars93$range = cars93$Max.Price - cars93$Min.Price
# OR
cars93_2 <- mutate(cars93,range = Max.Price - Min.Price )

mean(cars93$Price, na.rm = T)

summarise(cars93, avg_price = mean(Price, na.rm=T),
                  sd_price = sd(Price, na.rm = T))

grp_cars93 <- group_by(cars93, Type)
summarise(grp_cars93, avg_price = mean(Price, na.rm=T),
          sd_price = sd(Price, na.rm = T))

cars93 %>% 
  group_by(Type) %>% 
  summarise(avg_price = mean(Price, na.rm=T),
            sd_price = sd(Price, na.rm = T))

##################### Exercises ########################
survey <- read.csv("survey.csv", stringsAsFactors = T)

MaleNonSmokers <- filter(survey,
                         Sex == "Male" & Smoke == "Never")
PulseGT80 <- survey %>% 
                filter(Pulse > 80) %>% 
                select(Sex, Exer, Smoke, Pulse)

RtHand <- survey %>% 
                mutate(Ratio_Hnd = Wr.Hnd / NW.Hnd) %>% 
                select(Ratio_Hnd, Clap, Age)

DescStats <- survey %>% 
                summarise(avg_age = mean(Age),
                          sd_age = sd(Age))
DescGrp <- survey %>% 
                group_by(Sex) %>% 
                summarise(avg_age = mean(Age),
                          sd_age = sd(Age))                

orders <- read.csv("Orders.csv")
ord_det <- read.csv("Ord_Details.csv")
df <- inner_join(orders, ord_det, by='Order.ID')

items <- read.csv("Items.csv")
combo_df <- inner_join(items, df, by="Item.ID")
# OR

combo_df <- orders %>% 
              inner_join(ord_det,by='Order.ID') %>% 
              inner_join(items, by="Item.ID")

courses <- read.csv("Courses.csv")
sched <- read.csv("CourseSchedule.csv")

all_data <- sched %>% 
              rename(CourseID=CourseCode) %>% 
              inner_join(courses, by="CourseID")

########## tidyr ###################
library(tidyr)

table4a
gather(table4a, `1999`, `2000`, key='year', 
       value='cases')
# or
table4a %>% gather(`1999`, `2000`, key='year',
                   value='cases')
# or
table4a %>% gather(-country, key='year',
                   value='cases')

table4b %>% gather(-country, key="year",
                   value="population")

table4a %>% pivot_longer(-c(country), 
                         names_to = "year",
                         values_to = "cases")

###### Spreading 
table2 %>% spread(key="type", value = "count")
#or
table2 %>% pivot_wider(names_from = "type",
                       values_from = "count")

#### Separate
table3
table3 %>% separate(rate, 
                    into = c('cases','pop'),
                    convert = T)

#### Uniting or concatenating
table5
table5 %>% unite(new_col, century, year, sep="")


stocks <- data.frame(year=c(rep(2015,4),rep(2016,3)),
                     qtr=c(1,2,3,4,2,3,4),
                     return=c(23,76,90,24,58,102,42))
stocks %>% complete(year,qtr)
mu_return <- mean(stocks$return, na.rm = T)
stocks %>% complete(year,qtr,
                    fill=list(return=mu_return))

sales <- data.frame(region=c(rep("North",3),
                             rep("South",4),rep("East",4)),
                    product=c("a","b","c","a","b","c",
                              "d","d","a","b","c"),
                    amt=c(23,54,67,23,10,10,36,78,29,12,33))
sales %>% complete(region,product)

##### Fill
df <- data.frame(region=c("North",NA,NA,"South",NA, NA, NA),
                 sales=c(23,45,67,21,56,90,24))
df %>% fill(region)

stocks %>% 
  complete(year,qtr) %>% 
  fill(return, .direction = "up")

##### Exercises
comb1 <- read.csv("comb1.csv")
lng_combo1 <- comb1 %>% 
                  pivot_longer(-c(District),names_to = "ItemType",
                               values_to = "qty")
comb2 <- read.csv("comb2.csv")
sep_comb2 <- comb2 %>% 
                separate(PatientID,
                         into = c("projectID",
                                  "SiteID",
                                  "PatientNumber"))
