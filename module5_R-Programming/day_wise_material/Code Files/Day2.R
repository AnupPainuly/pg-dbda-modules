setwd("C:/Training/Academy/R Course (C-DAC)/Datasets")
cars93 <- read.csv("Cars93.csv")
dim(cars93)
str(cars93)

autocoll <- read.csv("AutoCollision.csv", 
                     stringsAsFactors = TRUE)
str(autocoll)
# autocoll$Age <- factor(autocoll$Age)

bollyw <- read.csv("Bollywood_2015_2.csv", header = F)
colnames(bollyw) <- c("Movie","BO","Budget","Verdict")

diamonds <- read.csv("Diamonds.csv", sep=";", dec = ",")
# or
diamonds <- read.csv2("Diamonds.csv")

mem <- read.csv("members.txt",header = T,
                sep=" ",skip = 1)

########### Reading Excel ##############
library(readxl)
brupt <- read_excel("bankruptcy.xlsx",sheet = "data")

qual1 <- read_excel("quality.xlsx", sheet = 'quality',
                    range = "A1:D6")
qual2 <- read_excel("quality.xlsx", sheet = 'quality',
                    range = "H1:J16")
write.csv(qual2, "qual2.csv", row.names = F)

library(writexl)
write_xlsx(qual2, "qual2.xlsx")


### Subsetting the data frame
autocoll[5,]
autocoll[1:5,]
autocoll[20:25,]
autocoll[,3:4]

ss <- subset(autocoll, Vehicle_Use == "Business")
ss2 <- subset(autocoll, Claim_Count > 400)
ss3 <- subset(autocoll, Age=="A" & Severity>500)
ss4 <- subset(autocoll, Age=="A" | Severity>500)

ss_cars <- subset(cars93, 
                  Type=='Small' & Price > 10,
                  select = c(Manufacturer,Model,Price,Origin))

ss_cars <- subset(cars93, 
                  select = c(Manufacturer,Model,Price,Origin))

######################Exercises##########################
orders <- read.csv("Orders.csv", stringsAsFactors = T)
ss_ords <- subset(orders, Payment.Terms=="Online")

data("mtcars")
write.csv(mtcars, "mtcars.csv")

ss_mtcars <- mtcars[c(2,18,30,12),]

class(mtcars)

library(tidyverse)
tbl_mtcars <- as_tibble(mtcars)
class(tbl_mtcars)

s_autoc <- arrange(autocoll, Claim_Count)
s_autoc <- arrange(autocoll, desc(Claim_Count))

## magrittr
s_autoc <- autocoll %>% arrange(Claim_Count)

s_autoc <- arrange(autocoll, Vehicle_Use, Severity)
s_autoc <- arrange(autocoll, Vehicle_Use, desc(Severity))

### Selecting
ss_cars <- select(cars93, 1:5)
ss_cars <- select(cars93, Model:Price)
ss_cars <- select(cars93, contains("en"))
