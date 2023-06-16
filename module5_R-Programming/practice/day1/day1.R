print("hello world")
d <- c(3.5, 0.9, 2.4, 12.5)
f <- c("a","b","abc")
d*2
class(d)

#simple interest calculation
p <- c(14000, 10000, 25000, 5000)
n <- 2
r <- 6
print(si <- p*n*r/100)

a <- c(1,2,3)
b <- c(4,5,6)
a*b

#The result of a+c overflows
c <- c(1,2,3,4);class(c)
a+c

length(c)
sum(c)
mean(c)
c[4]

#List data type in R
print(tg <- list(34,"sd",T))
tg*3 #error


s <- c('a','c','c','d')
print(cf <- factor(s))
class(s)
class(cf)

#Missing Values
f <- NA 
is.na(f)

x <- c(31,NA,131,NA)
sum(is.na(x))

#data airquality
data('airquality')
class(airquality)
airquality$Ozone
#counting the na values in Ozone column of airquality dataframe
sum(is.na(airquality$Ozone))

#Not a number
d <- 0
e <- 0
w <- d/e
is.na(w)

#Infinity
d <- 38
e <- 0
w <- d/e
is.infinite(w)
is.finite(w)

#Matrix

m <- matrix(c(1,2,3,4,5),2,3);m
#create a matrix by row
m <- matrix(c(1,3,4,2,4,1),2,3,byrow=T);m

#Binding rows and columns
a <- c(1,2,3,4)
b <- c(3,4,5,6)
cbind(a,b)
rbind(a,b)

a <- c(3,5,6,2,10,12)
b <- c(1,2,2.3,3,5)
cbind(a,b)

#array
h <- array(dim=4);h
h[1] <- 4
h[3] <- 3.2;h

f <- array(dim=c(2,3,4));f

#Data Frame
a <- c(3,5,6,2)
b <- c(2,2.3,3,5)

ds <- data.frame(a,b);ds

#Importing data from csv
data_sets <- file.path("/home","darkstar","Documents","pg-dbda","module5_R-Programming","Day Wise Study Material","Datasets","Gasoline.csv")
b <- read.csv(data_sets);b


