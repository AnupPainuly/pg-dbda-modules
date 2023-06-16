#setting up the working environment
setwd("/home/darkstar/Documents/pg-dbda/module5_R-Programming/assignments/R_Assignment")

#loading the libraries
library(dplyr)
library(ggplot2)

#loading the data set
HR <- read.csv("HR_Attrition.csv")
str(HR)

#Q1.Filter the data with Marital Status as Single and Job Role as Research Director
subset_HR <- subset(HR,MaritalStatus == 'Single' & JobRole == 'Research Director')

#Q2.Generate a frequency table for all the Job Roles
table_freq <- table(HR$JobRole)
freq_df <- as.data.frame(table_freq)

#Q3.Calculate the mean Hourly Rate for every Gender and plot the means with ggplot functions.
avg_hourlyRate <- HR %>% 
  group_by(Gender) %>% 
    summarise(avg_hourlyRate = mean(HourlyRate))

ggplot(avg_hourlyRate,aes(x=Gender,y=avg_hourlyRate,fill=Gender))+
  scale_fill_brewer(palette="Green")+
    geom_bar(stat="identity",color="black")+
      theme_classic()+ggtitle("Bar Plot\nGenderwise mean hourly rate")+
        labs(y="Mean Hourly Rate")


#Q4.Generate Histogram for Monthly Income
ggplot(data=HR, aes(x=MonthlyIncome))+
      geom_histogram(bins=5, fill="black",color="green")+
        labs(x="Monthly Income", y="frequency")+
          ggtitle("Histogram\nFrequency of Monthly Income")+
            theme_classic()

#Q5.Generate the scatter plot with X-axis as Monthly Rate and 
#Y-axis as Monthly Income with color as Department

ggplot(data=HR,aes(x=MonthlyRate,y=MonthlyIncome,col=Department))+
    geom_point()+
      geom_smooth(method=lm,se=F)+
        scale_colour_brewer(palette="Set1")+
          theme_classic()+ggtitle("Scatter Plot\nMonthly Income vs Monthly Rate")
