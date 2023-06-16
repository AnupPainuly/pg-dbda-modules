library(XML)
filePath <- "C:/Training/Academy/XML with R/foods.xml"
document <-  xmlTreeParse(filePath, useInternalNodes = TRUE)

rootNode <- xmlRoot(document)
xmlName(rootNode)

names(rootNode)
class(rootNode)

rootNode[1]
rootNode[[1]]
# XML to dataframe
foods <- xmlToDataFrame(document)
##########################
library(xml2)

# food_lst <- as_list(read_xml(filePath))
food <- read_xml(filePath)
xml_root(food)
xml_length(food)
xml_children(food)

xml_child(food, search = 4)

obs <- xml_child(food, search = 4)
des <- xml_child(obs, "description")
xml_contents(des)
library(magrittr)
food  %>% 
  xml_child(search = 4)  %>% 
  xml_child("description")  %>%  
  xml_contents()

food  |> 
  xml_child(search = 3)  |> 
  xml_child("price")  |>  
  xml_contents()

setwd("C:/Training/Academy/R Course (C-DAC)/Datasets")
### Writing to XML
library(MESS)
write.xml(mtcars, "mtcars.xml")

######### JSON ##################
library(jsonlite)
jsonData <- fromJSON("contacts.txt")
class(jsonData)
names(jsonData)

#nested objects
jsonData$phoneNumber
class(jsonData$phoneNumber)
jsonData$phoneNumber$number

##Converting data frames into JSON
mt_JSON <- toJSON(mtcars)
mt_JSON

mt_DS <- fromJSON(mt_JSON)


## Exer
jsonData <- fromJSON("GB_category_id.json")
class(jsonData)
names(jsonData)

jsonData$kind