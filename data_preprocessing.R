# transform the data from the raw html format into a text format


library(readr)
library(xml2)
library(dplyr)
library(rvest)
library(magrittr)

setwd("~/R/DS 10_k report")
 

# read the html file 
html_file <- read_html("Edgar filings_HTML view/Form 10-K/104169/104169_10-K_2020-03-20_0000104169-20-000011.html") 

# select all tables in there
html_tables_all <- html_table(html_file, fill = TRUE)


# 
# # select a single table and store it in a dataframe
# table_one <- html_tables_all[[101]] %>% as.data.frame()
# 
# # print the raw table
# table_one
# 
# 


# write the template function
table_to_text <- function(table) {
  
  # change empty values to NA
  table[table == ""] <- NA 
  
  # delete columns with more than 10% NA
  table_new = table[,!sapply(table, function(x) mean(is.na(x)))> 0.9]
  print(table_new)
  # decide for header row (check if first entry of the row is not NA)
  if (!is.na(table_new[1, 1])){
    header_index <- 0
  } else if(!is.na(table_new)[2, 1]){
    header_index <- 1
  } else if (!is.na(table_new)[3, 1]){
    header_index <- 2
  } else if (!is.na(table_new)[4, 1]){
    header_index <- 3
  } else if (!is.na(table_new)[5, 1]){
    header_index <- 4
  } else if (!is.na(table_new)[6, 1]){
    header_index <- 5
  } else {header_index <- 5}
  

  # header row stays the same in all iterations
  header = table_new[header_index, ]

  start_row = header_index + 1
  
  #current_row = table_new[7, ]
  #sentence = " "

  number_of_sentences <- nrow(table_new) - header_index
  
  # final output variable (appended over all rows and all columns)
  text_file <- " "
  
  
  # loop over all rows that are non header
  for (n in 1:number_of_sentences){
    
    # reset sentence variable for each new row
    sentence <- " "
    current_row = table_new[header_index+n, ] #!!!!!!!!!!!!!!!!!!!
    
    # loop in a certain row over all columns
    for (i in 1:ncol(table_new)){
      text <- paste0(header[i], " ", current_row[i], " ")
      sentence <- paste0(sentence, text)
    }
    
    text_file <- paste0(text_file, " \n ## ", sentence ) # why \t or \n not working?
    
  }


  

  return(text_file)
  
}


# loop over all tables and append the texts into one big file 

final_text <- ""

# go through all tables and get the text
for (index in 1:length(html_tables_all)){
  
  
  current_table <- html_tables_all[[index]] 
  current_table_text <- table_to_text(current_table)
  
  final_text <- paste0(final_text, " \n next table:", index, " ", current_table_text)
  
  
  print(index)
}

final_text

write_file(final_text, "final_table_data.txt")








##### junk 

# change empty values to NA

# table_two <- table_one
# 
# 
# table_two[table_two == ""] <- NA
# table_two = table_two[,!sapply(table_two, function(x) mean(is.na(x)))> 0.9]
# 
# 
# if (!is.na(table_two[1, 1])){
#   variable <- 0
# } else if(!is.na(table_two)[2, 1]){
#   variable <- 1
# } else if (!is.na(table_two)[3, 1]){
#   variable <- 2
# } else if (!is.na(table_two)[4, 1]){
#   variable <- 3
# } else if (!is.na(table_two)[1, 1]){
# 
# } else {variable <- 999}
# 
# variable
# 
# header <- table_two[variable, ]
# header
# 
# 
# start_row = variable + 1
# start_row 
# 
# number_of_sentences <- nrow(table_two) - variable
# number_of_sentences
# 
# text_file <- " "
# sentence <- " "
# current_row <- table_two[variable+1,]
# current_row
# 
# header[1]
# 
# paste0(header[1], " ", current_row[1], " ")
