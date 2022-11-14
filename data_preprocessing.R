# transform the data from the raw html format into a text format


library(readr)
library(xml2)
library(dplyr)
library(rvest)
library(magrittr)


setwd("/Users/felix/Documents/git/Vide")


########### Read in the HTML files and store them in a text format in the data/raw folder ########### 


# import function used in the loop
table_to_text <- function(table) {

  
  # change empty values to NA
  table[table == ""] <- NA 
  # delete columns with more than 10% NA
  table_new = table[,!sapply(table, function(x) mean(is.na(x)))> 0.9]
  
  if (ncol (table_new) == 0){
    empty_string = " "
    return(empty_string)
  } 
  
  
  # decide for header row (check if first entry of the row is not NA)
  if (!is.na(table_new[2, 1])){
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
    
    text_file <- paste0(text_file, " \n ## ", sentence ) 
    
  }
  
  return(text_file)
  
}


# import company list
company_list <- read_csv("data/our_company_list.csv")



# indexes
# for each iteration, select year and company like this: company_list[i, 2] and company_list[i, 3]

#loop_year <- company_list[1, 3]
#loop_company <- company_list[1, 2]
#nrow(company_list)

# read the html file 
html_path <- "data/all_fillings_html/Form 10-K/"
counter <- 0
year_index <- 0


for (idx in 1:nrow(company_list)){
  counter <- counter + 1
  print(paste0(counter, "/", nrow(company_list)))


  # get the yearly index for reports of a given company 
  if (company_list[idx, 3] == 2022){
    year_index <- 3
  } else if (company_list[idx, 3] == 2021){
 
    year_index <- 2
  } else {
    year_index <- 1

  }
  
  
  company <- company_list[idx, 2]
  year <- company_list[idx, 3]
  
  # get the full path of each report
  company_path <- paste0(html_path, company)
  file_name <- list.files(company_path)[year_index]
  complete_path <- paste0(html_path, company, "/", file_name)
  
  
  ##########################################################################################
  
  # we need the file two times
  # once for the normal written text and once to get all tables
  html_file <- read_html(complete_path)
  
  ### the normal written text
  html_normal_text <- html_text(html_file)
  
  ### extract all tables from the report
  html_tables_all <- html_table(html_file, fill = TRUE)
  
  
  html_table_text <- ""
  for (index in 1:length(html_tables_all)){
    

    # selec a different table in each iteration
    current_table <- html_tables_all[[index]] 
    
    current_table_text <- table_to_text(current_table)
    html_table_text <- paste0(html_table_text, " \n next table:", index, " ", current_table_text)
    
    
  }
  
  
  
  complete_report <- paste0(html_normal_text, "The table text starts here: ", html_table_text)
  
  
  file_name <- paste0("data/annual_report_", company, "_", year, ".txt")
  
  write_file(complete_report, file_name)
  
  
}




company <- company_list[1, 2]
company





#####################JUNK####################################
# # loop over annual report files
# for (idx in 1:nrow(company_codes)){
#   print(company_names[idx, 1])
#   print("first loop")
#   # get the correct path for each company
#   company_path <- paste0(html_path, company_codes[idx, 1])
#   
#   
#   for (year in 1:length(relevant_years)){
#     print(relevant_years[year])
#     print("second loop")
#     counter <- counter + 1
#     print(counter)
#     # get the file name of the report
#     file_name <- list.files(company_path)[year]
#     
#     # correct and final path of the report
#     complete_path <- paste0(company_path, "/", file_name)
#     
#     
#     ##########################################################################################
# 
#     # we need the file two times
#     # once for the normal written text and once to get all tables
#     html_file <- read_html(complete_path)
#     
#     ### the normal written text
#     html_normal_text <- html_text(html_file)
#     
#     ### the tables 
#     html_tables_all <- html_table(html_file, fill = TRUE)
#     
#     
#     html_table_text <- ""
#     
#     # go through all tables and get the text
#     for (index in 1:length(html_tables_all)){
#       
#       print("third loop")
#       current_table <- html_tables_all[[index]] 
#       print("after tables")
#       current_table_text <- table_to_text(current_table)
#       print("after function")
#       html_table_text <- paste0(html_table_text, " \n next table:", index, " ", current_table_text)
#       
#       
#       print(index)
#     }
#     
#     
#     
#     complete_report <- paste0(html_normal_text, "The table text starts here: ", html_table_text)
#     
#   
#     file_name <- paste0("data/annual_report_", company_names[idx, 1], "_", relevant_years[year], ".txt")
#     
#     write_file(complete_report, file_name)
#     
#     
#   }
#   
# }




#####################older junk ###################################

# 
# file_name <- paste0("annual_report_", company_names[1, 1], "_", relevant_years[1], ".txt")
# file_name
# 
# 
# testing
# 
# path <- "data/all_fillings_html/Form 10-K/21344/21344_10-K_2020-02-24_0000021344-20-000006.html"
# html_file <- read_html(path)
# 
# 
# ### the tables
# html_tables_all <- html_table(html_file, fill = TRUE)
# 
# 
# table <- html_tables_all[[1]]
# 
# 
# 
# next_one <- table_to_text(html_tables_all[[1]])
# 
# 
# next_one
# 
# 
# 
# # change empty values to NA
# 
# table[table == ""] <- NA 
# 
# print("function 1")
# # delete columns with more than 10% NA
# table_new = table[,!sapply(table, function(x) mean(is.na(x)))> 0.9]
# 
# print("function")
# print(table_new)
# # decide for header row (check if first entry of the row is not NA)
# 
# if (!is.na(table_new[1, 1])){
#   header_index <- 0
# } else if(!is.na(table_new)[2, 1]){
#   header_index <- 1
# } else if (!is.na(table_new)[3, 1]){
#   header_index <- 2
# } else if (!is.na(table_new)[4, 1]){
#   header_index <- 3
# } else if (!is.na(table_new)[5, 1]){
#   header_index <- 4
# } else if (!is.na(table_new)[6, 1]){
#   header_index <- 5
# } else {header_index <- 5}
# 
# 
# 
# 
# header_index
# 
# 


#html_file <- read_html("Edgar filings_HTML view/Form 10-K/104169/104169_10-K_2020-03-20_0000104169-20-000011.html") 


# 
# html_file <- read_html("Edgar filings_HTML view/Form 10-K/21344/21344_10-K_2020-02-24_0000021344-20-000006.html")
# text_file <- html_text(html_file)
# 
# text_file
# 
# write_file(text_file, "new_file_report.txt")













########### Read in the HTML files and transform the tables into strings and add them ########### 















# 
# 
# 
# 
# # select all tables in there
# html_tables_all <- html_table(html_file, fill = TRUE)
# 
# 
# # 
# # # select a single table and store it in a dataframe
# # table_one <- html_tables_all[[101]] %>% as.data.frame()
# # 
# # # print the raw table
# # table_one
# # 
# # 
# 
# 
# # write the template function
# table_to_text <- function(table) {
#   
#   # change empty values to NA
#   table[table == ""] <- NA 
#   
#   # delete columns with more than 10% NA
#   table_new = table[,!sapply(table, function(x) mean(is.na(x)))> 0.9]
#   print(table_new)
#   # decide for header row (check if first entry of the row is not NA)
#   if (!is.na(table_new[1, 1])){
#     header_index <- 0
#   } else if(!is.na(table_new)[2, 1]){
#     header_index <- 1
#   } else if (!is.na(table_new)[3, 1]){
#     header_index <- 2
#   } else if (!is.na(table_new)[4, 1]){
#     header_index <- 3
#   } else if (!is.na(table_new)[5, 1]){
#     header_index <- 4
#   } else if (!is.na(table_new)[6, 1]){
#     header_index <- 5
#   } else {header_index <- 5}
#   
# 
#   # header row stays the same in all iterations
#   header = table_new[header_index, ]
# 
#   start_row = header_index + 1
#   
#   #current_row = table_new[7, ]
#   #sentence = " "
# 
#   number_of_sentences <- nrow(table_new) - header_index
#   
#   # final output variable (appended over all rows and all columns)
#   text_file <- " "
#   
#   
#   # loop over all rows that are non header
#   for (n in 1:number_of_sentences){
#     
#     # reset sentence variable for each new row
#     sentence <- " "
#     current_row = table_new[header_index+n, ] #!!!!!!!!!!!!!!!!!!!
#     
#     # loop in a certain row over all columns
#     for (i in 1:ncol(table_new)){
#       text <- paste0(header[i], " ", current_row[i], " ")
#       sentence <- paste0(sentence, text)
#     }
#     
#     text_file <- paste0(text_file, " \n ## ", sentence ) # why \t or \n not working?
#     
#   }
# 
# 
#   
# 
#   return(text_file)
#   
# }
# 
# 
# # loop over all tables and append the texts into one big file 
# 
# final_text <- ""
# 
# # go through all tables and get the text
# for (index in 1:length(html_tables_all)){
#   
#   
#   current_table <- html_tables_all[[index]] 
#   current_table_text <- table_to_text(current_table)
#   
#   final_text <- paste0(final_text, " \n next table:", index, " ", current_table_text)
#   
#   
#   print(index)
# }
# 
# final_text
# 
# write_file(final_text, "final_table_data.txt")
# 
# 
# 
# 
# 
# 
# 
# 
# ##### oldest junk ############# 
# 
# # change empty values to NA
# 
# # table_two <- table_one
# # 
# # 
# # table_two[table_two == ""] <- NA
# # table_two = table_two[,!sapply(table_two, function(x) mean(is.na(x)))> 0.9]
# # 
# # 
# # if (!is.na(table_two[1, 1])){
# #   variable <- 0
# # } else if(!is.na(table_two)[2, 1]){
# #   variable <- 1
# # } else if (!is.na(table_two)[3, 1]){
# #   variable <- 2
# # } else if (!is.na(table_two)[4, 1]){
# #   variable <- 3
# # } else if (!is.na(table_two)[1, 1]){
# # 
# # } else {variable <- 999}
# # 
# # variable
# # 
# # header <- table_two[variable, ]
# # header
# # 
# # 
# # start_row = variable + 1
# # start_row 
# # 
# # number_of_sentences <- nrow(table_two) - variable
# # number_of_sentences
# # 
# # text_file <- " "
# # sentence <- " "
# # current_row <- table_two[variable+1,]
# # current_row
# # 
# # header[1]
# # 
# # paste0(header[1], " ", current_row[1], " ")
