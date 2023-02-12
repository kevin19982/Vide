# further debugging: table 126
# change NA to ""



#Preprocessing goal: transform the data from the raw html format into the right JSON format 

library(readr)
library(xml2)
library(dplyr)
library(rvest)
library(magrittr) # %>%
library(compare) # compare
library(rjson) # load the file
library(jsonlite) # write the file
library(stringr) # str_detect 

setwd("/Users/felix/Documents/git/Vide/new/recent_12_02/Vide")

# import a list with all available data
company_list <- read_csv("data/our_company_list.csv")


### two auxiliary functions to transform tables into text

# transforms a tables row into raw text
# for each column, there will be a sentence
row_to_text <- function(table, header_index, iter, indices_subheaders){
  
  # the header is the same in all rows
  header <- table[header_index, ]
  
  #iterate through rows  
  current_row <- table[header_index+iter, ]
  
  # define current sub-header (first check if there is one)
  if (any(indices_subheaders == ncol(table)-1)){
    
    # then go and find the last sub-header
    for (i in 1:header_index+iter){
      if (indices_subheaders[i] == ncol(table)-1){
        subheader <- table[i, 1]
      }
    }
  }
  
  # skip the row if current_row = sub-header row
  if (indices_subheaders[header_index+iter] == ncol(table)-1){
    res <- ""
    return(res)
  }
  
  res <- ""
 
  if (ncol(table) > 1){
    for (col in 2:ncol(table)){
      
      # change (123) to ( 123 )
      if (grepl("\\(", current_row[col])){
        
        string <- as.character(current_row[col])
        string <- sub("\\(", "\\( ", string)
        string <- sub("\\)", " \\)", string)
        current_row[col] <- string}
      
      if (is.na(current_row[col])){
        res <- ""
        return(res)
      }
      
      
      if (exists("subheader")){
        #res <- paste0(res, "In ", subheader, " the ", current_row[1], " of ", header[col], " is ", current_row[col], " . \n")
        res <- c(res, paste0("In ", subheader, " the ", current_row[1], " of ", header[col], " is ", current_row[col], " . "))
      } else{
        #res <- paste0(res, "the ", current_row[1], " of ", header[col], " is ", current_row[col], " .  \n")
        res <- c(res, paste0("the ", current_row[1], " of ", header[col], " is ", current_row[col], " . "))
        
        
        
        }
      
      
      
    } 
    
  } else{res <- ""}
  
  return(res)
}

# calls row_to_text and transforms a table into raw text
# for each cell in the table, there will be a sentence
table_to_text <- function(table) {

  
  ### first, the table needs to be cleaned
  
  # delete identical columns 
  if (compare(table[,1], table[,2])[2] == TRUE){
    table <- table[,-1]
  }
  if (compare(table[,1], table[,2])[2] == TRUE){
    table <- table[,-1]
  }
  
  
  
  # change empty values to NA
  table[table == ""] <- NA 
  table[table == "$"] <- NA 
  # delete columns with more than 10% NA
  table_new = table[,!sapply(table, function(x) mean(is.na(x)))> 0.5]
  
  # counts "NA" in each row and selects subheaders if = ncol()-1
  indices_subheaders <- apply(table_new, 1, function(x) sum(is.na(x)))
  
  # delete empty tables
  if (ncol (table_new) == 0){
    empty_string = ""
    return(empty_string)
  } 
  
  
  # change all NA's to empty values
  ################################
  #table_new[is.na(table_new)] <- ""
  
  
  # inside the tables there are sometimes sub-headers => this information needs to be 
  # incorporated in each sentence
  # first we need to find them and then use the info in each subsequent row

  
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
  number_of_sentences <- nrow(table_new) - header_index
  
  #print(dim(table_new))
  #print(indices_subheaders)
  
  table_text <- ""
  for (iter in 1:number_of_sentences){
    row_text <- row_to_text(table = table_new, header_index = header_index, iter = iter, indices_subheaders = indices_subheaders)
    #table_text <- paste0(table_text, row_text) #, "next_row")
    table_text <- c(table_text, row_text)
    
  }
  
  table_text <- paste0(table_text) # , "#end#")
  
  
  #print(table_text)
  return(table_text)
  
}

# calls table_to_text and transforms a html file into raw text
# index = number of iteration 
file_to_text <- function(idx){
  
  # get company information
  company_code <- company_list$cik[idx]
  date <- company_list$date.filed[idx]
  accession <- company_list$accession.number[idx]
  form <- company_list$form.type[idx]
  
  # create the path
  root_path <- "data/all_filings_html/Form 10-K/"
  file_name <- paste0(company_code, "_", form, "_", date, "_", accession, ".html")
  complete_path <- paste0(root_path, company_code, "/", file_name)
 
   # error checking
  #print(complete_path)
  #print(file.exists(complete_path))
  
  
  # load entire file into memory as string
  html_file <- read_file(complete_path) 

  
  
  ################################### STEP 1 ###################################
  ### extract all table and transform them into raw text
  # load all the tables 
  html <- read_html(complete_path) #encoding = "unicode")
  tables <- html_table(html, fill = TRUE)
  
  
  all_tables <- ""
  
  # iterate through all tables and transform to text
  for (i in 1:length(tables)){ #1:length(tables)
    current_table <- tables[[i]]
    current_table_text <- table_to_text(current_table)
    
    #placeholder <- paste0("\n #", i, " \n")
    #print(paste0("progress: ", i, "/" , length(tables)))
    #all_tables <- paste0(all_tables, placeholder, current_table_text)
    #all_tables <- paste0(all_tables, " ", current_table_text)
    all_tables <- c(all_tables, current_table_text)
  }
  #write(all_tables,"all_tables_text_output.txt")
  # return(all_tables)
  #return(all_tables)
  
  
  ################################### STEP 2 ###################################
  ### delete all non transformed data from the raw text and combine table text + normal text
  
  html_file <- read_file(complete_path) 
  #write(html_file, "html_raw.txt")
  
  # split before and after each table to filter them out
  text_splitted <- strsplit(html_file, "<table|</table>")
  
  
  table_text <- ""
  # first string is not of any importance
  # transform the html code into real text and skip all tables
  for (i in 2:length(text_splitted[[1]])){
    if (text_splitted[[1]][i] == ""){
    } 
    else if (text_splitted[[1]][i] == "</div></div>"){
    } 
    else if (grepl("cellpadding", text_splitted[[1]][i]) | grepl("border-collapse:collapse;", text_splitted[[1]][i])){
    } 
    else {
      
      # \\s+ detects all spaces in the text
      text <- read_html(text_splitted[[1]][i]) %>% html_text2() %>% str_replace_all("\\s+", " ") 
      table_text <- paste0(table_text, text)
    }
  }
  
  dot_space <- strsplit(table_text, "\\. ")
  
  
  # just to visualize the results
  dot_combined <- ""
  for (i in 1:length(dot_space[[1]])){
    dot <- paste0(dot_space[[1]][i], " . ")
    dot_combined <- c(dot_combined, dot)
    
  }
  #write(dot_combined, "all_text_text_output.txt")
  
  ################################### STEP 3 ###################################
  ### combine the output as a txt file
 
  # all_tables
  # dot_combined
  full_text <- c(all_tables, dot_combined)
  
  # as string not list 
  full_text_string <- ""
  for (sentence in 2:length(full_text)){ # 1:length(tables)
    if (full_text[[sentence]] == ""){} 
    else if (full_text[[sentence]] == full_text[[sentence-1]]){} 
    else {
      #print(full_text[[sentence]])
      
      #model_template[[1]][["pre_text"]][index] <- tables[[sentence]]
      
      full_text_string <- paste0(full_text_string, full_text[[sentence]])
      #index <- index + 1
    }
  }
  
  
  
  # write the file
  company_date <- as.character(company_list$date.filed[idx])
  company_year <- strsplit(company_date, "-")[[1]][1]
  company_name <- company_list$company.name[idx]
  
  file_name <- paste0(company_name, "_",company_year, ".txt")
  path <- paste0("data/", file_name)
  write(full_text_string, path)
  print(path)
  
  return(full_text_string)
}


# goes through all 
for (report in 1:nrow(company_list)){
  try({
  print(paste0("Report: ", report))
  full_text <- file_to_text(report)
  })
  
}


# 
# 
# 
# company_date <- as.character(company_list$date.filed[1])
# company_year <- strsplit(company_date, "-")[[1]][1]
# company_name <- company_list$company.name[1]
# 
#     
# 
# company_date
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 














#write(full_text, "data/full_text.txt")
# now get the correct form (plan for tomorrow => iteration through all files)
model_template <- rjson::fromJSON(file = "data/model_input_template.json")


index <- 1
full_text_string <- ""


for (sentence in 2:length(full_text)){ # 1:length(tables)
  if (full_text[[sentence]] == ""){} 
  else if (full_text[[sentence]] == full_text[[sentence-1]]){} 
  else {
    print(full_text[[sentence]])
    
    #model_template[[1]][["pre_text"]][index] <- tables[[sentence]]
    
    full_text_string <- paste0(full_text_string, full_text[[sentence]])
    #index <- index + 1
  }
}

write(full_text_string, "data/full_text_string.txt")




write_json(model_template, "data/entire_report_2.json")











#--------------------------------- debugging and cleaning help



# replica of the function from above
if (compare(table_69[,1], table_69[,2])[2] == TRUE){
  table_69 <- table_69[,-1]
  print("reduced")

}


# change empty values to NA
table_69[table_69 == ""] <- NA

table_69[table_69 == "$"] <- NA

# delete columns with more than 10% NA
table_69 = table_69[,!sapply(table_69, function(x) mean(is.na(x)))> 0.5]

# nchar(string)

# indices_subheaders <- apply(table_69, 1, function(x) sum(is.na(x)))
# if (any(indices_subheaders == ncol(table_69)-1)){
#   print("yes")
# } else {print("no")}
# 
# 
# print(dim(table_new))
# print(indices_subheaders)

################################################################################################################
################################################################################################################

# from 1:ncol(company_list)
#tables <- file_to_text(2)


# 
# # now get the correct form (plan for tomorrow => iteration through all files)
# 
# model_template <- rjson::fromJSON(file = "data/model_input_template.json")
# 
# 
# index <- 1
# 
# for (sentence in 2:length(tables)){ # 1:length(tables)
#   if (tables[[sentence]] == ""){} 
#   else if (tables[[sentence]] == tables[[sentence-1]]){} 
#   else {
#     print(tables[[sentence]])
#     model_template[[1]][["pre_text"]][index] <- tables[[sentence]]
#     index <- index + 1
#     }
# }
#   
# write_json(model_template, "data/entire_report_2.json")






# model_template <- rjson::fromJSON(file = "data/model_input_template.json")
# model_template[[1]][["pre_text"]][1] <- "test"
# model_template[[1]][["pre_text"]][2] <- "test_2"
# model_template[[1]][["pre_text"]][3] <- "test_3"
# model_template[[1]][["pre_text"]][4] <- "test_4"
# model_template[[1]][["pre_text"]][5] <- "test_5"
# model_template[[1]][["pre_text"]][6] <- "test_6"
# model_template[[1]][["pre_text"]][7] <- "test_7"
# model_template[[1]][["pre_text"]][8] <- "test_8"
# model_template[[1]][["pre_text"]][9] <- "test_9"
# model_template[[1]][["pre_text"]][10] <- "test_10"
# model_template[[1]][["pre_text"]][11] <- "test_11"
# write_json(model_template, "data/json_format_testing.json")







################################################################################################################
################################################################################################################
# get the text (first without the function and everything manually)
# 
# complete_path <- "data/all_filings_html/Form 10-K/1018724/1018724_10-K_2021-02-03_0001018724-21-000004.html"
# # load entire file into memory as string
# 
# html_file <- read_file(complete_path) 
# #write(html_file, "html_raw.txt")
# 
# 
# # split before and after each table to filter them out
# text_splitted <- strsplit(html_file, "<table|</table>")
# 
# 
# table_text <- ""
# # first string is not of any importance
# # transform the html code into real text and skip all tables
# for (i in 2:length(text_splitted[[1]])){
#   if (text_splitted[[1]][i] == ""){
#   } 
#   else if (text_splitted[[1]][i] == "</div></div>"){
#   } 
#   else if (grepl("cellpadding", text_splitted[[1]][i]) | grepl("border-collapse:collapse;", text_splitted[[1]][i])){
#   } 
#   else {
#     
#     # \\s+ detects all spaces in the text
#     text <- read_html(text_splitted[[1]][i]) %>% html_text2() %>% str_replace_all("\\s+", " ") 
#     table_text <- paste0(table_text, text)
#   }
# }
# 
# dot_space <- strsplit(table_text, "\\. ")
# 
# 
# # just to visualize the results
# dot_combined <- ""
# for (i in 1:length(dot_space[[1]])){
#   dot <- paste0(dot_space[[1]][i], ". \n")
#   dot_combined <- paste0(dot_combined, dot)
#   
# }
# 
# write(dot_combined, "dot_combined.txt")
# 
# 


#write(table_text, "text_text_text.txt")

# now, split the text into sentences
####################################get to work ############


#dot <- paste0("\n", strsplit(table_text, "\\."))

#write(dot_space, "dot_space.txt")

#write(dot, "dot.txt")


# test_a <- c("a", "b", "c")
# test_1 <- c("1", "2", "3")
# 
# test_combined <- c(test_a, test_1)
# 
# 
# 
# length(splitted_again[[1]])
# 
# splitted_again[[1]][3]





# 
# sentenced <- ""
# 
# for (idx in 1:length(table_text)){
#   if (!grepl(".", table_text[idx])){
#   } else {
#     #to_be_splitted <- sub(".", ".. ", table_text[idx])
#     split <- strsplit(table_text[idx], ". ")
#     sentenced <- c(sentenced, split)
#     
#   }
  
  
  
#}

# 
# sentenced[2]
# 
# table_text[2]
# 
# 
# write(table_text, "text_text_01.txt")
# 
# 
# table_text[2]


# 
#     else {
#       table_text <- ""
#       try({
#         test <- read_html(text_splitted[[1]][i]) %>% html_text2()
#         empty_list <- c(list_with_tables, table_text)
#       })
#     }
#   }


# 
# # split string into list at the beginning and the end of every table 
# text_splitted <- strsplit(html_file, "<table|</table>")
# 
# # empty lists for later
# list_with_tables <- list()
# 
# # first, iterate through all indices in the list and transform them into real text
# for (i in 1:length(text_splitted[[1]])){
# 
#   
#   if (text_splitted[[1]][i] == ""){
#     
#   } else if (text_splitted[[1]][i] == "</div></div>"){
#     
#   } else if (grepl("cellpadding", empty_list[i]) | grepl("border-collapse:collapse;", empty_list[i])){
#     
#   }
#   
# 
#   
#   else {
#     table_text <- ""
#     try({
#       test <- read_html(text_splitted[[1]][i]) %>% html_text2()
#       empty_list <- c(list_with_tables, table_text)
#     })
#   }
# }
# 
# # this iteration recognizes table entries and transforms them into raw text
# for (i in 1:length(empty_list)){
#   
#   # if empty_list[i] contains "cellpadding" => if list[i] is former table
#   if (grepl("cellpadding", empty_list[i]) | grepl("border-collapse:collapse;", empty_list[i])){
#     
#     
#     # save the index of table
#     index_for_tables <- c(index_for_tables, i)
#     
#     # select table
#     current_table <- tables[[length(index_for_tables)]]
#     table_text <- table_to_text(current_table)
#     
#     
#     # Table number: ### 
#     empty_list[i] <- paste0("T: ", length(index_for_tables), " \n ", table_text)
#     
#     #print("True")
#     
#   } else {
#     #print("False")
#   }
#   
# }
# 






################################################################################################################
################################################################################################################



# root_path <- "data/all_filings_html/Form 10-K/"
# counter <- 0 
# # iterate through all companies 
# for (idx in 1:nrow(company_list)){
#   
#   # print progres
#   #counter <- counter + 1
#   #print(paste0(counter, "/", nrow(company_list)))
#   
#   # company infos
#   company_code <- company_list$cik[idx]
#   date <- company_list$date.filed[idx]
#   accession <- company_list$accession.number[idx]
#   form <- company_list$form.type[idx]
#   
#   # create the path
#   file_name <- paste0(company_code, "_", form, "_", date, "_", accession, ".html")
#   complete_path <- paste0(root_path, company_code, "/", file_name)
#   print(complete_path)
#   print(file.exists(complete_path))
#   
#   # load file into memory
#   html_file <- read_file(complete_path) # as string
#   
#   # just for the tables
#   html <- read_html(complete_path)
#   tables <- html_table(html, fill = TRUE)
# 
#   # split string into list at the beginning and the end of every table 
#   text_splitted <- strsplit(html_file, "<table|</table>")
#   
#   # empty lists for later
#   empty_list <- list()
#   index_for_tables = list()
#   
#   # for verification
#   counter_if_1 <- 0
#   counter_if_2 <- 0
#   
#   # first, iterate through all indices in the list and transform them into real text
#   for (i in 1:length(text_splitted[[1]])){
#     #print(i)
#     
#     if (text_splitted[[1]][i] == ""){
#       counter_if_1 <- counter_if_1 + 1
#       #empty_list <- c(empty_list, text_splitted[[1]][i])
#       
#     } else if (text_splitted[[1]][i] == "</div></div>"){ 
#       
#       counter_if_2 <- counter_if_2 + 1
#       #empty_list <- c(empty_list, text_splitted[[1]][i])
#     }  
#     else {
#       test <- ""
#       try({
#         test <- read_html(text_splitted[[1]][i]) %>% html_text2()
#         #print(test)
#         empty_list <- c(empty_list, test)
#       })
#     }
#   }
#   
#   
#   # this iteration recognizes table entries and transforms them into raw text
#   for (i in 1:length(empty_list)){
#     
#     # if empty_list[i] contains "cellpadding" => if list[i] is former table
#     if (grepl("cellpadding", empty_list[i]) | grepl("border-collapse:collapse;", empty_list[i])){
#       
#       
#       # save the index of table
#       index_for_tables <- c(index_for_tables, i)
#       
#       # select table
#       current_table <- tables[[length(index_for_tables)]]
#       table_text <- table_to_text(current_table)
#       
#       
#       # Table number: ### 
#       empty_list[i] <- paste0("T: ", length(index_for_tables), " \n ", table_text)
#       
#       #print("True")
#       
#     } else {
#       #print("False")
#       }
#     
#   }
#   
#   # define finale variable
#   final_string <- ""
#   
#   
#   # finally we add all list entries together
#   for (i in 2:length(empty_list)){
#     #final_string <- paste0(final_string, "\n", empty_list[i])
#     final_string <- paste0(final_string, "\n #", i, ": ", empty_list[i])
#   }
#   
#   
#   # write the file in the correct file and with the correct name
#   new_file_name <- paste0("data/annual_report_", company_code, "_", date, ".txt")
#   
#   # save the file 
#   write(final_string, new_file_name)
#   
# }




####################################### junk 


# # check for existence
# if (exists("subheader")){
#   res <- paste0("In ", subheader, " ")
# } else {res <- ""}
# 
# 
# if (iter == 1){
#   res <- paste0(res,  header[1], " ")
# }
# if (ncol(table) > 1){
#   for (col in 2:ncol(table)){
#     res <- paste0(res, "the ", current_row[1], " of ", header[col], " is ", current_row[col], " . \n ")
#       
#   } }
# else {res <- ""}
# 




# 
# # if sub header => don't make a sentence in this row
# if(indices_subheaders[header_index+iter] == ncol(table)-1){
#   res <- ""
#   return(res)
# }
# 
# if (iter == 1){
#   
#   res <- paste0(res,  header[1], " ")}
# 
# # check if there is a sub-header
# if (any(indices_subheaders == ncol(table)-1)){
#   
#   # then go and find the last sub-header
#   for (i in 1:header_index+iter){
#     if (indices_subheaders[i] == ncol(table)-1){
#       subheader <- table[i, 1]
#     }
#   }
#   
#   # print res with sub-header
#   res <- paste0("In ", subheader, " ", res, "the ", current_row[1], " of ", header[col], " is ", current_row[col], " . \n ")
#   
# 
#   
# } else {
#   # print res without sub-header
#   res <- paste0(res, "the ", current_row[1], " of ", header[col], " is ", current_row[col], " . \n ")
#     }
# 



# # check for existence
# if (exists("subheader")){
#   res <- paste0("In ", subheader, " ")
# } else {res <- ""}
# 
# 
# if (iter == 1){
#   res <- paste0(res,  header[1], " ")
# }
# if (ncol(table) > 1){
#   for (col in 2:ncol(table)){
#     res <- paste0(res, "the ", current_row[1], " of ", header[col], " is ", current_row[col], " . \n ")
#       
#   } }
# else {res <- ""}
# 
#return(res)
#}



# 
# # split string into list at the beginning and the end of every table 
# text_splitted <- strsplit(html_file, "<table|</table>")
# 
# # empty lists for later
# list_with_tables <- list()
# 
# # first, iterate through all indices in the list and transform them into real text
# for (i in 1:length(text_splitted[[1]])){
# 
#   
#   if (text_splitted[[1]][i] == ""){
#     
#   } else if (text_splitted[[1]][i] == "</div></div>"){
#     
#   } else if (grepl("cellpadding", empty_list[i]) | grepl("border-collapse:collapse;", empty_list[i])){
#     
#   }
#   
# 
#   
#   else {
#     table_text <- ""
#     try({
#       test <- read_html(text_splitted[[1]][i]) %>% html_text2()
#       empty_list <- c(list_with_tables, table_text)
#     })
#   }
# }
# 
# # this iteration recognizes table entries and transforms them into raw text
# for (i in 1:length(empty_list)){
#   
#   # if empty_list[i] contains "cellpadding" => if list[i] is former table
#   if (grepl("cellpadding", empty_list[i]) | grepl("border-collapse:collapse;", empty_list[i])){
#     
#     
#     # save the index of table
#     index_for_tables <- c(index_for_tables, i)
#     
#     # select table
#     current_table <- tables[[length(index_for_tables)]]
#     table_text <- table_to_text(current_table)
#     
#     
#     # Table number: ### 
#     empty_list[i] <- paste0("T: ", length(index_for_tables), " \n ", table_text)
#     
#     #print("True")
#     
#   } else {
#     #print("False")
#   }
#   
# }
# 


