# get all annual reports for all companies and years listed below

# importing libraries
library(httr)
library(jsonlite)
library(dplyr)
library(readr)
library(data.table)
library(stringr)
library(readr)
library(xml2)
library(rjson)

# api package
library(edgar)


# set working directory
setwd("/Users/felix/Documents/git/Vide")


# a list of CIK codes from the API 
# for the API, company codes are deeded to get the reports
# they can be gathered here:


company_codes_all <- read_json("https://www.sec.gov/files/company_tickers.json") %>% rbindlist() # API request to get the company codes
write.csv(company_codes_all, "data/company_codes_all.csv", row.names = FALSE) # for late use

# for now we get the codes manually


#### data for our model
company_names <- c("amazon", "walmart", "cocacola", "apple", "ebay")
company_codes <- c(1018724, 104169, 21344, 320193, 1065088)
relevant_years <- c(2020, 2021, 2022)


# define the user agent (needed for an APi request here)
user_agent <- "Felix Froschauer felix.froschauer@student.uni-tuebingen.de"


# use Edgar package to get all annual reports 
# https://cran.r-project.org/web/packages/edgar/edgar.pdf

# all html files 
get_fillings <- getFilingsHTML(company_codes, "10-K", relevant_years, quarter = 1, useragent = user_agent)

# we can delete folder full_text and folder Master Indexes ()




# 
# html_file <- read_html("Edgar filings_HTML view/Form 10-K/21344/21344_10-K_2020-02-24_0000021344-20-000006.html")
# text_file <- html_text(html_file)
# 
# text_file
# 
# write_file(text_file, "new_file_report.txt")



# now store the data in the right folder and make the format more intuitive
# path <- paste0("Edgar filings_HTML view/Form 10-K/", company_codes[1], "/")
# 
# 
# 
# old_name <- list.files(path)[1]
# new_name <- paste0(path, company_names[1], "_", relevant_years[1], ".html")
# 
# file.rename(paste0(path, old_name), paste0(path, new_name))
# 











# variables
# 
# 
# list_of_companies <- c("amazon")
# 
# ### working with edgar package
# 
# 
# get_fillings <- getFilingsHTML(1018724, "10-K", 2020, quarter = 1, useragent = user_agent)
# 
# amazon <- 1018724
# walmart <- 104169
# cola <- 21344
# airbnb <- 1559720
# 
# 
# 
# 
# 
# get_fillings <- getFilingsHTML(airbnb, "10-K", 2021, quarter = 1, useragent = user_agent)
# # transform data into minimal HTML? to make it readable for the model
# 
# result <- read_html("Edgar filings_HTML view/Form 10-K/21344/21344_10-K_2020-02-24_0000021344-20-000006.html")
# 
# 
# data <- read_file("Edgar filings_full text/Form 10-K/21344/21344_10-K_2020-02-24_0000021344-20-000006.txt")
# 
# data
# 
# library(rvest)
# 
# 
# strip_html <- function(s) {
#   html_text(read_html(s))
# }
# 
# 
# html_file <- read_html("Edgar filings_HTML view/Form 10-K/21344/21344_10-K_2020-02-24_0000021344-20-000006.html")
# 
# text_file <- html_text(html_file)
# 
# text_file
# 
# write_file(text_file, "new_file_report.txt")
# 




##################### junk ###############################

################## Part I: Get adjusted company codes ################## 

# # get a list of CIK codes from the API 
# cik_codes <- read_json("https://www.sec.gov/files/company_tickers.json")
# cik_codes <- rbindlist(cik_codes)
# write.csv(cik_codes, "data/company_tickers.csv", row.names = FALSE)
# 
# 
# # to get the complete CIK codes, zeros have to be added such that to length is equal to 10
# adjusted_cik_vector <- matrix(nrow = nrow(cik_codes), ncol = 1)
# 
# # this function is only called once in this script
# adjust_cik_codes <- function(cik_codes) {
#   for (i in 1:nrow(cik_codes)){
#     
#     #determine number of zeros needed
#     num_zeros <- 10 - str_length(cik_codes$cik_str[i])
#     list_zeros <- rep(0, num_zeros)
#     string_zeros <- str_replace_all(str_replace_all(toString(list_zeros), ",", ""), " ", "")
#     
#     # add zeros to the beginning
#     new_cik_code <- paste0(string_zeros, cik_codes$cik_str[i])
#     
#     # store the adjusted values in a predefined list
#     adjusted_cik_vector[i] <- new_cik_code
#   }
#   
#   # either return the vector or we could already change the column inside this function
#   return(adjusted_cik_vector)
# }
# 
# # execute function from above
# new_column <- adjust_cik_codes(cik_codes)
# cik_codes$cik_str <- new_column
# 
# 
# write.csv(cik_codes, "data/company_tickers_adjusted.csv", row.names = FALSE)
# 

################## Part II: --------------------------------- ##################

# # read in the data from above
# adjusted_cik_codes <- read_csv("data/company_tickers_adjusted.csv")
# 
# 
# 
# # define url 
# amazon_url <- "https://data.sec.gov/api/xbrl/companyfacts/CIK0001018724.json"
# 
# # GET to receive the data from the API (with user agent as needed https://www.sec.gov/os/accessing-edgar-data)
# 
# # get all data for amazon
# amazon <- GET(url = amazon_url, config = add_headers(`User-Agent` = user_agent, 
#                                                      `Accept-Encoding` = "gzip, deflate"))
# amazon$status_code
# 
# content_2 <- content(amazon, as="text", encoding = "UTF-8") %>% fromJSON()
# 
# 
# 
# 
# # get only specific data for amazon
# 
# specific <- "data.sec.gov/api/xbrl/frames/us-gaap/AccountsPayableCurrent/USD/CY2020.json"
# 
# specific_json <- GET(url = specific, config = add_headers(`User-Agent` = user_agent,
#                                                           `Accept-Encoding` = "gzip, deflate"))
# 
# 
# 
# specific_json$status_code
# b
# 
# raw <- content(specific_json, as="text", encoding = "UTF-8") %>% fromJSON(flatten=FALSE)
# 
# 
# 
# ## testing json file 
# 
# one_company <- fromJSON(file= "data/CIK0000001750.json")
# 
# print(one_company)
# 
# write.table(one_company, file = "data/one_company.txt") #, sep="\t", row.names = TRUE, col.names= NA)

















#next_step <- amazon$content %>% read_html(encoding = "UTF-8", as="text") %>% fromJSON(amazon, flatten=FALSE)
#data_raw <- try(content(amazon, as="text", encoding ="UTF-8") %>% fromJSON(amazon, flatten=FALSE))

#amazon_2

# amazon <- read_json(amazon_url, 
#                     config = add_headers(
#                       "User-Agent" = user_agent, 
#                       "Accept-Encoding" = "gzip", "deflate"))$sz



#object.size(year.master)/1000000

