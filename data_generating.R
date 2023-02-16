# get all annual reports for all companies and years listed below


# importing libraries
library(edgar) # API package
library(jsonlite) # read_json
library(data.table) # rbindlist()
library(ff) # move folder to different directory
library(magrittr) # %>%
library(tidyr) # only for the gather function 



# set working directory to project folder
setwd("your_path")


# get a list of company codes from the API 
company_codes_all <- read_json("https://www.sec.gov/files/company_tickers.json") %>% rbindlist() 

# save the file 
write.csv(company_codes_all, "data/company_codes_all.csv", row.names = FALSE) # for late use



#### data for our model
# input for the API

company_codes <- c(1018724, 1065088, 21344)
relevant_years <- c(2021, 2022)


# use Edgar package to get all annual reports 
# https://cran.r-project.org/web/packages/edgar/edgar.pdf

# user-agent for api request
user_agent <- "Felix Froschauer felix.froschauer@student.uni-tuebingen.de"


# getFilingsHTML retrieves all specified company-year 10-k reports as .txt and .html
get_fillings <- getFilingsHTML(company_codes, "10-K", relevant_years, quarter = 1, useragent = user_agent)

# save the file
write.csv(get_fillings, "data/our_company_list.csv", row.names = FALSE)


# for a clean structure, we can delete folder "full_text" and folder "Master Indexes"
# and move the relevant files to have a clear structure

old_path <- "Edgar filings_HTML view"
new_path <- "data/all_filings_html"

file.move(old_path, new_path)


# not needed, can be deleted
old_path_text <- "Edgar filings_full text"
new_path_text <- "data/all_filings_txt"

file.move(old_path_text, new_path_text)
  
# not needed, can be deleted
old_path_idx <- "Master Indexes"
new_path_idx <- "data/master_indexes"

file.move(old_path_idx, new_path_idx)

