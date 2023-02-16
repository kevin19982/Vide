# get all annual reports for all companies and years listed below


# importing libraries
library(edgar) # API package
library(jsonlite) # read_json
library(data.table) # rbindlist()
library(ff) # move order to different directory
library(magrittr) # %>%
library(tidyr) # only for the gather function => specific package?



# set working directory
#setwd("/Users/felix/Documents/git/Vide/new/recent_12_02/Vide")
setwd("/Users/felix/Documents/cloud/Data-testing")


# get a list of company codes from the API 
company_codes_all <- read_json("https://www.sec.gov/files/company_tickers.json") %>% rbindlist() # API request to get the company codes

# save the file 
write.csv(company_codes_all, "data/company_codes_all.csv", row.names = FALSE) # for late use


# for now we get the codes manually
#### data for our model

company_names <- c("amazon", "walmart", "cocacola", "apple", "ebay")
#company_codes <- c(1018724, 104169, 21344, 320193, 1065088)
company_codes <- c(1018724, 1065088, 21344)
relevant_years <- c(2021, 2022)




# use Edgar package to get all annual reports 
# https://cran.r-project.org/web/packages/edgar/edgar.pdf

# user-agent for api request
user_agent <- "Felix Froschauer felix.froschauer@student.uni-tuebingen.de"


# all html files 

# getFilingsHTML retrieves all specified company-year 10-k reports as .txt and .html
get_fillings <- getFilingsHTML(company_codes, "10-K", relevant_years, quarter = 1, useragent = user_agent)

write.csv(get_fillings, "data/our_company_list.csv", row.names = FALSE)


# for a clean structure, we can delete folder "full_text" and folder "Master Indexes"
# and move the relevant under a more obvious structure


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

