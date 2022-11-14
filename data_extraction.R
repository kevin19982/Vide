# get all annual reports for all companies and years listed below


# importing libraries
library(edgar) # API package
library(jsonlite) # read_json
library(data.table) # rbindlist()
library(ff) # move order to different directory
library(magrittr) # %>%
library(tidyr) # only for the gather function => specific package?



# set working directory
setwd("/Users/felix/Documents/git/Vide")


# a list of CIK codes from the API 
# for the API, company codes are deeded to get the reports
# they can be gathered here:
company_codes_all <- read_json("https://www.sec.gov/files/company_tickers.json") %>% rbindlist() # API request to get the company codes

# save the file 
write.csv(company_codes_all, "data/company_codes_all.csv", row.names = FALSE) # for late use


# for now we get the codes manually
#### data for our model

### need to be set as variable (the dataframe!)

company_names <- c("amazon", "walmart", "cocacola", "apple", "ebay")
company_codes <- c(1018724, 104169, 21344, 320193, 1065088)
relevant_years <- c(2020, 2021, 2022)

year_2020 <- rep(2020, 5)
year_2021 <- rep(2021, 5)
year_2022 <- rep(2022, 5)


data <- data.frame(name = company_names, code = company_codes, year_2020, year_2021, year_2022) %>% 
  gather(key="delete", value="year", 3:5) %>% select(-delete)


# export the data for later use
write.csv(data, "data/our_company_list.csv", row.names = FALSE)





# define the user agent (needed for an APi request here)
user_agent <- "Felix Froschauer felix.froschauer@student.uni-tuebingen.de"


# use Edgar package to get all annual reports 
# https://cran.r-project.org/web/packages/edgar/edgar.pdf

# all html files 
get_fillings <- getFilingsHTML(company_codes, "10-K", relevant_years, quarter = 1, useragent = user_agent)

# we can delete folder full_text and folder Master Indexes ()


#### move the data into the data directory

old_path <- "Edgar filings_HTML view"
new_path <- "data/all_fillings_html"

file.move(old_path, new_path)




