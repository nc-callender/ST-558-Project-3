# ST-558-Project-3

## Description of Purpose
This repository contains documents relating to the analysis of a Diabetes Health Indicators Dataset.  The dataset was split based on the Education level of the patients. The education levels were:

  - Elementary
  - Some High School
  - High School Graduate
  - Some College or Technical School
  - College Graduate

For each education level, exploratory data analysis was conducted. Models were developed using a training set.  The models were then evaluated on their ability to predict response on a test set.

## Packages
The following R packages were used:  

  - tidyverse  
  - caret    

## Render Code
The following code was used to render a single parameterized rmd file to generate the five separate md files for github. 

EducationLevel <- c("Elementary", "Some High School","High School Graduate", "Some College or Technical School", "College Graduate")

output_file <- paste0(EducationLevel, "_Analysis.md")

params = lapply(EducationLevel, FUN = function(x){list(EducationLevel = x)})  

reports <- tibble(output_file, params)  

apply(reports, MARGIN= 1, FUN = function(x){rmarkdown::render(input="Project 3.Rmd", output_file=x[[1]], params = x[[2]], output_options = list(toc=TRUE, toc_depth= 3))})

## Links
Links to the html files for each level:  

  [Elementary](https://nc-callender.github.io/ST-558-Project-3/Elementary_Analysis)  
  [Some High School](https://nc-callender.github.io/ST-558-Project-3/Some%20High%20School_Analysis)  
  [High School Graduate](https://nc-callender.github.io/ST-558-Project-3/High%20School%20Graduate_Analysis)  
  [Some College or Technical School](https://nc-callender.github.io/ST-558-Project-3/Some%20College%20or%20Technical%20School_Analysis)   
  [College Graduate](https://nc-callender.github.io/ST-558-Project-3/College%20Graduate_Analysis)    
