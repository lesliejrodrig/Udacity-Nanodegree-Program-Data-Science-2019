# Udacity-Seattle-AirBnb-DataScienceNanodegree

## Motivation For This Project
This project is part of a Udacity Data Scientist for Enterprise Nanodegree Program. I will be analyzing Seattle AirBnB data using the Cross-Industry Process for Data Mining (CRISP-DM). 
In this project, I will look to answer the following questions:

1. Which AirBnB listings are the most expensive?
2. Does the overall price of AirBnbs change throughout the year? (Possible impact by seasons)
3. What areas of Seattle do customers prefer to stay in and which do they avoid? (Popular Neighborhoods vs Unpopular Neighborhoods)
4. What is the average price in the popular neighborhood versus the unpopular neighborhood?

## Libraries And Technologies Used
Numpy
Pandas
Matplotlib
Jupyter Notebook
Python

## Files In This Repository
I have gathered the Seattle AirBnB data provided by Kaggle here (https://www.kaggle.com/airbnb/seattle). 
The following files are in this repository: 
Listings.csv - includes full descriptions of listings and average review score 
Reviews.csv - includes unique id for each reviewer and detailed comments 
Calendar.csv - includes listing id and the price and availability for that day
Data Science Blog Post Project.ipynb - Notebook used to gather, assess, prepare, and analyze the datasets and answer my questions
README.md 

## Summary Of Results of Analysis
1. Which AirBnB listings are the most expensive?
Listing ID 4825073 is the most expensive AirBnB listing on average from my entire dataset. Listing ID 3345341 is almost as expensive as well and worth a mention.

2. Does the overall price of AirBnbs change throughout the year? (Possible impact by seasons)
The first few months of the year experience lower prices with a quick spike in the summer where the average prices jump above 100 and then higher prices towards the end of the year. 

3. What areas of Seattle do customers prefer to stay in and which do they avoid? (Popular Neighborhoods vs Unpopular Neighborhoods)
Tourists prefer to stay in the areas of Capitol Hill, Ballard, Minor, Queen Anne, and Belltown.
Tourists tend to avoid booking listings in the areas of Fairmount Park, North College Park, Roxhill, Pinehurst, and South Park

4. What is the average price in the popular neighborhood versus the unpopular neighborhood?
The  most popular neighbourhood averages around 116 while some of the other popular neighbourhoods like Belltown average a higher price at almost 140.
North College Park has average price around 100 but the second least popular neighbourhood (Fairmount Park) has average prices over 350! Possibly reason for this discussed in notebook.

Link to my blog post can be found here: https://medium.com/@lrodrig/an-analysis-on-seattle-airbnb-data-bbf3726f55d6

## Acknowledgements
Udacity and Kaggle and StackOverflow