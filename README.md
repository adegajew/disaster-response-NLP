# Disaster Response
This project was done as a part of data science/ ML course. The idea is to classify disaster messages so that they can be sent to an appropriate disaster relief agency.    

## 🔥 Content:

- [About](#about)
- [File Descriptions](#file-descriptions)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Acknowledgments](#acknowledgments)

## About
Dataset contains messages sent during disaster events. ETL pipeline is to process messages and load them to sql db. Machine Learning pipeline is to categorize these events. WebApp is to display visualization and categorize new messages.

In the ML Preparation step different multiclass classification methods were tested: decision tree, kNN, ADAboost.

## File Descriptions
Jupyter notebooks ETL Pipeline Preparation & ML Pipeline Preparation are for initial data analysis and preparation for data pipelines.

💥 data -  this folder contains:
+ datasets: disaster_categories.csv, disaster_messages.csv
+ process_data.py with ETL pipeline that cleans data

## Prerequisites
Anaconda distribution of Python 3.*.

NLTK data preparation part requires additional downloads:
- nltk.download('punkt')
- nltk.download('stopwords')
- nltk.download('wordnet')
- nltk.download('omw-1.4')

## Installation
1. Run the following commands in the project's root directory:
- To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- To run ML pipeline that trains classifier and saves it to pickle
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

In order to run a web app type in the command line: python run.py

## Acknowledgments
Data sets and idea for the project were provided by [Udacity](https://www.udacity.com/).