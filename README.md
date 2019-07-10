# Patent:ly
Patently: the app that helps inventors through the first steps of the patenting process.


## Getting Started
Patent:ly is an app that uses natural language processing (NLP) and ML document similarity to identify the most similar abstracts in the patent dataset to the user's input.  


## Prerequisites
This tool was developed using:
    Python (v3.7.3)
    pandas (v0.24.2)
    scikit-learn (v0.21.2)
    spaCy (v2.1) with en_core_web_sm
    Flask (v1.0.2)


## Installing
To get this app running in your machine, download this repository in its entirety. 

App run file:
patently/flask-app/app.py

Development Jupyter file:
patently-development.ipynb

Subset of patent dataset:
patently/flask-app/data/export_patents_view_main_tokenized_v2.csv


## Development
To expand to a larger dataset or other fields, go to the Google Cloud Platform and download from Google Patents Public Dataset the "PatentsView Data".  You will need to set up an account and authentication.





