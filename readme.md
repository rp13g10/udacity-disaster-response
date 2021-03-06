# Disaster Response Pipelines

This project aims to take free-text messages from social networks and identify the topics being discussed (in relation to natural disasters). It contains the following elements:
* An ETL pipeline which processes the provided csv files into a usable format
* A ML pipeline which trains a multi-output classifer which tags messages with categories
* A dashboard which allows users to classify new messages and displays some visualizations of the data contained in the training dataset.

Since model training took longer than expected, I didn't spend much time experimenting with this aspect of the project (instead opting to focus more on the dashboard). Future improvements which I might have implemented include:
* Use bigrams instead of individual words, combined with PCA to keep the total number of features under control
* Resample data to address the category imbalance (duplicate records from less common categories)
* Switch to a neural network model. LSTM/GRU architectures are commonly used to interpret textual information, and being able to leverage the power of my GPU through CUDA would probably have significantly reduced training time.


## Requirements

* dash >= 1.9.1
* dash-daq >= 0.4.0
* matplotlib >= 3.2.1
* nltk >= 3.4.4
* numpy >= 1.18.1
* networkx >= 2.4
* plotly >= 4.6.0
* scikit-learn >= 0.22.2
* sqlalchemy >= 1.3.16
* waitress >= 1.4.3
* wordcloud >= 1.6.0
* xgboost >= 1.0.2

## Project Outline
The structure of this project is outlined below:

__ETL Pipeline__

* app/data/process_data.py
    * Ingests data from the provided csv files,  parses them into a format which can be used for model creation and saves the output to a SQLite database
    * <i>python app/data/process_data.py app/data/disaster_messages.csv app/data/disaster_categories.csv app/data/udacity.db</i>

__ML Pipeline__

* app/models/train_classifier.py
    * Ingests the data generated by process_data.py, uses it to train and evaluate a classifier object. Saves the object and generated metrics to disk.
    * I left this to run overnight, training takes several hours
    * <i>python app/models/train_classifier.py app/data/udacity.db app/models/model.pkl</i>

__Dashboard__

* app/App.py
    * This module defines the underlying flask app which will run the dashboard. Keeping it separate is helpful for larger apps which implement logins and other more complex functionality.
* app/Utilities.py
    * Contains the functions required to either generate more complex html structures, or process/visualize data. These feed in to the page content as defined in Index.py
* app/Index.py
    * This module defines the overall page layout, and the content of static page elements such as the nav bar. When called, it will serve the dashboard on port 8080
    * <i>python app/Index.py</i>