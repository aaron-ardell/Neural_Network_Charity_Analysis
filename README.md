# Neural_Network_Charity_Analysis

## Overview:
The purpose of this project is to provide a deep_learning neural network model to analysis and classify the success of charitable donations in a supplied dataset that is capable of predicting whether applicants will be successful if funded by our client, Alphabet Soup.

#### We will use the following stages to analyze the data:
- preprocessing the data
- compile, train and evaluate the model
- optimizing the model

#### Resources:
Data Source: charity_data.csv

## Results:

#### Preprocessing:
- The column IS_SUCCESSFUL contains binary information and is the target variable for our neural network.
- The following columns APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT are the features for our model.
- The columns EIN  and NAME possess identifying labels that are not repeated, thus unnecessary in our machine learning process. These columns were dropped.

#### Compiling, Training, and Evaluating the Model
- This deep learning model is made of two hidden layers with 80 and 20 neurons respectively, both using the 'relu' activation function.
- The following results were produce: loss: 0.7714 and accuracy: 0.6587, below the 75% goal.
- The deciding factor in the optimization process was reinserting the NAME category as a feature.

## Summary:

#### Optimization:
- Initially, providing some added adjustments to binning parameters provided minor results.
  - When the NAME catagory was returned as a feature, this is ultimately what pushed the model over the 75% accuracy rating.
  
  ```
  # Look at NAME counts for binning
  name_counts = application_df['NAME'].value_counts()
  name_counts
  
  # Choose a cut-off value and create a list of names_replace to be replaced.
  names_replace = list(name_counts[name_counts<5].index)
  names_replace

  # Replace in dataframe
  for name in names_replace:
    application_df['NAME'] = application_df['NAME'].replace(name,"Other")

  # Check to make sure binning was successful
  application_df['NAME'].value_counts()
  ```

- Increasing the number of epochs from 5 to 100 only provided a slight increase in prediction accuracy of 77.66% to 78.16%.
- A third attempt at adjusting the neuron counts for the hidden layers posed weaker impact on the prediction strength of the model and became irrelevant after the adjustment to the first attempt concerning the NAME feature.
- Final results of the model: loss: 0.7165 - accuracy: 0.7816.
