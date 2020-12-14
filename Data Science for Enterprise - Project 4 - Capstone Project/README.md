# Udacity Capstone Sparkify Project

### Project Definition
Sparkify is a fictional popular digital music service similar to Spotify or Pandora. With Sparkify, many users stream their favorite songs with this service and are able to do so through the free tier which places advertisements between songs or using the premium subscription model. Users are able to upgrade, downgrade, or cancel their service at any time, so it's very important to make sure the users love the service. <br>
Every time a user interacts with the service whether they are playing songs, logging out, liking a song with a thumbs up, hearing an ad, or downgrading their service, it generates data. All this data contains the key insights for keeping the users happy and helping the business thrive.<br>
With this project, I will be leveraging the data to predict which users are at risk to churn, either in the form of downgrading their service or cancelling the service altogether. By being able to accurately identify the users before they leave the service, Sparkify could potentially offer discounts and incentives and potentially save the business millions in revenue.

### Project Components
These are the components you'll find in this project.<br/>
1. Sparkify Python Notebook<br/>
This Sparkify Python Notebook contains all the code executed against the Sparkify dataset. In the notebook, I load and explore the data, cleansing where possible, and using the final dataset to build multiple predictive models that help me to determine when a user may be on the path to churning and cancelling their subscription with Sparkify.<br/>

2. Sparkify Notebook HTML Output<br/>
This is from the Sparkify notebook, after successfully running all the cells, the notebook was saved and downloaded as an HTML file for viewing purposes.<br/>

3. README.md<br/>
This current file

### Libraries And Technologies Used
PySpark (Spark libraries and specific functions)<br>
Numpy <br>
Pandas <br>
Matplotlib <br>
Jupyter Notebook <br>
Python <br>

### Summary of Analysis
In this project, I needed to create various predictive models to determine which model would be best for predicting churn in the Sparkify dataset.<br>
2 predictive models were created, and each were evaluated based on their Accuracy and F1-Score<br>
1. Logistic Regression Model - Resulted in Accuracy of ~.7 and F1 Score of ~.58<br>
2. GBT Classifier Model - Resulted in Accuracy of ~.66 and F1 Score of ~.63<br>
Based on the above, it seems like the GBT Classifier was the optimal model to use to predict churn based on the balance of Accuracy and F1-Score. Also the GBT Classifier can certainly be fine tuned more to increase the metrics.

### Tuning of GBT Classifier Model
After tuning the GBT classifier, I was able to improve the model's metrics to an Accuracy of ~.73 and F1 Score of ~.68.<br>
These are the parameters that worked best for my GBT Classifer after tuning. <br>
    -maxIter (20) <br>
    -maxDepth (3)<br>
    -minInfoGain (0) <br>
    -minInstancesPerNode (1)<br>

### Conclusion
Conclusion:<br>
In this project, I was able to implement a python notebook that loads, explores, and cleans customer data which is then used to build a machine learning model that can predict customer churn. In the dataset used for the predictive models, I had 9 features (not including the churn feature nor userId). I was able to build 2 machine learning models which are 1. Logistic Regression Model and 2. GBT Classifier Model. After comparing the Accuracy and F1-Score for each model, I determined that GBT Classifier Model was best balanced for predicting customer churn.<br><br>

Improvements:<br>
I think some improvements could certainly be made to this implementation. In this particular project, I have only used subset of the data (12MB). The full dataset is 12GB which would certainly make a HUGE difference and would allow me to have a better idea on which features may be worth a serious look and which features may be more important when creating a predictive model. Although 12MB of data certainly provides enough for this initial model, there can be some inaccuracies that could have been missed. Also, in the models I created, I simply used some specific features that I thought were relevant for my model although I could have certainly used more or all the features in my final dataset and possibly have gotten a better model. I also could have tuned the model's hyperparameters more, although this would have taken a lot more time and computing power. <br><br>

Reflection:<br>
I enjoyed working through this project as it reflected a very real example. This is certainly something that all types of customers go through where they want to make sure they can really understand their clients better and be able to have a proactive approach rather than a reactive approach when it comes to their customers leaving. As explained in previous lessons of the Udacity Nanodegree program, the most important aspect of this project as well as for any other project with lots of data, is exploring and preprocessing the data. In order to really have models that are useful in a real world example, there needs to be a lot of thinking on how you can preprocess your data to make sure you can create a dataset that is ready for training models with. With bad preprocessing, the models created may not be as effective as you might want.

### Blog Post:
You can find the blog post explaining the technical details of the project in the link below. <br>
https://medium.com/@lrodrig/predicting-customer-churn-with-pyspark-6a4526cdc0b7

### Acknowledgements
Udacity for providing relevant dataset and Nanodegree based on real world experience

### Resources
Resources used for this project include dataset provided by Udacity, help from the Udacity Knowledge Hub, and Google/StackOverflow
