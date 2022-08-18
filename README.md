# Nextlab


## Question 4
During the fourth semester, we were asked to do an assignment on app store in which we received a data contains app details like rating, number of users, primary genre, secondary genre etc. We were asked to rank the apps. My classmates used the ratings to rank the apps but the apps that don't have many users receiving a 5 star rating get an overall 5 star rating. So I used the number of users as weights and then multiplied it to the rating of the app.

## Question 5
Backpropagation is an algorithm used in artificial intelligence (AI) to fine-tune mathematical weight functions and improve the accuracy of an artificial neural network's outputs.

1. If the data set is large:
We can just simply remove the rows with missing data values.
It is the quickest way, we use the rest of the data to predict the values.

2. For smaller data sets:
We can substitute missing values with the mean or average of the rest of the data using the pandas' data frame in python. There are different ways to do so, such as df.mean(), df.fillna(mean).
