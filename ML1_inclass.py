"""
ML1 In-Class
.py file
"""
# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
# (https://colab.research.google.com/github.com/UVADS/DS-3021/blob/main/
# 04_ML_Concepts_I_Foundations/ML1_inclass.py#scrollTo=9723a7ee)

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
# (https://colab.research.google.com/github.com/UVADS/DS-3001/blob/main/
# 04_ML_Concepts_I_Foundations/ML1_inclass.ipynb#scrollTo=9723a7ee)
# %%

# %%
# import packages
from turtle import color
from pydataset import data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
# set the dimension on the images
import plotly.io as pio
pio.templates.default = "plotly_dark" # set dark theme

# %%
iris = data('iris')
iris.head()

# %%
# What mental models can we see from these data sets?
# A mental model that we can see from this data set is model where physical measurements of flowers (sepal length, sepal width, petal length, and petal width) are used to distinguish between different species
# What data science questions can we ask?
# Can we accurately predict the species of an iris flower using its physical measurements?
# Which features (sepal or petal dimensions) are most important for classification?
# How well do different machine learning models perform on this dataset?

# %%
"""
Example: k-Nearest Neighbors
"""
# We want to split the data into train and test data sets. To do this,
# we will use sklearn's train_test_split method.
# First, we need to separate variables into independent and dependent
# dataframes.

X = iris.drop(['Species'], axis=1).values  # features
y = iris['Species'].values  # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
# we can change the proportion of the test size; we'll go with 1/3 for now

# %%
# Now, we use the scikitlearn k-NN classifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# %%
# now, we check the model's accuracy:
neigh.score(X_train, y_train)

# %%
# now, we test the accuracy on our testing data.
neigh.score(X_test, y_test)

# %%
"""
Patterns in data
"""
# Look at the following tables: do you see any patterns? How could a
# classification model point these out?
# Yes, there are clear patterns in the tables, especially when the data are grouped by species. Petal measurements differ a lot between species, while sepal measurements overlap more. A classification model can pick up on these patterns by grouping similar flowers together and using those similarities to predict the species of new flowers.
patterns = iris.groupby(['Species'])
patterns['Sepal.Length'].describe()

# %%
patterns['Sepal.Width'].describe()

# %%
patterns['Petal.Length'].describe()

# %%
patterns['Petal.Width'].describe()

# %%
# scatter plot using plotly
fig = px.scatter_3d(iris, x='Sepal.Length', y='Sepal.Width', z='Petal.Length',
                 color='Species', title='Iris Sepal Dimensions')
fig.show()
# %%
"""
Mild disclaimer
"""
# Do not worry about understanding the machine learning in this example!
# We go over kNN models at length later in the course; you do not need to
# understand exactly what the model is doing quite yet.
# For now, ask yourself:

# 1. What is the purpose of data splitting?
# To separate data into training and evaluation sets so the model can be tested on unseen data and avoid overfitting.
# 2. What can we learn from data testing/validation?
# To measure how well the model generalizes and to tune parameters without biasing performance.
# 3. How do we know if a model is working?
# A model is working if it performs well on validation/testing data using appropriate metrics and shows consistent results.
# 4. How could we find the model error?
# Model error can be found by comparing predictions to true values using error metrics.

# If you want, try changing the size of the test data
# or the number of n_neighbors and see what changes!
