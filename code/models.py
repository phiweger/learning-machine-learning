# 1 ............................................................................
# Where we fit a model.


import numpy as np
from sklearn.linear_model import LinearRegression


X = 2 * np.random.rand(100, 1)
n, m = 4, 8  # ground truth
noise = np.random.randn(100, 1)  # sampled from the 'standard normal' distribution


y = n + m * X + noise


model = LinearRegression()
model.fit(X, y)
n_hat, m_hat = model.intercept_, model.coef_


print(
    'n ... truth:', n, ', estimate:', '{0:.2f}'.format(*n_hat), '\n'
    'm ... truth:', m, ', estimate:', '{0:.2f}'.format(*m_hat[0]), '\n')


# 2 ............................................................................
# Where we fit a model using gradient descent.
# Note that we'll use some of the variables from # 1.


eta = 0.1  # learning rate
n_iterations = 1000
size = 100  # number of samples
X_b = np.c_[np.ones((100, 1)), X]  # make X compatible for matmul equation


theta = np.random.randn(2, 1)  # random initialization
for _ in range(n_iterations):
    gradients = 2 / size * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
    # Gradients are the set of partial derivatives of the MSE .. mean squared error.
    # The matrix algebra is from the 'normal equation', which is a closed form
    # solution to the MSE optimization. Yeah, I know.


print(
    'n ... truth:', n, ', estimate:', '{0:.2f}'.format(*theta[0]), '\n'
    'm ... truth:', m, ', estimate:', '{0:.2f}'.format(*theta[1]), '\n')


# create figure
with open('lr_data.csv', 'w+') as out:
    out.write('x,y\n')
    for i in range(len(X)):
        out.write('{},{}\n'.format(*X[i], *y[i]))

with open('lr_gradient_descent.csv', 'w+') as out:
    out.write('n,m,eta,iteration\n')  # header

    for eta in [0.01, 0.1, 0.5]:
        theta = np.random.randn(2, 1)  # random initialization

        for i in range(n_iterations):
            gradients = 2 / size * X_b.T.dot(X_b.dot(theta) - y)
            theta = theta - eta * gradients
            out.write(
                '{},{},{},{}\n'.format(*theta[0], *theta[1], eta, i))

'''R

library(ggplot2)

lr <- read.table('lr_gradient_descent.csv', sep=',', header=T)
data <- read.table('lr_data.csv', sep=',', header=T)

theme_min <-
theme_bw() +
    theme(
        panel.border=element_blank(),
        # panel.grid.minor=element_blank(),
        panel.grid.major=element_blank(),
        panel.spacing=unit(2, 'lines'),
        strip.background=element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1))

p <-
ggplot() +
    geom_point(
        data=data,
        mapping=aes(x, y), size=0.5) +
    facet_wrap(~eta) +
    geom_abline(
        data=lr[lr$iteration < 10,],
        mapping=aes(slope=m, intercept=n, colour=iteration)) +
    theme_min

q <-
ggplot(lr, aes(n, m, colour=iteration)) +
    geom_point(size=0.5) +
    facet_wrap(~eta, scale='free') +
    theme_min

ggsave('lr_gradient_descent_lines.png', p, height=7, width=20, units='cm')
ggsave('lr_gradient_descent_points.png', q, height=7, width=20, units='cm')
'''


# 3 ............................................................................
# Where we evaluate our model.


from collections import Counter

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score as accuracy

'''
wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

7. Attribute Information:
   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   5. class:
      -- Iris Setosa
      -- Iris Versicolour
      -- Iris Virginica
'''


iris = datasets.load_iris()
X = iris.data
# We want to perform logistic regression, so we can only have two classes.
# To generalize this to more classes see softmax regression.
y = [1 if i else 0 for i in iris.target == 2]  # 1 if Iris Virginica else 0
class_name = iris.target_names


# For now, we pretend we froze a certain part of the data away as a test set.
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    random_state=42,  # seed
    stratify=y,       # Counter(y_test), try w/o stratification
    test_size=0.33)


model = LogisticRegression(solver='lbfgs')
# solver .. 'optimizer', like gradient descent
# lbfgs .. Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm
model.fit(X_train, y_train)
model.coef_  # 1 coefficient for each feature
y_hat = model.predict(X_val)


# some basic metrics
confusion_matrix(y_val, y_hat)
accuracy(y_val, y_hat)


# Now we will use the loss function explicitly.
from sklearn.metrics import log_loss, mean_squared_error, median_absolute_error
# log_loss ... logistic regression
# MSE ...      linear regression, l2 norm (Euclidean distance)
# MAE ...      linear regression, l1 norm (Manhatten distance)
log_loss(y_hat, y_val)


# 4 ............................................................................
# Where we look at the loss function a bit closer.


# Let's watch the model train. We can diagnose overfitting by looking at the
# dynamics of the loss fn.
X_train, X_val, y_train, y_val = train_test_split(
    X, y, random_state=42, stratify=y, test_size=0.33)


err_train, err_val, size_train = [], [], []
model = LogisticRegression(solver='lbfgs')


for m in range(2, len(X_train)):  # lbfgs needs at least two classes to start
    model.fit(X_train[:m], y_train[:m])

    y_train_hat = model.predict(X_train[:m])  # predict the same data wejust learned
    y_val_hat = model.predict(X_val)  # predict the whole validation set

    err_train.append(log_loss(y_train_hat, y_train[:m]))
    err_val.append(log_loss(y_val_hat, y_val))
    size_train.append(m)


with open('log_loss.csv', 'w+') as out:
    result = zip(size_train, err_train, err_val)
    out.write('size_train,err_train,err_val\n')
    for i in result:
        out.write('{},{},{}\n'.format(*i))


'''R

library(ggplot2)
library(tidyr)
library(RColorBrewer)

df_wide <- read.table('log_loss.csv', sep=',', header=T)  # turn from wide to long fmt
df_long <- gather(df, error, log_loss, err_train:err_val, factor_key=TRUE)

p <-
ggplot(df_long, aes(x=size_train, y=log_loss, colour=error)) +
    geom_line() +
    scale_color_brewer(palette='Set1') +
    xlab('size of training set') +
    ylab('log loss') +
    theme_bw() +
    theme(
        panel.border=element_blank(),
        panel.grid.major=element_blank())

ggsave('log_loss.png', p, width=10, height=7, units='cm')
'''


# 5 ............................................................................
# Where we use cross validation to tune hyperparameters.
# Note that we'll use some of the variables from # 4.


from sklearn.model_selection import GridSearchCV


X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y, test_size=0.33)
# Note how we now specify a 'test' not 'validation' set alongside the training set.
# This is because we will use cross validation on the training set, which makes
# the validation split implicitly.


model = LogisticRegression(solver='liblinear')
# Optimizer liblinear needed bc/ we wnat to fit w/ 2 penalties, see below.
# C = 1 / lambda, see stackoverflow, 21816346
param_grid = {
    # If you have no idea, try consecultive orders of magnitude away from the default.
    'C': [0.001, 0.01, 0.1, 1, 2, 3, 10, 100, 1000],
    # Or exhaust the parameter space.
    'penalty': ('l1', 'l2')}
# Note that small imbalanced samples might provide counterintuitive CV results,
# see e.g. stackoverflow, 32889929.


cv = GridSearchCV(model, param_grid, scoring='accuracy').fit(X_train, y_train)
cv.best_estimator_
cv.best_params_
cv.cv_results_


y_hat = cv.predict(X_test)
confusion_matrix(y_test, y_hat)
accuracy(y_test, y_hat)  # Booya!
