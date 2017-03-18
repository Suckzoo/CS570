import numpy as np
from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error


def main(rs=108):
    data = pd.read_csv("./data/Hitters.csv", header=0)
    response_var = -1
    y_vec = data.ix[:, response_var].as_matrix().reshape(-1, 1)
    y_label = data.columns[response_var]

    x_label = ", ".join(data.columns[1:-1])
    x_mat = data.ix[:, 1:-1].as_matrix()
    x_mat = x_mat.reshape(-1, x_mat.shape[1])

    x_train, x_test, y_train, y_test = train_test_split(x_mat, y_vec, test_size=0.2, random_state=rs)

    # Linear Regression
    rss, r2, mse = multi_var_hitter(x_train, x_test, y_train, y_test, x_label)
    print("Linear Regression Result")
    print("RSS: {}".format(rss))
    print("R^2: {}".format(r2))
    print("MSE: {}".format(mse))
    print()

    # Ridge Regression
    best_lambda_ridge, best_lambda_lasso = get_best_lambda_value_ridge_lasso(data)
    rss, r2, mse = multi_var_hitter_ridge(x_train, x_test, y_train, y_test, x_label, best_lambda_ridge)
    print("Ridge Regression Result")
    print("RSS: {}".format(rss))
    print("R^2: {}".format(r2))
    print("MSE: {}".format(mse))
    print("Best lambda value: {}".format(best_lambda_ridge))
    print()

    # lasso
    rss, r2, mse = multi_var_hitter_lasso(x_train, x_test, y_train, y_test, x_label, best_lambda_lasso)
    print("lasso Result")
    print("RSS: {}".format(rss))
    print("R^2: {}".format(r2))
    print("MSE: {}".format(mse))
    print("Best lambda value: {}".format(best_lambda_lasso))
    print()

def get_best_lambda(estimator, X, y):
    lambda_candidate = [2 ** k for k in range(0, 30)]

    default_estimator = estimator(alpha=0)

    best_lambda = 0
    best_score = cross_val_score(default_estimator, X, y=y, cv=10).mean()

    for l in lambda_candidate:
        regr = estimator(alpha=l)
        scores = cross_val_score(regr, X, y=y, cv=10)
        score = scores.mean()
        if best_score < score:
            best_score = score
            best_lambda = l

    return best_lambda, best_score

def get_best_lambda_value_ridge_lasso(data):
    """
    Implement Here
    The grader will call this function to get the lambda value,
    and run the functions with hidden test data.
    Do not write exact value on best_lambda_ridge and best_lambda_lasso.
    You should implement the function to find the best lambda value.
    """
    x_mat = data.ix[:, 1:-1].as_matrix()
    x_mat = x_mat.reshape(-1, x_mat.shape[1])
    y_vec = data.ix[:, -1].as_matrix().reshape(-1, 1)

    best_lambda_ridge, best_score_ridge = get_best_lambda(linear_model.Ridge, x_mat, y_vec)
    best_lambda_lasso, best_score_lasso = get_best_lambda(linear_model.Lasso, x_mat, y_vec)

    return best_lambda_ridge, best_lambda_lasso


def multi_var_hitter(x_train, x_test, y_train, y_test, x_label):
    regr = linear_model.LinearRegression()

    regr.fit(x_train, y_train)
    predicted_y_test = regr.predict(x_test)
    rss = np.sum((predicted_y_test - y_test) ** 2)
    r2 = r2_score(y_test, predicted_y_test)
    mse = mean_squared_error(y_test, predicted_y_test)
    return rss, r2, mse


def multi_var_hitter_ridge(x_train, x_test, y_train, y_test, x_label, best_lambda):
    regr = linear_model.Ridge(alpha=best_lambda)

    regr.fit(x_train, y_train)
    predicted_y_test = regr.predict(x_test)
    rss = np.sum((predicted_y_test - y_test) ** 2)
    r2 = r2_score(y_test, predicted_y_test)
    mse = mean_squared_error(y_test, predicted_y_test)

    return rss, r2, mse


def multi_var_hitter_lasso(x_train, x_test, y_train, y_test, x_label, best_lambda):
    regr = linear_model.Lasso(alpha=best_lambda)

    regr.fit(x_train, y_train)
    predicted_y_test = regr.predict(x_test)
    rss = np.sum((predicted_y_test - y_test) ** 2)
    r2 = r2_score(y_test, predicted_y_test)
    mse = mean_squared_error(y_test, predicted_y_test)

    return rss, r2, mse


if __name__ == "__main__":
    main()

