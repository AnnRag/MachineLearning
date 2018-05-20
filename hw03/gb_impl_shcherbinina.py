{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "#coding=utf-8\n",
    "\n",
    "from sklearn.base import BaseEstimator\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "import numpy as np\n",
    "\n",
    "\n",
    "# Параметрами с которыми вы хотите обучать деревья\n",
    "TREE_PARAMS_DICT = {'max_depth': 1}\n",
    "# Параметр tau (learning_rate) для вашего GB\n",
    "TAU = 0.05\n",
    "\n",
    "def loss_function(y, h):\n",
    "    return (h-y)**2 / 2.0\n",
    "\n",
    "def loss_grad(y,h):\n",
    "    return y-h\n",
    "\n",
    "class SimpleGB(BaseEstimator):\n",
    "    def __init__(self, tree_params_dict, iters, tau):\n",
    "        self.tree_params_dict = tree_params_dict\n",
    "        self.iters = iters\n",
    "        self.tau = tau\n",
    "        \n",
    "    def fit(self, X_data, y_data):\n",
    "        self.base_algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, y_data)\n",
    "        self.estimators = []\n",
    "        curr_pred = self.base_algo.predict(X_data)\n",
    "        for iter_num in range(self.iters):\n",
    "            # Нужно посчитать градиент функции потерь\n",
    "            grad = loss_grad(y_data,curr_pred) # TODO\n",
    "            # Нужно обучить DecisionTreeRegressor предсказывать антиградиент\n",
    "            # Не забудьте про self.tree_params_dict\n",
    "            algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, -grad) # TODO\n",
    "\n",
    "            self.estimators.append(algo)\n",
    "            # Обновите предсказания в каждой точке\n",
    "            curr_pred += self.tau * algo.predict(X_data) # TODO\n",
    "        return self\n",
    "    \n",
    "    def predict(self, X_data):\n",
    "        # Предсказание на данных\n",
    "        res = self.base_algo.predict(X_data)\n",
    "        for estimator in self.estimators:\n",
    "            res += self.tau * estimator.predict(X_data)\n",
    "        # Задача классификации, поэтому надо отдавать 0 и 1\n",
    "        return res > 0.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
