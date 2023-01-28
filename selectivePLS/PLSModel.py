import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV

class NormalPLS:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y):
        self.X = X
        self.y = y
        w_list = []
        t_list = []
        p_list = []
        q_list = []
        t = np.zeros((X.shape[0], 1))
        q = np.zeros((1, 1))
        p = np.zeros(((X.shape[1], 1)))
        for i in range(self.n_components):
            X = X - (t @ p.T)
            y = y - (q * t)
            w = (X.T @ y) / np.linalg.norm(X.T @ y)
            t = X @ w
            p = (X.T @ t) / (t.T @ t)
            q = (y.T @ t) / (t.T @ t)
            w_list.append(w)
            t_list.append(t)
            p_list.append(p)
            q_list.append(q)
        self.W = np.hstack(w_list)
        self.T = np.hstack(t_list)
        self.P = np.hstack(p_list)
        self.q = np.vstack(q_list)

    def predict(self, X, output_latent_variables=False):
        t_list = []
        for i in range(self.n_components):
            t = X @ self.W[:, i].reshape(-1, 1)
            X = X - (t @ self.P[:, i].reshape(1, -1))
            t_list.append(t)
        T_pred = np.hstack(t_list)
        y_pred = T_pred @ self.q
        if output_latent_variables:
            return y_pred, T_pred
        else:
            return y_pred

    def vip(self, feature_names):
        vip_values = np.zeros((self.W.shape[0],))
        s = np.diag(((self.T.T @ self.T) @ self.q.reshape(1, -1).T) @ self.q.reshape(1, -1)).reshape(self.W.shape[1],
                                                                                                     -1)
        sum_s = np.sum(s)
        for i in range(self.W.shape[0]):
            weight = np.array([(self.W[i, j] / np.linalg.norm(self.W[:, j])) ** 2 for j in range(self.W.shape[1])])
            vip_values[i] = np.sqrt(self.W.shape[0] * (s.T @ weight) / sum_s)
        vip_data = pd.DataFrame(np.array([feature_names, vip_values]).T, columns=['FeatureName', 'VIP'])
        vip_data.sort_values('VIP', ascending=False, inplace=True)
        return vip_data

    def show_weight_plot(self, feature_names, component):
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax.bar(feature_names, self.W[:, component], 0.5)
        ax.axhline(0, color='black')
        ax.tick_params(axis='x', rotation=90)
        ax.tick_params(axis='y', labelsize=15)
        ax.set_ylabel('Weight', fontsize=15)
        ax.set_title(
            'Weight of latent component ' + str(component) + '. ' + f'y loading is {self.q.ravel()[component]:.3f}',
            fontsize=20)
        plt.show()

class selectivePLS:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y, latent_to, alpha_from=0, alpha_to=1, alpha_num=100, cv=4):
        # Set fitting information
        self.latent_to = latent_to
        self.alpha_from = alpha_from
        self.alpha_to = alpha_to
        self.alpha_num = alpha_num
        self.cv = cv
        # Split X
        X_L = X[:, :latent_to]
        X_R = X[:, latent_to:]
        # Remove X_R effects from X_L
        remove_model = RidgeCV(alphas=np.linspace(alpha_from, alpha_to, alpha_num), cv=cv)
        remove_model.fit(X_R, y)
        y_removed = y - remove_model.predict(X_R)
        self.remove_model_alpha = remove_model.alpha_
        # Normal PLS Regression
        pls_model = NormalPLS(self.n_components)
        pls_model.fit(X_L, y_removed)
        self.pls_model = pls_model
        # Parameter estimation (Final model)
        _, X_latent = pls_model.predict(X_L, output_latent_variables=True)
        selective_PLS_model = RidgeCV(alphas=np.linspace(alpha_from, alpha_to, alpha_num), cv=cv)
        selective_PLS_model.fit(np.hstack([X_latent, X_R]), y)
        self.selective_PLS_model = selective_PLS_model

    def predict(self, X):
        X_L = X[:, :self.latent_to]
        X_R = X[:, self.latent_to:]
        _, X_latent = self.pls_model.predict(X_L, output_latent_variables=True)
        y_pred = self.selective_PLS_model.predict(np.hstack([X_latent, X_R]))
        return y_pred

    def show_weight_plot(self, feature_names, component):
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax.bar(feature_names, self.pls_model.W[:, component], 0.5)
        ax.axhline(0, color='black')
        ax.tick_params(axis='x', rotation=90)
        ax.tick_params(axis='y', labelsize=15)
        ax.set_ylabel('Weight for projection to latent space', fontsize=15)
        ax.set_title('Weight of latent component ' + str(
            component) + '. ' + f'y loading is {self.selective_PLS_model.coef_.ravel()[component]:.3f}', fontsize=20)
        plt.show()