import numpy as np

from ._base import NeighborsBase


class KNeighborsRegression(NeighborsBase):
    '''
    k nearest neightboar regretion

    parameters
    ----------
    n_neighboars : int, default=5
        jumlah tetangga yang akan digunakan sebagai referensi perhitungan

    weights : {'uniform', 'distance'}, default='uniform'
        bobot untuk menentukan prediksi
        'uniform' : semua tetannga mempunyai bobot yang sama
        'distance' : semakin dekat tetangga dengan titik target, semakin besar bobotnya

    p : int, default=2
        power dari minkowski distance
        - p=1 setara dengan manhattan_distance (l1)
        - p=2 setara dengan euclidian_distance (l2)

    returns
    -------
    None

    '''

    def fit(self, X, y):
        '''
        fit k-nearest neighbors classifier dari raining dataset

        parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            training data
        y : {array-like} of shape (n_samples)
            target values
        '''

        return self._fit(X, y)


    def predict(self, X):
        '''
        prediksi target berdasarkan data yang diberikan

        paramaters
        ----------
        X : {array-like} of shape (n_queries, n_features)
            test sample

        return
        ------
        y_pred : ndarray of shape (n_queries, n_features)
            hasil prediksi dari input samplme
        '''

        # validasi input
        if np.ndim(X) == 0:
            raise ValueError('gunakan data test (X) dengan tipe data sejenis array')

        # konvesi input ke ndarray
        X = np.array(X).copy()

        # inisiasi
        n_queries = X.shape[0]
        y_pred = np.empty(n_queries)

        for i, X_i in enumerate(X):
            # cari tetangga terdekat
            neigh_id, neigh_dist = self._kneighbors(X_i)

            # dapatkan list label tetangga terdekat
            neigh_y = self._y[neigh_id]

            # hitung bobot
            weights = self._get_weights(neigh_dist)

            if self.weights == 'uniform':
                y_pred[i] = np.mean(neigh_y)
            elif self.weights == 'distance':
                num = np.sum(neigh_y * weights)
                denom = np.sum(weights)
                y_pred[i] = num/denom

        return y_pred
