import numpy as np

from ._base import NeighborsBase


class KNeighborsClassifier(NeighborsBase):
    '''
    k nearest neightboar classifier

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

        parameters
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
        n_classes = self.classes.shape[0]
        y_prob = np.empty((n_queries, n_classes))
        y_pred = np.empty(n_queries, dtype=int)

        for i, X_i in enumerate(X):
            # cari tetangga terdekat
            neigh_id, neigh_dist = self._kneighbors(X_i)

            # dapatkan list label tetangga terdekat
            neigh_y = self._y[neigh_id]

            # hitung bobot
            weights = self._get_weights(neigh_dist)

            # hitung probabilitas
            for j, class_j in enumerate(self.classes):
                class_match = (neigh_y == class_j).astype(int)

                if self.weights == 'uniform':
                    class_counts = np.sum(class_match)
                elif self.weights == 'distance':
                    class_counts = np.dot(weights, class_match)

                y_prob[i, j] = class_counts

            # menentukan prediksi
            id_max_i = np.argmax(y_prob[i])
            y_pred[i] = self.classes[id_max_i]

        return y_pred


    def predict_proba(self, X):
        '''
        prediksi probabilitas untuk data test X

        parameters
        ----------
        X : {array-like} of shape (n_queries, n_features)
            test sample

        return
        ------
        y_prob : ndarray of shape (n_queries, n_features)
            hasil probabilitas dari input samplme
        '''

        # validasi input
        if np.ndim(X) == 0:
            raise ValueError('gunakan data test (X) dengan tipe data sejenis array')

        # konvesi input ke ndarray
        X = np.array(X).copy()

        # inisiasi
        n_queries = X.shape[0]
        n_classes = self.classes.shape[0]
        y_prob = np.empty((n_queries, n_classes))

        for i, X_i in enumerate(X):
            # cari tetangga terdekat
            neigh_id, neigh_dist = self._kneighbors(X_i)

            # dapatkan list label tetangga terdekat
            neigh_y = self._y[neigh_id]

            # hitung bobot
            weights = self._get_weights(neigh_dist)

            # hitung probabilitas
            for j, class_j in enumerate(self.classes):
                class_match = (neigh_y == class_j).astype(int)

                if self.weights == 'uniform':
                    class_counts = np.sum(class_match)
                elif self.weights == 'distance':
                    class_counts = np.dot(weights, class_match)

                y_prob[i, j] = class_counts

            # normalisasi probabilitas
            sum_i = np.sum(y_prob[i])
            y_prob[i] /= sum_i

        return y_prob
