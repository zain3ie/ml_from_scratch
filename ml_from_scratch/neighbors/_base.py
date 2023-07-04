import numpy as np


class NeighborsBase:
    '''
    nearest neightboar base

    parameters
    ----------
    n_neighboars : int, default=5
        jumlah tetangga yang akan digunakan sebagai referensi perhitungan

    weights : {'uniform', 'distance'}, default='uniform'
        bobot untuk menentukan prediksi
        'uniform' : semua tetangga mempunyai bobot yang sama
        'distance' : semakin dekat tetangga dengan titik target, semakin besar bobotnya

    p : int, default=2
        power dari minkowski distance
        - p=1 setara dengan manhattan_distance (l1)
        - p=2 setara dengan euclidian_distance (l2)
    '''

    def __init__(self, n_neighbors=5, p=2, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p

        self._validate()


    def _validate(self):
        '''
        fungsi untuk memvalidasai atribute kelas
        '''
        if self.weights not in ['uniform', 'distance']:
            raise ValueError("pilih jenis bobot (weight) antara 'uniform' atau 'distance'")


    def _fit(self, X, y):
        '''
        fit k-nearest neighbors classifier dari raining dataset

        parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            training data
        y : {array-like} of shape (n_samples)
            target values
        '''

        # validasi input
        if np.ndim(X) == 0:
            raise ValueError('gunakan data sample (X) dengan tipe data sejenis array')

        if np.ndim(y) == 0:
            raise ValueError('gunakan data target (y) dengan tipe data sejenis array')

        if len(X) != len(y):
            raise ValueError('jumlah data sample (X) harus sama dengan data jumlah target (y)')

        # konversi input ke ndarray dan simpan data ke kelas
        self._X = np.array(X).copy()
        self._y = np.array(y).copy()

        # simpan daftar unik target ke kelas
        self.classes = np.unique(y)


    def _compute_distance(self, p1, p2):
        '''
        hitung distance diantara 2 titik menggunakan minkowski distance

        parameters
        ----------
        p1 : {array-like} of shape (n_features)
            titik pertama

        p2 : {array-like} of shape (n_features)
            titik kedua

        returns
        -------
        dist : float
            jarak minkowski dari dua buah titik
        '''

        # selisih absolut antara p1 & p2
        abs_diff = np.abs(p1-p2)

        # pangkatkan selisih dan jumlahkan element
        sigma_diff = np.sum(abs_diff**self.p)

        # akarkan hasil sigma
        dist = sigma_diff**(1/self.p)

        return dist


    def _get_weights(self, dist):
        '''
        menghitung bobot berdasarkan array jarak

        parameters
        ----------
        dist : ndarray
            input jarak

        returns
        -------
        weights_arr : array dengan dimensi yang sama dengan dist
            if weights == 'uniform', maka return None
        '''

        if self.weights == 'uniform':
            weights_arr = None
        elif self.weights == 'distance':
            weights_arr = np.where(dist==0, 0, 1/(dist**2))

        return weights_arr


    def _kneighbors(self, X):
        '''
        fungsi untuk mencari tetangga dari suatu titik

        parameters
        ----------
        X : {array-like} of shape (n_features)
            titik yang ingin dicari tetangganya

        returns
        -------
        neigh_id : ndarray of shape (n_neighboars)
            list index tetangga terdekat dari titik X

        neigh_dist : ndarray of shape (n_neighboars)
            list jarak tetangga terdekat dari titik X
        '''

        # inisialisasi list tetangga terdekat
        neigh_id = np.full(self.n_neighbors, -1)
        neigh_dist = np.full(self.n_neighbors, np.inf)

        for id, X_i in enumerate(self._X):
            # hitung jarak
            current_dist = self._compute_distance(p1=X, p2=X_i)

            # jika tetangga ini jaraknya lebih dekat
            # dari tetangga terjauh di list tetangga terdekat
            if current_dist < neigh_dist[-1]:
                # ganti tetangga paling jauh dari list tetangga terdekat
                # dengan tetangga baru
                neigh_id[-1] = id
                neigh_dist[-1] = current_dist

                # urutkan ulang list tetangga terdekat
                id_sorted = np.argsort(neigh_dist)
                neigh_id = neigh_id[id_sorted]
                neigh_dist = np.sort(neigh_dist)

        return neigh_id, neigh_dist
