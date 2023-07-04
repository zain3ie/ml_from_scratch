from ml_from_scratch.neighbors import KNeighborsClassifier, KNeighborsRegression


X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
clf.fit(X, y)
y_pred = clf.predict(X)
y_proba = clf.predict_proba(X)

reg = KNeighborsRegression(n_neighbors=3, weights='distance')
reg.fit(X, y)
y_pred_reg = reg.predict(X)

print(y_proba)
print(y_pred)
print(y_pred_reg)
