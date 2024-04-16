
# from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# a = [[0.5, 37, 128],
#      [1.0, 44, 1024],
#      [0.1, 22, 512]]
# scaler = StandardScaler()
# normalized_features = scaler.fit_transform(a)
# print(normalized_features)

# s2 = StandardScaler()
# b = [[1,], [3,], [5,]]
# normalized_speedup = scaler.fit_transform(b)
# print(normalized_speedup)

# a = {"a":1, "b":2, "c":3}
# b = {k: v+1 for k, v in a.items()}
# print(b)

# sel = VarianceThreshold()
# a = [[0,1,2],
#      [0, 1, 3],
#      [0, 0, 1]]
# b = sel.fit_transform(a)
# print(b)

a = "1024kB"
print(a[:-2])

minutes = 410*1024/64*0.5
print(minutes / 60)

# def preprocessing_features(record_list: List[ProgramRecord]):
#     # Extract features from record_list
#     features = [record.features for record in record_list]

#     # Initialize a scaler
#     scaler = StandardScaler()

#     # Fit the scaler to the features and transform them
#     normalized_features = scaler.fit_transform(features)

#     # Replace original features with normalized ones
#     for record, features in zip(record_list, normalized_features):
#         record.features = features

#     return record_list