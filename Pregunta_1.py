import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

datos = pd.read_csv("smogon.csv")
vec = TfidfVectorizer(ngram_range=(1,3))
x = vec.fit_transform(datos["moves"])

print("\nEl vocabulario tiene esta cantidad de elementos:")
print(len(vec.vocabulary_))
print("\nEste es el vocabulario:")
print(vec.vocabulary_)

data_frame_tfidf = pd.DataFrame(x.toarray(),columns=vec.get_feature_names_out())
print("\nMatriz TF-IDF con cabecera de elementos del vocabulario:")
print(data_frame_tfidf)

print("\nAhora hacemos K-means:")
km = KMeans(n_clusters=5, n_init=100)
lista = km.fit_predict(data_frame_tfidf)
print(lista)

datos.drop(["moves"], axis=1, inplace=True)
datos.drop(["texto"], axis=1, inplace=True)
datos.drop(["url"], axis=1, inplace=True)
datos["Tipo"] = lista
print(datos)

datos.to_csv("NuevosDatos.csv")