import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

datos = pd.read_csv("smogon.csv")

tipos = [
    "fire", "water", "grass", "electric", "ice", "fighting",
    "posion", "ground", "flying", "psychic", "bug", "rock",
    "ghost", "dragon", "dark", "steel", "fairy"
]

datos["moves"] = datos["moves"].str.lower()
for tipo in tipos:
    datos["moves"] = datos["moves"].str.replace(
        tipo, f" {tipo} ", regex=False
    )
datos["moves"] = datos["moves"].str.replace(r"\s+", " ", regex=True)

vec = TfidfVectorizer(ngram_range=(1,3),vocabulary=tipos)
x = vec.fit_transform(datos["moves"])

data_frame_tfidf = pd.DataFrame(x.toarray(),columns=vec.get_feature_names_out())

km = KMeans(n_clusters=17, n_init=100)
lista = km.fit_predict(data_frame_tfidf)

datos.drop(["moves"], axis=1, inplace=True)
datos.drop(["texto"], axis=1, inplace=True)
datos.drop(["url"], axis=1, inplace=True)
datos["Tipo"] = lista
print(datos)

datos.to_csv("Pokemon_por_tipos.csv")