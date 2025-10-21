# Importamos librerías necesarias
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans

# Lista de tipos Pokémon
tipos = [
    "fire", "water", "grass", "electric", "ice", "fighting",
    "poison", "ground", "flying", "psychic", "bug", "rock",
    "ghost", "dragon", "dark", "steel", "fairy"
]

# Cargamos los datos
datos_pokemon = pd.read_csv('smogon.csv')

# Convertimos la columna a texto y minúsculas
datos_pokemon["moves"] = datos_pokemon["moves"].astype(str).str.lower()

# Limpiamos caracteres no deseados
textos_limpios = []
for texto in datos_pokemon["moves"]:
    nuevo_texto = re.sub(r'[^a-z0-9\s]', ' ', texto)
    textos_limpios.append(nuevo_texto)
datos_pokemon["moves"] = textos_limpios

# Añadimos espacios a los tipos y limpiamos espacios extra
for tipo in tipos:
    datos_pokemon["moves"] = datos_pokemon["moves"].str.replace(tipo, f" {tipo} ", regex=False)
datos_pokemon["moves"] = datos_pokemon["moves"].str.replace(r'\s+', ' ', regex=True).str.strip()

# TF-IDF general
vec = TfidfVectorizer(ngram_range=(1,3))
x = vec.fit_transform(datos_pokemon["moves"])
data_frame_tfidf = pd.DataFrame(x.toarray(), columns=vec.get_feature_names_out())

# K-Means (17 grupos)
km = KMeans(n_clusters=17, n_init=100)
lista = km.fit_predict(data_frame_tfidf)

# Conteo por tipos definidos
vec_count = CountVectorizer(vocabulary=tipos)
X_count = vec_count.fit_transform(datos_pokemon["moves"])
matriz_conteos = pd.DataFrame(X_count.toarray(), columns=vec_count.get_feature_names_out())

# Detectamos los dos tipos con mayor frecuencia por Pokémon
matriz_mayores = [[], []]
for i in range(len(matriz_conteos)):
    fila = matriz_conteos.iloc[i].sort_values(ascending=False)
    tipo_1 = fila.index[0]
    tipo_2 = fila.index[1] if len(fila) > 1 else None
    matriz_mayores[0].append(tipo_1)
    matriz_mayores[1].append(tipo_2)

df_mayores = pd.DataFrame({"Tipo 1": matriz_mayores[0], "Tipo 2": matriz_mayores[1]})

# Unimos y guardamos
datos_pokemon.drop(["moves", "texto", "url"], axis=1, inplace=True)
datos_pokemon = pd.concat([datos_pokemon, matriz_conteos, df_mayores], axis=1)
datos_pokemon["Primer_Cluster"] = lista
datos_pokemon.to_csv("Pokemon_Pregunta_3_1.csv")
