# Importamos librerías necesarias
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Tipos de Pokémon
tipos = [
    "fire", "water", "grass", "electric", "ice", "fighting",
    "poison", "ground", "flying", "psychic", "bug", "rock",
    "ghost", "dragon", "dark", "steel", "fairy"
]

# Cargamos el archivo CSV
datos_pokemon = pd.read_csv('smogon.csv')

# Convertimos la columna a texto y minúsculas
datos_pokemon["moves"] = datos_pokemon["moves"].astype(str).str.lower()

# Limpiamos caracteres no deseados con re
textos_limpios = []
for texto in datos_pokemon["moves"]:
    nuevo_texto = re.sub(r'[^a-z0-9\s]', ' ', texto)
    textos_limpios.append(nuevo_texto)

# Reemplazamos la columna con texto limpio
datos_pokemon["moves"] = textos_limpios

# Agregamos espacios antes y después de los tipos
for tipo in tipos:
    datos_pokemon["moves"] = datos_pokemon["moves"].str.replace(tipo, f" {tipo} ", regex=False)

# Quitamos espacios extra
datos_pokemon["moves"] = datos_pokemon["moves"].str.replace(r'\s+', ' ', regex=True).str.strip()

# Creamos el vector TF-IDF (1 a 3 palabras)
vec = TfidfVectorizer(ngram_range=(1,3))
x = vec.fit_transform(datos_pokemon["moves"])

# Mostramos el vocabulario y su tamaño
print("\nEl vocabulario tiene esta cantidad de elementos:")
print(len(vec.vocabulary_))
print("\nEste es el vocabulario:")
print(vec.vocabulary_)

# Convertimos a DataFrame para ver la matriz TF-IDF
data_frame_tfidf = pd.DataFrame(x.toarray(),columns=vec.get_feature_names_out())
print("\nMatriz TF-IDF con cabecera de elementos del vocabulario:")
print(data_frame_tfidf)

# Aplicamos K-Means con 17 clústeres
print("\nAhora hacemos K-means:")
km = KMeans(n_clusters=17, n_init=100)
lista = km.fit_predict(data_frame_tfidf)
print(lista)

# Eliminamos columnas y añadimos el resultado del clúster
datos_pokemon.drop(["moves", "texto", "url"], axis=1, inplace=True)
datos_pokemon["Primer_Cluster"] = lista
print(datos_pokemon)

# Guardamos el resultado en un nuevo CSV
datos_pokemon.to_csv("Pokemon_Pregunta_1.csv")