import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Cargar datos de productos desde un archivo JSON
with open('data.json', 'r') as file:
    productos = json.load(file)

# Preprocesamiento para el cálculo de similitud del coseno
product_descriptions = [f"{producto['brand']} {producto['color']} {producto['style']}" for producto in productos]
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(product_descriptions)

num_similar = 5
similarity_threshold = 0.2
input_product_name =  "Zapatillas Adidas Performance"
product_index = next((i for i, producto in enumerate(productos) if producto["name"] == input_product_name), None)
cosine_similarities = cosine_similarity(tfidf_matrix[product_index], tfidf_matrix).flatten()
similar_indices = np.argpartition(cosine_similarities, -num_similar)[-num_similar:]

# Usar una lista para almacenar los productos similares y sus similitudes
similar_productos = []
for i in similar_indices:
    if cosine_similarities[i] >= similarity_threshold:
        similar_productos.append({"product": productos[i], "similarity": cosine_similarities[i]})
print(similar_productos)