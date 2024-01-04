from flask import Flask, request, jsonify
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Cargar datos de productos desde un archivo JSON
with open('data.json', 'r') as file:
    productos = json.load(file)

# Preprocesamiento para el cálculo de similitud del coseno
product_descriptions = [
    f"{producto['brand']} {producto['color']} {producto['style']}" for producto in productos]
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(product_descriptions)


def get_similar_products(input_producto, num_similar=5, similarity_threshold=0.2):
    input_product_name = input_producto["name"]

    product_index = next((i for i, producto in enumerate(
        productos) if producto["name"] == input_product_name), None)
    if product_index is None:
        return []

    cosine_similarities = cosine_similarity(
        tfidf_matrix[product_index], tfidf_matrix).flatten()
    similar_indices = np.argpartition(
        cosine_similarities, -num_similar)[-num_similar:]

    # Usar una lista para almacenar los productos similares y sus similitudes
    similar_productos = []
    for i in similar_indices:
        if cosine_similarities[i] >= similarity_threshold:
            similar_productos.append(
                {"product": productos[i], "similarity": cosine_similarities[i]})

    return similar_productos


@app.route('/products', methods=['GET'])
def get_all_productos():
    return jsonify(productos)


@app.route('/product/<string:nombre>', methods=['GET'])
def get_producto_by_nombre(nombre):
    producto = next((p for p in productos if p['name'] == nombre), None)
    if producto:
        return jsonify(producto)
    else:
        return "Producto no encontrado", 404


@app.route('/recomendar/<string:nombre>', methods=['GET'])
def recomendar_productos(nombre):
    producto = next((p for p in productos if p['name'] == nombre), None)
    if producto:
        similar_productos = get_similar_products(producto)
        return jsonify(similar_productos)
    else:
        return "Producto no encontrado", 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
