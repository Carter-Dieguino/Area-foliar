import matplotlib.pyplot as plt
import networkx as nx

# Crear un grafo dirigido
G = nx.DiGraph()

# Agregar nodos con sus etiquetas
nodos = [
    ("A", "Inicio"),
    ("B", "Carga/Captura Imagen"),
    ("C", "Preprocesamiento de Imagen"),
    ("D", "Segmentación de Imagen"),
    ("E", "Cálculo del Área de la Hoja"),
    ("F", "Cálculo del NDVI"),
    ("G", "Extracción de Información Adicional"),
    ("H", "Interfaz de Usuario y Salida de Datos"),
    ("I", "Fin"),
]

G.add_nodes_from([nodo[0] for nodo in nodos])

# Agregar aristas
aristas = [
    ("A", "B"),
    ("B", "C"),
    ("C", "D"),
    ("D", "E"),
    ("D", "F"),
    ("E", "G"),
    ("F", "G"),
    ("G", "H"),
    ("H", "I"),
]

G.add_edges_from(aristas)

# Dibujar el grafo
pos = nx.spring_layout(G, seed=42)  # para posiciones consistentes
plt.figure(figsize=(12, 8))

nx.draw(G, pos, with_labels=False, arrows=True)
for nodo, (x, y) in pos.items():
    texto = dict(nodos)[nodo]
    plt.text(x, y, texto, fontsize=12, ha='right', va='bottom')

plt.title('Diagrama de Flujo del Software de Análisis de Hoja')
plt.axis('off')  # Ocultar los ejes
plt.show()
