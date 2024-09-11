import tkinter as tk
from tkinter import filedialog, simpledialog
import cv2
import os
import pandas as pd
from datetime import datetime
from PIL import Image, ImageTk
import numpy as np
import xlsxwriter

# Funciones de línea.py
def cargar_imagen(imagen):
    alto, ancho = imagen.shape[:2]
    cuarta_parte = imagen[:, 3*ancho//4:]
    return cuarta_parte

def medir_longitud_linea(imagen):
    def distancia_puntos(p1, p2):
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris_blur = cv2.GaussianBlur(gris, (5, 5), 0)

    # Aplicar una transformación morfológica para mejorar la detección de la línea
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(gris_blur, cv2.MORPH_CLOSE, kernel)

    bordes = cv2.Canny(morphed, 50, 200)
    contornos, _ = cv2.findContours(bordes.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        contorno_largo = max(contornos, key=cv2.contourArea)
        extremos = np.squeeze(contorno_largo)
        inicio = extremos[extremos[:, 0].argmin()]
        fin = extremos[extremos[:, 0].argmax()]
        longitud_linea = distancia_puntos(inicio, fin)

        # Dibuja la línea medida en la imagen para verificar
##        cv2.line(imagen, tuple(inicio), tuple(fin), (0, 255, 0), 2)

        return longitud_linea
    return 0

def calcular_area_hoja(image, PIXELS_PER_UNIT):
    # Convertir a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Mejorar la detección mediante una transformación morfológica
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)

    # Umbralizar para obtener una imagen binaria (hoja en blanco y fondo en negro)
    _, thresholded_image = cv2.threshold(morphed, 127, 255, cv2.THRESH_BINARY_INV)

    # Limitar el área de trabajo a las tres cuartas partes izquierdas
    altura, anchura = thresholded_image.shape
    area_trabajo = thresholded_image[:, :3 * anchura // 4]

    # Encontrar los contornos en el área de trabajo
    contornos, _ = cv2.findContours(area_trabajo, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Iniciar el área total con el contorno más externo
    area_total = 0
    for i, contorno in enumerate(contornos):
        # `RETR_CCOMP` genera una jerarquía para distinguir contornos externos (hoja) e internos (agujeros)
        area_contorno = cv2.contourArea(contorno)
        if i == 0:
            # El primer contorno encontrado debería ser el contorno externo (la hoja en sí)
            area_total += area_contorno
        else:
            # Contornos subsiguientes son agujeros y se restan del área total
            area_total -= area_contorno

    # Dibujar solo el contorno externo de la hoja en la imagen original
    image[:, :3 * anchura // 4] = cv2.drawContours(image[:, :3 * anchura // 4], contornos, 0, (0, 0, 255), 2)

    # Calcular el área en función de la referencia de píxeles por unidad
    area_unidad2 = area_total / (PIXELS_PER_UNIT ** 2)

    # Mostrar la imagen con el contorno dibujado
##    cv2.imshow('Hoja Procesada', image)
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()

    return area_unidad2




##import os
##from tkinter import filedialog

class LeafAreaCameraApp:
    def __init__(self, root):
        # Inicialización e interfaz gráfica
        self.root = root
        self.root.title("Calculadora de Área Foliar con Cámara")
        self.folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.folder_name, exist_ok=True)
        self.image_counter = 1
        self.data = []
        self.longitud_linea = 0
##        self.area_hoja_pix = 0  # Almacenar el último valor de `area_hoja_pix`

        # Configuración predeterminada
        self.valor_equivalencia = 1.0
        self.unidad_equivalencia = "cm"

        # Ventana de datos guardados
        self.data_window = None
        self.images_frame = None
        self.selected_session = tk.StringVar(value=self.folder_name)

        # Marco para la cámara
        self.camera_frame = tk.Frame(root)
        self.camera_frame.pack(padx=10, pady=10)

        # Inicializar la cámara
        self.camera = cv2.VideoCapture(1)

        # Label para mostrar la vista de la cámara
        self.camera_label = tk.Label(self.camera_frame)
        self.camera_label.pack()

        # Botón para capturar imagen
        self.capture_button = tk.Button(self.root, text="Capturar Imagen", command=self.process_image)
        self.capture_button.pack(pady=10)

        # Botón para elegir la medida de referencia
        self.measurement_button = tk.Button(self.root, text="Seleccionar Medida", command=self.ajustar_medida)
        self.measurement_button.pack(pady=10)

        # Botón para ver imágenes y Excel
        self.view_data_button = tk.Button(self.root, text="Ver Imágenes y Excel", command=self.mostrar_datos_guardados)
        self.view_data_button.pack(pady=10)

        # Label para mostrar resultados
        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack()


        self.ultimo_area_hoja_pix = 0  # Agrega esta línea para almacenar el último valor
        self.ultima_longitud_linea = 0  # Para guardar el valor anterior de longitud_linea


        self.update_camera()

    def update_camera(self):
        ret, frame = self.camera.read()
        if ret:
            height, width = frame.shape[:2]

            # Dibuja la línea divisoria de la cuarta parte
            fourth_divider = width * 3 // 4
            cv2.line(frame, (fourth_divider, 0), (fourth_divider, height), (0, 255, 0), 2)

            # Parte izquierda para el área de la hoja
            leaf_area = frame[:, :fourth_divider]
            contornos_hoja = self.detectar_contornos_hoja(leaf_area)

            # Cálculo en tiempo real del área de la hoja en píxeles, pero mantén el valor anterior si no hay cambios
            nuevo_area_hoja_pix = cv2.contourArea(contornos_hoja[0]) if contornos_hoja else 0
            if nuevo_area_hoja_pix != 0:
                self.ultimo_area_hoja_pix = nuevo_area_hoja_pix

            # Recalcular `longitud_linea` cada vez para obtener el valor actualizado
            nueva_longitud_linea = medir_longitud_linea(frame[:, fourth_divider:])

            # Si la longitud ha cambiado, actualiza el área
            if nueva_longitud_linea != self.ultima_longitud_linea:
                self.ultima_longitud_linea = nueva_longitud_linea

            PIXELS_PER_UNIT = self.ultima_longitud_linea / self.valor_equivalencia if self.valor_equivalencia != 0 else 1

            # Cálculo del área en unidades cuadradas usando la longitud de referencia
            area_hoja_cm2 = self.ultimo_area_hoja_pix / PIXELS_PER_UNIT ** 2 if self.ultima_longitud_linea != 0 else 0

            # Superponer el área calculada y longitud de referencia en la imagen de la cámara
            overlay = frame.copy()
            cv2.drawContours(overlay, contornos_hoja, -1, (0, 0, 255), -1)  # Rellenar el contorno de la hoja en rojo

            # Parte derecha para la longitud de la línea
            line_area = frame[:, fourth_divider:]
            contornos_linea = self.detectar_contornos_linea(line_area)

            # Ajustar el offset para el contorno de la línea
            offset = fourth_divider
            if contornos_linea:
                cv2.drawContours(overlay, [contornos_linea[0] + np.array([[offset, 0]])], -1, (255, 0, 255), -1)
                cv2.putText(overlay, f"L: {self.ultima_longitud_linea:.2f}px ({self.valor_equivalencia}{self.unidad_equivalencia})",
                            (offset + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # Texto para el área de la hoja en la imagen
            cv2.putText(overlay, f"Area hoja (sombra): {self.ultimo_area_hoja_pix:.2f}px^2, {area_hoja_cm2:.2f}{self.unidad_equivalencia}^2",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Aplicar transparencia para superponer el texto y contornos
            alpha = 0.25
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Convertir imagen a RGB para mostrarla en Tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Mostrar la imagen actualizada
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)


                                # Muestra los contornos de la hoja
##            overlay = frame.copy()
##            cv2.drawContours(overlay, contornos_hoja, -1, (0, 0, 255), 2)
##            cv2.imshow('Contornos', overlay)

        # Llamar a la actualización en tiempo real cada 10 ms
        self.camera_label.after(10, self.update_camera)




    def detectar_contornos_hoja(self, imagen):
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        _, umbral = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) # Ajustar valor X: (gris, X, 255, ...)
        contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contornos

    def detectar_contornos_linea(self, imagen):
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        gris_blur = cv2.GaussianBlur(gris, (5, 5), 0)
        bordes = cv2.Canny(gris_blur, 30, 150)
        contornos, _ = cv2.findContours(bordes.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contornos

    def ajustar_medida(self):
        opciones = ["1 cm", "1 in", "Personalizado"]
        seleccion = simpledialog.askstring("Medida", "Selecciona una medida o introduce una personalizada:", initialvalue="1 cm")

        if seleccion == "1 cm":
            self.valor_equivalencia = 1.0
            self.unidad_equivalencia = "cm"
        elif seleccion == "1 in":
            self.valor_equivalencia = 2.54
            self.unidad_equivalencia = "in"
        else:
            personalizada = simpledialog.askstring("Medida Personalizada", "Ingresa una medida (ej. 5 cm, 2 in):")
            try:
                valor, unidad = personalizada.split()
                self.valor_equivalencia = float(valor)
                self.unidad_equivalencia = unidad
            except ValueError:
                tk.messagebox.showerror("Error", "Formato no válido. Por favor, ingresa el valor y la unidad.")
                self.ajustar_medida()

    def process_image(self):
        ret, frame = self.camera.read()
        if ret:
            alto, ancho = frame.shape[:2]

            # Dibujar la línea divisoria
            linea_divisoria = 3 * ancho // 4
            cv2.line(frame, (linea_divisoria, 0), (linea_divisoria, alto), (0, 255, 0), 2)

            # Medir la longitud de la línea negra en la cuarta parte derecha
            self.longitud_linea = medir_longitud_linea(cargar_imagen(frame))
            PIXELS_PER_UNIT = self.longitud_linea / self.valor_equivalencia

            # Calcular el área de la hoja en las tres cuartas partes izquierdas
            area_hoja = calcular_area_hoja(frame, PIXELS_PER_UNIT)

            # Configurar el texto de los datos de la hoja y la línea
            texto_area = f"Area real de la Hoja: {area_hoja:.2f} {self.unidad_equivalencia}^2"
            texto_longitud = f"Longitud real de la Linea: {self.longitud_linea:.2f} pixeles ({self.valor_equivalencia} {self.unidad_equivalencia})"

            # Obtener el nombre de la imagen y la marca de tiempo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = f'imagen_hoja_{self.image_counter}_{timestamp}.png'
            texto_nombre = f"Nombre: {image_name}"
            texto_fecha_hora = f"Fecha y Hora: {timestamp}"

            # Añadir espacio adicional para el texto en la parte inferior
            espacio_texto = 120
            nueva_imagen = np.zeros((alto + espacio_texto, ancho, 3), dtype=np.uint8)

            nueva_imagen[:alto, :] = frame

            inicio_texto = alto + 20

            # Añadir el texto en líneas separadas
            cv2.putText(nueva_imagen, texto_area, (10, inicio_texto), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(nueva_imagen, texto_longitud, (10, inicio_texto + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(nueva_imagen, texto_nombre, (10, inicio_texto + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(nueva_imagen, texto_fecha_hora, (10, inicio_texto + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Guardar la imagen con el área de texto adicional
            image_path = os.path.join(self.folder_name, image_name)
            cv2.imwrite(image_path, nueva_imagen)

            # Guardar la información con la unidad de medida para cada imagen
            self.data.append([image_name, area_hoja, image_path, self.unidad_equivalencia])

            self.image_counter += 1
            self.result_label.config(text=f"Longitud de la línea: {self.longitud_linea:.2f} píxeles ({self.valor_equivalencia} {self.unidad_equivalencia}), Área de la hoja: {area_hoja:.2f} {self.unidad_equivalencia}²")
            self.save_excel()

            if self.data_window is not None and self.images_frame is not None:
                self.actualizar_datos_guardados()

    def save_excel(self):
        excel_path = os.path.join(self.folder_name, f'{self.folder_name}.xlsx')
        workbook = xlsxwriter.Workbook(excel_path)
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', 'Nombre de Imagen')
        worksheet.write(f'B1', f'Área de la Hoja ({self.unidad_equivalencia}²)')
        worksheet.write('C1', 'Ruta de Imagen')
        worksheet.write('D1', 'Unidad')
        row = 1
        for image_name, area, path, unidad in self.data:
            worksheet.write(row, 0, image_name)
            worksheet.write(row, 1, area)
            worksheet.write(row, 2, path)
            worksheet.write(row, 3, unidad)
            row += 1
        workbook.close()

    def listar_carpetas(self):
        # Listar todas las carpetas en el directorio actual que parecen sesiones
        return [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('202')]

    def mostrar_datos_guardados(self):
        if self.data_window is None:
            self.data_window = tk.Toplevel(self.root)
            self.data_window.title("Datos Guardados")
            self.images_frame = tk.Frame(self.data_window)
            self.images_frame.pack(pady=10)

            # Añadir la lista de sesiones
            carpetas_sesiones = self.listar_carpetas()
            tk.OptionMenu(self.data_window, self.selected_session, *carpetas_sesiones, command=self.cargar_sesion).pack(pady=10)

            excel_button = tk.Button(self.data_window, text="Abrir Excel", command=self.open_excel)
            excel_button.pack(pady=10)

        self.actualizar_datos_guardados()

    def cargar_sesion(self, session):
        """Cargar la sesión seleccionada desde el menú desplegable"""
        self.selected_session.set(session)
        self.data = []

        # Leer datos de Excel si existe
        excel_path = os.path.join(session, f'{session}.xlsx')
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
            for _, row in df.iterrows():
                self.data.append([row['Nombre de Imagen'], row['Área de la Hoja (cm²)'], row['Ruta de Imagen'], row['Unidad']])

        self.actualizar_datos_guardados()

    def actualizar_datos_guardados(self):
        for widget in self.images_frame.winfo_children():
            widget.destroy()

        for image_name, area, path, unidad in self.data:
            img_label = tk.Label(self.images_frame, text=f"{image_name}: {area:.2f} {unidad}²")
            img_label.pack()

            open_button = tk.Button(self.images_frame, text="Abrir Imagen", command=lambda p=path: self.open_image(p))
            open_button.pack()

    def open_excel(self):
        session = self.selected_session.get()
        excel_path = os.path.join(session, f'{session}.xlsx')
        if os.path.exists(excel_path):
            os.startfile(excel_path)
        else:
            print(f"No se encontró el archivo de Excel: {excel_path}")

    def open_image(self, image_path):
        if os.path.exists(image_path):
            os.startfile(image_path)
        else:
            print("No se encontró la imagen")



if __name__ == "__main__":
    root = tk.Tk()
    app = LeafAreaCameraApp(root)
    root.mainloop()
