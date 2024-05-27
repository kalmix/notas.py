"""
app.py
Este archivo contiene el código principal de la aplicación.
También se encarga de procesar las imágenes de los documentos de respuestas.

"""

import os
import time
import pandas as pd
import numpy as np
import cv2
import pytesseract
import streamlit as st
from page import process_page
from constants import DESCRIPTION

# INSTALA pytesseract y pon la ruta de tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Guardar los datos de los estudiantes en un archivo CSV
def save_to_csv(data, filename="student_data.csv"):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

# Cargar los datos de los estudiantes desde un archivo CSV
def load_from_csv(filename="student_data.csv"):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    return pd.DataFrame(columns=["Nombre", "Puntaje", "Porcentaje"])


# Sidebar
st.sidebar.title("Menú")
page = st.sidebar.radio("Opciones", ["Inicio", "Logs"])

# Guardar los estados de la sesión.
# if 'correct_answers' not in st.session_state:
#     st.session_state['correct_answers'] = None
# if 'uploaded_correct_answers' not in st.session_state:
#     st.session_state['uploaded_correct_answers'] = False
# if 'answers' not in st.session_state:
#     st.session_state['answers'] = []
# if 'papers' not in st.session_state:
#     st.session_state['papers'] = []
# if 'uploaded_answer_sheets' not in st.session_state:
#     st.session_state['uploaded_answer_sheets'] = False
# if 'uploaded_correct_answers_file' not in st.session_state:
#     st.session_state['uploaded_correct_answers_file'] = None
# if 'uploaded_files' not in st.session_state:
#     st.session_state['uploaded_files'] = None
# if 'results' not in st.session_state:
#     st.session_state['results'] = []

default_values = {
    'correct_answers': None,
    'uploaded_correct_answers': False,
    'answers': [],
    'papers': [],
    'uploaded_answer_sheets': False,
    'uploaded_correct_answers_file': None,
    'uploaded_files': None,
    'results': []
}

# Initialize session state keys with default values if they do not exist
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value


def sort_clockwise(points):
    mx = sum(point[0] for point in points) / 4
    my = sum(point[1] for point in points) / 4
    return sorted(points, key=lambda x: (np.arctan2(x[0] - mx, x[1] - my) + 0.5 * np.pi) % (2 * np.pi), reverse=True)


def process_image(image_file):
    """
    Procesa una imagen de un documento de respuestas.
    Para ello se utiliza la detección de contornos. 
    Revise el archivo page.py para más detalles de la función process_page.

    @Parámetros:
        image_file (file): La imagen del documento de respuestas. (file object)

    Returns:
        tuple: La tupla contiene los siguientes elementos:
            - answers (list): Respuestas extraídas.
            - paper (array): Imagen con la perspectiva corregida.
    """
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
    ratio = len(image[0]) / 500.0
    original_image = image.copy()
    image = cv2.resize(image, (0, 0), fx=1 / ratio, fy=1 / ratio)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 250, 300)
    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    biggest_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            biggest_contour = approx
            break

    points = []
    desired_points = np.float32([[0, 0], [425, 0], [425, 550], [0, 550]])

    if biggest_contour is not None:
        for i in range(0, 4):
            points.append(biggest_contour[i][0])

    if len(points) != 4:
        test = process_page(image)
        st.info("No se detectó un contorno válido en la imagen. Se procederá a procesar la imagen.")
        #st.error("No se pudo detectar un contorno válido en la imagen.")
        #st.image(original_image, caption="Documento con respuestas")
        return test
        #return None, None, None

    points = sort_clockwise(points)
    points = np.float32(points) * ratio
    M = cv2.getPerspectiveTransform(points, desired_points)

    # Corregir la perspectiva de la imagen
    paper = cv2.warpPerspective(original_image, M, (425, 550))
    answers, paper, codes = process_page(paper)
    return answers, paper, codes


def compare_answers(student_answers, correct_answers):
    if student_answers is None or correct_answers is None:
        st.error("Las listas de respuestas están vacías")
        return 0
    if len(student_answers) != len(correct_answers):
        raise ValueError(
            "Las listas de respuestas tienen longitudes diferentes")
    return sum(1 for s, c in zip(student_answers, correct_answers) if s == c)

# Devuelve un mensaje toast
def toasty():
    st.toast("Respuestas correctas cargadas correctamente.", icon="✅")
    time.sleep(2)


def extract_name(image):
    """
    Extrae el nombre de la sección de encabezado del documento utilizando OCR.
    Básicamente lee el texto en la parte superior de la imagen (10% de la altura).
    Luego se pasa a escala de grises y se aplica OCR. Y finalmente se limpia el texto.

    Parámetros:
        image (array): La imagen del documento.

    Returns:
        str: El nombre detectado en la imagen.
    """
    height, width = image.shape[:2]
    header_section = image[:int(0.1 * height), :]
    gray_header = cv2.cvtColor(header_section, cv2.COLOR_BGR2GRAY)
    name = pytesseract.image_to_string(gray_header)
    name = name.replace("\n", " ").replace("\x0c", "").replace(
        "Matricula", "").replace("Nombre", "").replace("—", "")
    return name.strip()


# Home Page
if page == "Inicio":
    # st.markdown("# 🧪 Sistema De Calificación")
    st.markdown(
        '![Icon](https://ucarecdn.com/e5172373-21f1-422e-9287-f0771f429435/-/preview/100x100/)')
    st.markdown(DESCRIPTION)
    st.markdown("`Para reinicar la sesión, presiona Ctlr + R`")
    st.markdown("---")

    # File uploaders
    correct_answers_file = st.file_uploader("Selecciona el archivo con las respuestas correctas", type=[
                                            "jpg", "jpeg", "png"], key="correct_answers_file")
    uploaded_files = st.file_uploader("Selecciona uno o varios documentos a corregir", type=[
                                      "jpg", "jpeg", "png"], key="answer_sheet", accept_multiple_files=True)

    # Resetear la sesión si se sube un nuevo archivo de respuestas correctas
    if correct_answers_file is None and st.session_state['uploaded_correct_answers_file'] is not None:
        st.session_state['uploaded_correct_answers'] = False
        st.session_state['uploaded_correct_answers_file'] = None
        st.session_state['correct_answers'] = None
        st.session_state['correct_answers_paper'] = None

    if uploaded_files is None and st.session_state['uploaded_files'] is not None:
        st.session_state['uploaded_files'] = None
        st.session_state['results'] = []

    # Procesa el archivo con las respuestas correctas
    if correct_answers_file and not st.session_state['uploaded_correct_answers']:
        st.session_state.uploaded_correct_answers_file = correct_answers_file
        answers, paper, _ = process_image(correct_answers_file)
        if answers is not None and answers != -1:
            st.session_state.correct_answers = answers
            st.session_state['uploaded_correct_answers'] = True
            st.session_state['correct_answers_paper'] = paper
            st.image(paper, caption="Documento con respuestas correctas",
                     use_column_width=True)
            toasty()
            st.write(f"Respuestas correctas: {answers}")
        else:
            st.error("Error al procesar el documento con respuestas correctas.")

    elif st.session_state['uploaded_correct_answers'] and st.session_state['correct_answers_paper'] is not None:
        st.image(st.session_state['correct_answers_paper'],
                 caption="Documento con respuestas correctas", use_column_width=True)
        st.write(
            f"Respuestas correctas: {st.session_state['correct_answers']}")

    # Procesa los archivos de respuestas de los estudiantes
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.session_state.results = []
        for uploaded_file in uploaded_files:
            answers, paper, _ = process_image(uploaded_file)
            if answers is not None and answers != -1 and st.session_state['correct_answers'] is not None:
                name = extract_name(paper)
                score = compare_answers(
                    answers, st.session_state['correct_answers'])
                percentage = (
                    score / len(st.session_state['correct_answers'])) * 100
                trimmed_percentage = f"{percentage:.1f}"
                st.session_state.results.append(
                    (paper, answers, score, trimmed_percentage, name))
            else:
                st.session_state.results.append(
                    (None, None, None, "Error", None))

    if st.session_state.results:
        student_data = []
        st.session_state['answers'] = [result[1]
                                       for result in st.session_state.results]
        st.session_state['papers'] = [result[0]
                                      for result in st.session_state.results]
        st.session_state['uploaded_answer_sheets'] = True

        cols = st.columns(3)
        for idx, (paper, answers, score, percentage, name) in enumerate(st.session_state.results):
            if paper is not None:
                student_data.append(
                    {"Nombre": name, "Puntaje": score, "Porcentaje": percentage})
                with cols[idx % 3]:
                    st.image(
                        paper, caption=f"Documento {idx+1}", use_column_width=True)
                    if name == "":
                        name = "?"
                    st.write(f"Matricula: {name}")
                    st.write(f"Respuestas: {answers}")
                    st.write(f"Puntaje: {score}")
                    st.metric("Porcentaje", f"{percentage}%", "0%")
            else:
                st.error(
                    f"Error al procesar el documento {idx+1}. Por favor, sube un documento válido de respuestas.")

        # Exporta los datos de los estudiantes a un archivo CSV
        save_to_csv(student_data)

        df = pd.DataFrame(student_data)

        st.markdown("### Resultados de los estudiantes")
        if not df.empty:
            st.bar_chart(df.set_index("Nombre")["Puntaje"], color='#ff4b4b')

# Logs Page
elif page == "Logs":
    st.markdown(
        "![Hola](https://ucarecdn.com/93d5a051-a5a1-414e-ba26-9daaa72df335/icons8estadsticas96.png)")
    df = load_from_csv()
    # st.button("Limpiar la tabla de datos", on_click=save_to_csv(pd.DataFrame(columns=["Nombre", "Puntaje", "Porcentaje"])))

    if not df.empty:
        st.markdown("### Datos de los estudiantes")
        edited_df = st.data_editor(df, num_rows="dynamic")

        # Check if the DataFrame has been edited
        if not edited_df.equals(df):
            save_to_csv(edited_df)
            df = edited_df  # Update the DataFrame with the edited data

        st.markdown("### Estadísticas generales")
        average_score = df["Puntaje"].mean()
        max_score = df["Puntaje"].max()
        min_score = df["Puntaje"].min()

        st.write(f"Puntaje promedio: {average_score:.2f}")
        st.write(f"Puntaje máximo: {max_score}")
        st.write(f"Puntaje mínimo: {min_score}")

        st.markdown("### Distribución de puntajes")
        st.bar_chart(df["Puntaje"], color='#ff4b4b')

        st.markdown("### Porcentaje de puntajes")
        st.line_chart(df["Porcentaje"], color='#ff4b4b')

        st.markdown("### Distribución de nombres")
        name_counts = df["Nombre"].value_counts()
        st.bar_chart(name_counts, color='#ff4b4b')
    else:
        st.info("No hay datos disponibles. Suba documentos para generar estadísticas.")
