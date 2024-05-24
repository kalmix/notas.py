import numpy as np
import cv2
import streamlit as st
from page import ProcessPage

# Encabezado de la página (md)
st.markdown("# Sistema De Calificación")
st.markdown("Sube una imagen del documento con las respuestas correctas y luego un documento de respuestas a corregir. Una vez subidos los archivos, el sistema mostrará el puntaje obtenido por el estudiante, deslice hacia abajo para ver los resultados.")

# File uploaders
correct_answers_file = st.file_uploader("Selecciona el archivo con las respuestas correctas", type=["jpg", "jpeg", "png"], key="correct_answers_file")
uploaded_file = st.file_uploader("Selecciona un documento a corregir", type=["jpg", "jpeg", "png"], key="answer_sheet")

# Estado de la sesión para guardar las respuestas correctas
if 'correct_answers' not in st.session_state:
    st.session_state['correct_answers'] = None

# Clear session state variables if new files are uploaded
if st.session_state.get('uploaded_correct_answers', False) and not correct_answers_file:
    st.session_state['correct_answers'] = None
if st.session_state.get('uploaded_answer_sheet', False) and not uploaded_file:
    st.session_state['answers'] = None
    st.session_state['paper'] = None

def sort_clockwise(points):
    """
    Ordena los puntos en sentido horario.

    Parámetros:
        points (list): Lista de puntos a ordenar.

    Returns:
        list: Lista de puntos ordenados en sentido horario.

    """
    mx = sum(point[0] for point in points) / 4
    my = sum(point[1] for point in points) / 4
    return sorted(points, key=lambda x: (np.arctan2(x[0] - mx, x[1] - my) + 0.5 * np.pi) % (2 * np.pi), reverse=True)


def process_image(image_file):
    """
    Procesa una imagen de un documento de respuestas. Extrae las respuestas, corrige la perspectiva.

    Parámetros:
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
        st.error("No se pudo detectar un contorno válido en la imagen.")
        return None, None, None

    points = sort_clockwise(points)
    points = np.float32(points) * ratio
    M = cv2.getPerspectiveTransform(points, desired_points)
    paper = cv2.warpPerspective(original_image, M, (425, 550))
    answers, paper, codes = ProcessPage(paper)
    return answers, paper, codes


# Process uploaded correct answers file
if correct_answers_file and st.session_state['correct_answers'] is None:
    answers, paper, _ = process_image(correct_answers_file)
    if answers is not None and answers != -1:
        st.session_state.correct_answers = answers
        st.session_state['uploaded_correct_answers'] = True
        st.image(paper, caption="Documento con respuestas correctas", use_column_width=True)
        st.success("Respuestas correctas cargadas correctamente.")
    else:
        st.error("Error al procesar el documento con respuestas correctas.")

# Process uploaded answer sheet file
if uploaded_file and st.session_state['correct_answers'] is not None:
    answers, paper, _ = process_image(uploaded_file)
    if answers is not None and answers != -1:
        def compare_answers(student_answers, correct_answers):
            if len(student_answers) != len(correct_answers):
                raise ValueError("Las listas de respuestas tienen longitudes diferentes")
            return sum(1 for s, c in zip(student_answers, correct_answers) if s == c)

        st.session_state['answers'] = answers
        st.session_state['paper'] = paper
        st.session_state['uploaded_answer_sheet'] = True

        st.image(paper, caption="Documento escaneado", use_column_width=True)
        st.write(f"Respuestas: {answers}")
        st.write(f"Respuestas correctas: {st.session_state['correct_answers']}")
        st.write(f"Puntaje: {compare_answers(answers, st.session_state['correct_answers'])}")
    else:
        st.error("Error al procesar el documento a corregir.")
elif uploaded_file and st.session_state['correct_answers'] is None:
    st.error("Primero sube el documento con las respuestas correctas.")
