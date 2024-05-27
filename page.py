"""
page.py
Este archivo contiene el código para procesar la página de respuestas de un examen.
"""

import cv2
import numpy as np

epsilon = 10
test_sensitivity_epsilon = 10  # margen de error para la detección
choices = ['A', 'B', 'C', 'D', 'E', '?']

# leemos los marcadores de las esquinas
tags = [
    cv2.imread("markers/top_left.png", cv2.IMREAD_GRAYSCALE),
    cv2.imread("markers/top_right.png", cv2.IMREAD_GRAYSCALE),
    cv2.imread("markers/bottom_left.png", cv2.IMREAD_GRAYSCALE),
    cv2.imread("markers/bottom_right.png", cv2.IMREAD_GRAYSCALE)
]

# constantes de escalado específicas de la hoja de prueba
scaling = [605.0, 835.0]  # factor de escala para papel Letter
columns = [
    [72.0 / scaling[0], 33 / scaling[1]],
    [422.0 / scaling[0], 33 / scaling[1]]
]  # dimensiones de las columnas
radius = 10.0 / scaling[0]  # radio de las repuestas
# espaciado de las filas y columnas
spacing = [35.0 / scaling[0], 32.0 / scaling[1]]


def process_page(paper):
    answers = []  # contiene las respuestas
    # convertir la imagen a escala de grises
    gray_paper = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    codes = ["Test", 'Test']

    if (corners := find_corners(paper)) is None:  # encontrar las esquinas del área marcada
        return [-1], paper, [-1]

    # calcular dimensiones para escalar
    dimensions = [corners[1][0] - corners[0][0], corners[2][1] - corners[0][1]]

    # iterar sobre las preguntas del examen
    for k in range(2):  # columnas
        for i in range(25):  # filas
            questions = []
            for j in range(5):  # respuestas
                # coordenadas de la burbuja de respuesta
                x1 = int((columns[k][0] + j * spacing[0] - radius * 1.5)
                         * dimensions[0] + corners[0][0] + 5)
                y1 = int((columns[k][1] + i * spacing[1] - radius)
                         * dimensions[1] + corners[0][1] - 5.2)
                x2 = int((columns[k][0] + j * spacing[0] + radius * 1.5)
                         * dimensions[0] + corners[0][0] + 5)
                y2 = int((columns[k][1] + i * spacing[1] + radius)
                         * dimensions[1] + corners[0][1] - 5.2)

                # dibujar cuadrados alrededor de las burbujas
                cv2.rectangle(paper, (x1, y1), (x2, y2),
                              (255, 0, 0), thickness=1, lineType=8, shift=0)

                # recortar la burbuja de respuesta
                questions.append(gray_paper[y1:y2, x1:x2])

            # encontrar los valores medios de la imagen de las burbujas de respuesta
            means = [np.mean(question) for question in questions]

            # coordenadas para dibujar la respuesta detectada
            x1 = int((columns[k][0] - radius * 8) *
                     dimensions[0] + corners[0][0])
            y1 = int((columns[k][1] + i * spacing[1] + 0.5 *
                     radius) * dimensions[1] + corners[0][1])

            # ordenar por valor mínimo; ordenar por la burbuja más oscura
            min_arg = np.argmin(means)
            min_val = means[min_arg]

            # encontrar el segundo valor más pequeño
            means[min_arg] = 255
            min_val2 = means[np.argmin(means)]

            # verificar si los dos valores más pequeños están cerca en valor
            if min_val2 - min_val < test_sensitivity_epsilon:
                # si es así, entonces la pregunta tiene dos burbujas y es inválida
                min_arg = 5

            # añadir las respuestas al array
            answers.append(choices[min_arg])

    return answers, paper, codes


def find_corners(paper):
    # convertir la imagen de papel a escala de grises
    gray_paper = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)

    if (ratio := len(paper[0]) / 816.0) == 0:
        return -1

    corners = []  # array para contener las esquinas encontradas

    # intentar encontrar las etiquetas mediante la convolución de la imagen
    for tag in tags:
        # redimensionar las etiquetas al ratio de la imagen
        tag = cv2.resize(tag, (0, 0), fx=ratio, fy=ratio)

        # convolucionar la imagen
        convimg = cv2.filter2D(np.float32(cv2.bitwise_not(
            gray_paper)), -1, np.float32(cv2.bitwise_not(tag)))

        # encontrar el máximo de la convolución
        corner = np.unravel_index(convimg.argmax(), convimg.shape)

        # añadir las coordenadas de la esquina
        corners.append([corner[1], corner[0]])

    # dibujar el rectángulo alrededor de los markers detectados
    for corner in corners:
        cv2.rectangle(
            paper,
            (corner[0] - int(ratio * 25), corner[1] - int(ratio * 25)),
            (corner[0] + int(ratio * 25), corner[1] + int(ratio * 25)),
            (0, 255, 0),
            thickness=2,
            lineType=8,
            shift=0
        )

    # verificar si los marcadores detectados forman líneas aproximadamente paralelas al conectarse
    if any(
        abs(corners[i][0] - corners[i + 2][0]) > epsilon
        for i in range(2)
    ) or abs(corners[0][1] - corners[1][1]) > epsilon or abs(corners[2][1] - corners[3][1]) > epsilon:
        return None

    return corners
