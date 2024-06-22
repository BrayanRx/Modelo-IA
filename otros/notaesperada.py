import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Cargar los datos del archivo CSV proporcionado
file_path = "Encuesta docente\curso-promprof.csv"  # Ajusta el nombre del archivo según sea necesario
df_courses = pd.read_csv(file_path)

# Mostrar los primeros registros para verificar la carga
print(df_courses.head())

# Definir las variables lingüísticas y las funciones de pertenencia
grade = ctrl.Antecedent(np.arange(0, 21, 1), 'grade')
evaluation = ctrl.Antecedent(np.arange(0, 21, 1), 'evaluation')
expected_grade = ctrl.Consequent(np.arange(0, 21, 1), 'expected_grade')

grade['low'] = fuzz.trimf(grade.universe, [0, 0, 10])
grade['medium'] = fuzz.trimf(grade.universe, [8, 12, 14])
grade['high'] = fuzz.trimf(grade.universe, [12, 20, 20])

evaluation['low'] = fuzz.trimf(evaluation.universe, [0, 0, 14])
evaluation['medium'] = fuzz.trimf(evaluation.universe, [13, 15, 17])
evaluation['high'] = fuzz.trimf(evaluation.universe, [16, 20, 20])

expected_grade['low'] = fuzz.trimf(expected_grade.universe, [0, 0, 10])
expected_grade['medium'] = fuzz.trimf(expected_grade.universe, [8, 12, 14])
expected_grade['high'] = fuzz.trimf(expected_grade.universe, [12, 20, 20])

# Definir las reglas difusas
rule1 = ctrl.Rule(grade['high'] & evaluation['high'], expected_grade['high'])
rule2 = ctrl.Rule(grade['medium'] & evaluation['medium'], expected_grade['medium'])
rule3 = ctrl.Rule(grade['low'] & evaluation['low'], expected_grade['low'])
rule4 = ctrl.Rule(grade['high'] & evaluation['medium'], expected_grade['high'])
rule5 = ctrl.Rule(grade['medium'] & evaluation['high'], expected_grade['high'])
rule6 = ctrl.Rule(grade['low'] & evaluation['high'], expected_grade['medium'])
rule7 = ctrl.Rule(grade['medium'] & evaluation['low'], expected_grade['low'])

# Crear el sistema de control difuso
expected_grade_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
expected_grade_sim = ctrl.ControlSystemSimulation(expected_grade_ctrl)

# Función para calcular la nota esperada
def calculate_expected_grade(student_grade, course_evaluation):
    expected_grade_sim.input['grade'] = student_grade
    expected_grade_sim.input['evaluation'] = course_evaluation
    expected_grade_sim.compute()
    return expected_grade_sim.output['expected_grade']

# Añadir una columna de ejemplo para las notas de los estudiantes
df_courses['grade'] = np.random.randint(0, 21, size=len(df_courses))

# Aplicar la función a un DataFrame de ejemplo
df_courses['expected_grade'] = df_courses.apply(lambda row: calculate_expected_grade(row['grade'], row['Promedio']), axis=1)

# Mostrar el DataFrame con la columna de notas esperadas
print(df_courses.head())
