import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Cargar los datos de cursos y alumnos desde los archivos CSV
curso_promprof_path = "Modelo-IA/cursos-prom-car.csv"
alumnos_path = "Modelo-IA/datos-alumno.csv"

df_cursos = pd.read_csv(curso_promprof_path)
df_alumnos = pd.read_csv(alumnos_path)

# Definir los universos de discurso y las variables lingüísticas
prom_prof = ctrl.Antecedent(np.arange(0, 21, 1), 'prom_prof')
pond_acumulado = ctrl.Antecedent(np.arange(0, 21, 1), 'pond_acumulado')
nota_esperada = ctrl.Consequent(np.arange(0, 21, 1), 'nota_esperada')

# Definir las membresías para el promedio del profesor
prom_prof['bajo'] = fuzz.trimf(prom_prof.universe, [0, 0, 14])
prom_prof['medio'] = fuzz.trimf(prom_prof.universe, [13, 15, 17])
prom_prof['alto'] = fuzz.trimf(prom_prof.universe, [16, 20, 20])

# Definir las membresías para el promedio ponderado del alumno
pond_acumulado['bajo'] = fuzz.trimf(pond_acumulado.universe, [0, 0, 10])
pond_acumulado['medio'] = fuzz.trimf(pond_acumulado.universe, [8, 11, 14])
pond_acumulado['alto'] = fuzz.trimf(pond_acumulado.universe, [12, 20, 20])

# Definir las membresías para la nota esperada
nota_esperada['bajo'] = fuzz.trimf(nota_esperada.universe, [0, 0, 10])
nota_esperada['medio'] = fuzz.trimf(nota_esperada.universe, [8, 12, 14])
nota_esperada['alto'] = fuzz.trimf(nota_esperada.universe, [12, 20, 20])

# Definir las reglas de inferencia
rule1 = ctrl.Rule(prom_prof['alto'] & pond_acumulado['alto'], nota_esperada['alto'])
rule2 = ctrl.Rule(prom_prof['alto'] & pond_acumulado['medio'], nota_esperada['medio'])
rule3 = ctrl.Rule(prom_prof['alto'] & pond_acumulado['bajo'], nota_esperada['bajo'])
rule4 = ctrl.Rule(prom_prof['medio'] & pond_acumulado['alto'], nota_esperada['medio'])
rule5 = ctrl.Rule(prom_prof['medio'] & pond_acumulado['medio'], nota_esperada['medio'])
rule6 = ctrl.Rule(prom_prof['medio'] & pond_acumulado['bajo'], nota_esperada['bajo'])
rule7 = ctrl.Rule(prom_prof['bajo'] & pond_acumulado['alto'], nota_esperada['medio'])
rule8 = ctrl.Rule(prom_prof['bajo'] & pond_acumulado['medio'], nota_esperada['bajo'])
rule9 = ctrl.Rule(prom_prof['bajo'] & pond_acumulado['bajo'], nota_esperada['bajo'])

# Crear el sistema de control
nota_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
nota_sim = ctrl.ControlSystemSimulation(nota_ctrl)

# Función para calcular la nota esperada
def calcular_nota_esperada(prom_prof_val, pond_acumulado_val):
    nota_sim.input['prom_prof'] = prom_prof_val
    nota_sim.input['pond_acumulado'] = pond_acumulado_val
    nota_sim.compute()
    return nota_sim.output['nota_esperada']

# Crear el DataFrame final fusionando los dos conjuntos de datos
df_final = pd.merge(df_alumnos, df_cursos, left_on='curso-sec', right_on='Curso-Seccion')

# Calcular la nota esperada
df_final['nota-esperada'] = df_final.apply(lambda row: calcular_nota_esperada(row['Promedio'], row['pond-acumulado']), axis=1)

# Seleccionar las columnas deseadas
df_final = df_final[[
    'alumno', 'curso-sec', 'Promedio', 'Calculo', 'Estadistica', 'Fisica', 'Quimica',
    'Programacion', 'GestionEmp', 'Economia', 'area-interes', 'nota-esperada', 'nota-final'
]]

# Guardar el DataFrame final en un nuevo archivo CSV
output_path = "Modelo-IA/data/entrenamiento_final.csv"
df_final.to_csv(output_path, index=False)

# Mostrar las primeras filas del DataFrame final
print(df_final.head())
