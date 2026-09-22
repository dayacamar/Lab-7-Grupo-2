# Carga de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib as Path
import seaborn as sns 

# Carga de datos
df = pd.read_csv("smart_workout_raw_dataset.csv")

# Dimensiones del DataFrame
df.shape

# Impresión de las primeras filas
df.head()

# Identificación del tipo de cada variable
df.info()

# Creando nuevas variables con valores en minúsculas y sin espacios vacíos
categorical_columns = [
"persona",
"fitness_goal",
"experience_level",
"injury_constraint",
"bodyPart",
"equipment",
"target_muscle"
]
for column in categorical_columns:
    df[column + "_std"] = (
    df[column]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    
df[["fitness_goal", "fitness_goal_std"]].drop_duplicates().sort_values(
"fitness_goal_std"
)

df_usuario = (
df.groupby("user_id")
.agg(
age = ("age", "median"),
n_records = ("user_id", "size"),
n_exercises = ("exercise_id", "nunique"),
n_bodyparts = ("bodyPart_std", "nunique"),
n_equipment = ("equipment_std", "nunique"),
n_target_muscles = ("target_muscle_std", "nunique"),
mean_rating = ("rating", "mean"),
median_rating = ("rating", "median"),
sd_rating = ("rating", "std")
)
.reset_index()
)

# Dimensiones del nuevo DataFrame
df_usuario.shape
# Impresión de las primeras filas
df_usuario.head()

# Calculando el conteo de objetivos por usuario
conteo_objetivos = pd.crosstab(
df["user_id"],
df["fitness_goal_std"]
)

# Añadiendo al nombre de cada variable el prefijo n_goal_
conteo_objetivos = conteo_objetivos.add_prefix("n_goal_")

# Calculando el conteo de partes del cuerpo ejercitadas por usuario
conteo_bodyparts = pd.crosstab(
df["user_id"],
df["bodyPart_std"]
)
# Añadiendo al nombre de cada variable el prefijo n_body_
conteo_bodyparts = conteo_bodyparts.add_prefix("n_body_")


# Agregando variables que cuentan objetivos
df_usuario = df_usuario.merge(
conteo_objetivos,
on = "user_id",
how = "left"
)

# Agregando variables que cuentan partes del cuerpo ejercitadas
df_usuario = df_usuario.merge(
conteo_bodyparts,
on="user_id",
how="left"
)
# Proporción de cantidad de registros en general fitness
df_usuario["share_goal_general_fitness"] = (
df_usuario["n_goal_general_fitness"]
/ df_usuario["n_records"]
)

# Proporción de cantidad de registros en endurance
df_usuario["share_goal_endurance"] = (
df_usuario["n_goal_endurance"]
/ df_usuario["n_records"]
)

df_usuario["share_goal_fat_loss"] = (
df_usuario["n_goal_fat_loss"]
/ df_usuario["n_records"]
)

df_usuario["share_goal_mobility"] = (
df_usuario["n_goal_mobility"]
/ df_usuario["n_records"]
)

df_usuario["share_goal_muscle_gain"] = (
df_usuario["n_goal_muscle_gain"]
/ df_usuario["n_records"]
)



df_usuario["user_id"].nunique()

df_usuario.shape[0]

df_usuario["user_id"].duplicated().sum()

##pregunta 4
edad_entrenamientos_resistencia = df_usuario["age"].corr(df_usuario["n_goal_endurance"])
valoracion_cantidad_musculos = df_usuario["mean_rating"].corr(df_usuario["n_body_cardio"])
valoracion_perida_grasa = df_usuario["mean_rating"].corr(df_usuario["n_goal_general_fitness"])

df_fitness_goal = (df_usuario["n_goal_general_fitness"] == 0.0) | (df_usuario["n_goal_general_fitness"] == 0.0)
#print(df_fitness_goal)

#print("print")
#print(edad_entrenamientos_resistencia)
#print(valoracion_cantidad_musculos)
#print(valoracion_perida_grasa)

## pregunta 3
df_usuario["mean_rating"].plot(kind="box")
plt.show()

## mean rating
mas_bajo = df_usuario["mean_rating"].min()
print("Promedio más bajo:", mas_bajo)
mas_alto = df_usuario["mean_rating"].max()
print("Promedio más alto:", mas_alto)
df_mean_bajo = df_usuario[df_usuario['mean_rating'] == mas_bajo]
df_solo_rating = df_mean_bajo[['user_id', 'mean_rating']]
print(df_solo_rating)

## n records
mayores_records = df_usuario.sort_values(by="n_records", ascending=False).head()
df_mayores_records = mayores_records[["user_id", "n_records", "mean_rating"]]
print(df_mayores_records)

print("si")

# Punto 2
# Diagrama de dispersión para edad y cantidad de ejercicios de perdida de grasa

relacion_1 = df_usuario[["age", "n_goal_fat_loss"]]
relacion_1.plot(kind="scatter", x="age", y="n_goal_fat_loss", xlabel="Edad", ylabel="Cantidad de ejercicios para la perdida de grasa", title="Relación edad - cantidad de ejercicios para la perdida de grasa")
plt.show()

# Coeficiente de correlación 1:
corr1 = df_usuario["age"].corr(df_usuario["n_goal_fat_loss"])
print(f"Coeficiente de correlación 1: {corr1}")

# Diagrama de dispersión para valoración promedio y número de ejercicios de cardio
relacion_2 = df_usuario[["mean_rating", "n_body_cardio"]]
relacion_2.plot(kind="scatter", x="mean_rating", y="n_body_cardio", xlabel="Valoración promedio (0.0 - 5.0)", ylabel="Cantidad de ejercicios de cardio", title="Relación valoración promedio - cantidad de ejercicios de cardio")
plt.show()

#Coeficiente de correlación 2:
corr2 = df_usuario["mean_rating"].corr(df_usuario["n_body_cardio"])
print(f"Coeficiente de correlación 2: {corr2}")

# Diagrama de dispersión para valoración promedio y cantidad de ejercicios de condición física general
relacion_3 = df_usuario[["mean_rating", "n_goal_general_fitness"]]
relacion_3.plot(kind="scatter", x="mean_rating", y="n_goal_general_fitness", xlabel="Valoración promedio (0.0 - 5.0)", ylabel="Cantidad de ejercicios para acondicionamiento físico", title="Relación valoración promedio \n - Cantidad de ejercicios para acondicionamiento físico")
plt.show()

#Coeficiente de correlación 3:
corr3 = df_usuario["mean_rating"].corr(df_usuario["n_goal_general_fitness"])
print(f"Coeficiente de correlación 3: {corr3}")

#Pregunta 5
# Correlación de la tercera variable (n_bodyparts) con cada variable original
corr_bodyparts_rating = df_usuario["n_bodyparts"].corr(df_usuario["mean_rating"])
corr_bodyparts_goal = df_usuario["n_bodyparts"].corr(df_usuario["n_goal_general_fitness"])

print(f"Correlación de la tercera variable")
print(f"n_bodyparts vs mean_rating: {corr_bodyparts_rating}")
print(f"n_bodyparts vs n_goal_general_fitness: {corr_bodyparts_goal}")
