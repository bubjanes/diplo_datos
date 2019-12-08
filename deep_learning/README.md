# Aprendizaje Profundo|Diplomatura en ciencias de datos|UNC - FaMAF
## Brandon Janes
## Profesores: Cristian Cardellino y Milagro Teruel

## Práctico n°1
Consigna: Trabajé con el conjunto de datos de petfinder para predecir la velocidad de adopción de un conjunto de mascotas. Para ello, también se dispone de [esta competencia de Kaggle](https://www.kaggle.com/c/diplodatos-deeplearning-2019). Se trata de una tarea de clasificación.

**Data disponible**:

- train.csv 
- test.csv 
- submission.csv - A sample submission file in the correct format para Kaggle
- breed_labels.csv - type y breed name para cada breed ID; ID 1 es dog y 2 es cat
- color_labels.csv - decodificación de colors
- state_labels.csv - decodificación para cada state

**test.csv y train.csv variables**
- PetID (PID) - Unique hash ID of pet profile
- AdoptionSpeed - Categorical speed of adoption. 
- Type - Type of animal (1 = Dog, 2 = Cat)
- Name - Name of pet (Empty if not named)
- Age - Age of pet when listed, in months
- Breed1 - Primary breed of pet (Refer to BreedLabels dictionary)
- Breed2 - Secondary breed of pet, if pet is of mixed breed (Refer to BreedLabels dictionary)
- Gender - Gender of pet (1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets)
- Color1 - Color 1 of pet (Refer to ColorLabels dictionary)
- Color2 - Color 2 of pet (Refer to ColorLabels dictionary)
- Color3 - Color 3 of pet (Refer to ColorLabels dictionary)
- MaturitySize - Size at maturity (1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified)
- FurLength - Fur length (1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified)
- Vaccinated - Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)
- Dewormed - Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)
- Sterilized - Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)
- Health - Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)
- Quantity - Number of pets represented in profile
- Fee - Adoption fee (0 = Free)
- State - State location in Malaysia (Refer to StateLabels dictionary)

**AdoptionSpeed (target)**
- 0 - Pet was adopted on the same day as it was listed.
- 1 - Pet was adopted between 1 and 7 days (1st week) after being listed.
- 2 - Pet was adopted between 8 and 30 days (1st month) after being listed.
- 3 - Pet was adopted between 31 and 90 days (2nd & 3rd month) after being listed.
- 4 - No adoption after 100 days of being listed. (There are no pets in this dataset that waited between 90 and 100 days).

**Objectivos**:

1. Explorar la data y entenderla
2. Entender la funcionamento básico de un clasificador de red neuronal
3. Experimentar y analizar los parámetros y hyperparameters disponibles en Keras/TensorFlow
4. Elegir el modelo (parámetros, hyperparameters...etc) que da el mejor accuracy en predecir el adoption speed de las mascotas. 
5. Hacer un intento en la competición interno de Kaggle

## **¿Qué columnas afecta más el desempeño del clasificador?**

## Baseline trial:
- 2 one-hot-encoded features: gender, color1
- 1 embedded feature: breed1
- 2 numerical features: age, fee
- batch size = 32
- one hidden layer, size = 64

Baseline resultados: **Test loss: 1.78 - accuracy: 0.27**

## Experimentación con parámetros y hyper parámetros:

Después de probar la configuración baseline, fui agregando capas y experimentando agregando columnas para ver que configuración me da el mejor accuracy. 

Probé ```Breed2, Vaccinated, Health, MaturitySize, FurLength, Dewormed``` y ```Sterilized``` como embedded columns y me dio los siguientes resultados:

**Test loss: 1.5414583665221484 - accuracy: 0.347661793231964**

Después probé ```Breed2, Vaccinated, Health, MaturitySize, FurLength, Dewormed``` y ```Sterilized``` como one-hot columns y me dio los siguientes resultados:

**Test loss: 1.519851915871919 - accuracy: 0.34671705961227417**

Como se ve arriba, usando la configuración de one-hot vectors para mis nuevos features agregados me dió accuracy era 44 por ciento y el validation accuracy 35 por ciento una diferencia de casi 10 por ciento. Esto indica overfitting, que es un problema muy común cuando usas redes neuronales.

Despues agregé una capa oculta adicional de 20 nodos y hice ```kernal_regularizer=regularizers.I2(0.001),activity_regularizer=regularizer.I2(0.001)``` para optimizar los parámetros de las capas ocultas. También agregué ```StandardScaler``` para poder introducir scaling para varianza. 

Las únicas métricas que usé en este proyecto era accuracy y validation accuracy. 

## Final configuración:
- 30 epochs
- batch size = 32
- 9 one-hot-encoded features: Gender, Color1, Breed2, Vaccinated, Health, MaturitySize, FurLength, Dewormed, Sterilized
- 1 embedded feature: breed1
- 2 numerical features: age, fee
- 2 hidden layers (64 nodos, 20 nodos)
- Softmax activation funcion
- scaling with ```Sklearn StandardScaler```
- regularize with ```Keras regularizers```

**Conclusión:**
Sobre todo estuve un poco decepcionado con mis resultados tratando de predecir el adoption speed. Después de probar varios configuraciones de parámetros y hyper parametros, lo encontré difícil generar resultados significamente mejor. 

## Pendiente para trabajos futuros:
- Explorar variaciones en batch size
- Probar otros activation functions
- Usar redes recurrentes para analizar el texto de description 
- usar diferentes metricas para evaluar mi modelo
