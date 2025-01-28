import kagglehub as kg
import os
#Desactivar algunas advertencias de tensorflow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 
import tensorflow as tf
import numpy as np
from keras import models
from keras import layers
from keras import Sequential
from keras.api.callbacks import ModelCheckpoint
import pathlib 
import os
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox 
import re

def main(): 
    # descargarDataset() 
    # optimizarPipeline()
    # normalizarDatasetEntrenamiento()
    # normalizarDatasetValidacion()
    # entrenarModelo()
    # medirMetricasModeloEntrenado()
    crearInterfaz()

# Esta funcion solamente es necesaria ejecutarla una vez al inicio del proyecto
# Se encarga de descargar el dataset de la pagina de Kaggle
def descargarDataset():
    ruta = kg.dataset_download("markdaniellampa/fish-dataset")

    print("Ruta donde se descargo el dataset:", ruta)

def obtenerRutaDatasetEntrenamiento():
    
    rutaDataset = str(pathlib.Path.cwd()) + "/FishImgDataset/train"
    return rutaDataset

def obtenerRutaDatasetValidacion():
    rutaDataset = str(pathlib.Path.cwd()) + "/FishImgDataset/val"
    return rutaDataset

def obtenerRutaDatasetPruebas():
    rutaDataset = str(pathlib.Path.cwd()) + "/FishImgDataset/test"
    return rutaDataset

def obtenerTamanioLote():
    tamanioLote = 64
    return tamanioLote

def obtenerAltoImagen():
    altoImagen = 128
    return altoImagen

def obtenerAnchoImagen():
    anchoImagen = 128
    return anchoImagen

def cargarDatasetEntrenamiento():

    #Se crea el dataset para entrenamiento
    datasetEntrenamiento = tf.keras.utils.image_dataset_from_directory(
    obtenerRutaDatasetEntrenamiento(),
    seed=123,
    image_size=(obtenerAltoImagen(), obtenerAnchoImagen()),
    batch_size=obtenerTamanioLote())

   
    return datasetEntrenamiento
   


def cargarDatasetValidacion():
    #Se crea el dataset para validacion
    datasetValidacion = tf.keras.utils.image_dataset_from_directory(
    obtenerRutaDatasetValidacion(),
    seed=123,
    image_size=(obtenerAltoImagen(), obtenerAnchoImagen()),
    batch_size=obtenerTamanioLote())
    return datasetValidacion

def cargarDatasetPruebas():
    #Se crea el dataset para validacion
    datasetPruebas = tf.keras.utils.image_dataset_from_directory(
    obtenerRutaDatasetPruebas(),
    image_size=(obtenerAltoImagen(), obtenerAnchoImagen()),
    batch_size=obtenerTamanioLote())
    return datasetPruebas



def optimizarPipeline():
    # Creamos un objeto AUTOTONE para mejorar el tiempo de ejecucion del modelo
    # Ya que mejora el paralelismo y la carga de datos, es decir tensorflow se encarga de manejarlo
    AUTOTUNE = tf.data.AUTOTUNE

    datasetEntrenamiento = cargarDatasetEntrenamiento()
    datasetValidacion = cargarDatasetValidacion()
    
    # Esto se encarga de cargar el dataset en memoria, revolverlo 
    datasetEntrenamiento = datasetEntrenamiento.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    # Esto se encarga de cargar el dataset en memoria y revolverlo 
    datasetValidacion = datasetValidacion.cache().prefetch(buffer_size=AUTOTUNE)


def normalizarDatasetEntrenamiento():
    datasetEntrenamiento = cargarDatasetEntrenamiento()
    capaNormalizacion = layers.Rescaling(1./255)

    datasetEntrenamiento = datasetEntrenamiento.map(lambda x, y: (capaNormalizacion(x), y))

def normalizarDatasetValidacion():
    datasetValidacion = cargarDatasetValidacion()
    capaNormalizacion = layers.Rescaling(1./255)

    datasetValidacion = datasetValidacion.map(lambda x, y: (capaNormalizacion(x), y))






def crearModelo():
    datasetEntrenamiento = cargarDatasetEntrenamiento()
    cantidadClases = len(datasetEntrenamiento.class_names)

    

    # Se crean nuevas imagenes, utilizando un orden secuencial
    # Es decir sigue el siguiente orden:
    aumento_imagenes = Sequential(
    [
       
        # Vueltas horizontales aleatoriamente 
        layers.RandomFlip("horizontal",
                        input_shape=(obtenerAltoImagen(),
                                     obtenerAnchoImagen(),
                                    3)),
        # Rotacion aleatoria                            
        layers.RandomRotation(0.1),
        # Zoom aleatorio
        layers.RandomZoom(0.1),
        # Vueltas verticales aleatoriamente
        layers.RandomFlip("vertical",
                        input_shape=(obtenerAltoImagen(),
                                     obtenerAnchoImagen(),
                                    3)),
    ]
    )



    # Se construye el modelo secuancialmente es decir funcionara como una pila 
    modelo = Sequential([
        layers.InputLayer(input_shape=(obtenerAltoImagen(), obtenerAnchoImagen(), 3)), #Capa de entrada con el alto-ancho-canales-color de la imagen
        aumento_imagenes, # Se pasa como parametro mas imagenes creadas
        layers.Rescaling(1./255), # Normalizar la imagen a un rango de 0-1, para los colores RGB 
        # Padding="same": significa que la imagen mantenga su mismo tamaño
        # Activation="relu": se encarga de aplicar una funcion no lineal a la capa
        # Se utiliza Conv2D por que una imagen es una estructura bidimensional, es decir tiene ancho y largo
        # Se utiliza MaxPooling2D unicamente para extraer las caracteristicas mas importantes de la capa
        layers.Conv2D(16, 3, padding="same", activation="relu"), # Primera capa oculta convulacional con 16 filtros, se encarga de encontrar caracteristicas en la imagen
        layers.MaxPooling2D(), # Primera capa de Pooling la cual reduce el tamaño de la imagen, manteniendo caracteristicas esenciales
        layers.Conv2D(32, 3, padding="same", activation="relu"), # Segunda capa oculta convulacional con 32 filtros, se encarga de encontrar caracteristicas en la imagen
        layers.MaxPooling2D(), # Segunda capa de Pooling la cual reduce el tamaño de la imagen, manteniendo caracteristicas esenciales
        layers.Conv2D(64, 3, padding="same", activation="relu"), # Tercera capa oculta convulacional con 64 filtros, se encarga de encontrar caracteristicas en la imagen
        layers.MaxPooling2D(),  # Tercera capa de Pooling la cual reduce el tamaño de la imagen, manteniendo caracteristicas esenciales
        layers.Conv2D(128, 3, padding="same", activation="relu"), # Cuarta capa oculta convulacional con 128 filtros, se encarga de encontrar caracteristicas en la imagen
        layers.MaxPooling2D(),   # Cuarta capa de Pooling la cual reduce el tamaño de la imagen, manteniendo caracteristicas esenciales
        layers.Conv2D(256, 3, padding="same", activation="relu"),  # Quinta capa oculta convulacional con 256 filtros, se encarga de encontrar caracteristicas en la imagen
        layers.MaxPooling2D(),  # Quinta capa de Pooling la cual reduce el tamaño de la imagen, manteniendo caracteristicas esenciales
        layers.Conv2D(512, 3, padding="same", activation="relu"),  # Sexta capa oculta convulacional con 512 filtros, se encarga de encontrar caracteristicas en la imagen
        layers.MaxPooling2D(),   # Sexta capa de Pooling la cual reduce el tamaño de la imagen, manteniendo caracteristicas esenciales
        layers.Flatten(), # Capa encargada de convertir los datos en un arreglo de una dimension
        layers.Dense(512, activation="relu"),  # Primera capa densa con 512 neuronas
        layers.Dropout(0.3), # Apague aleatoriamente 30% de las neuronas
        layers.Dense(256, activation="relu"), # Segunda capa densa con 256 neuronas
        layers.Dropout(0.2), # Apague aleatoriamente 20% de las neuronas
        layers.Dense(128, activation="relu"),  # Tercera capa densa con 128 neuronas
        layers.Dropout(0.1), # Apague aleatoriamente 10% de las neuronas
        layers.Dense(64, activation="relu"),  # Cuarta capa densa con 64 neuronas
        layers.Dense(cantidadClases, name="Salidas")
    ])


    modelo.compile(optimizer="adam",
              # Como funcion de perdida se utiliza SparseCategoricalCrossentropy
              # La cual es adecuada para problemas de que se debe predecir una unica
              # clases para multiple variedad de clases, ademas nuestras etiquetas son numeros enteros, no vectores
              # From_Logits=True significa que el modelo va a retornar resultados no normalizados, que deben ser
              # normalizados con softmax al momento de predicir una imagen 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              # Como metricas tambien el accuracy 
              metrics=["accuracy"])

    # Se imprime el resumen del modelo
    modelo.summary()
    return modelo

def obtenerClases(datasetEntrenamiento):
    clases = datasetEntrenamiento.class_names
    return clases

# Esta funcion solamente es necesaria llamarla una vez, para entrenar el modelo y guardarlo
def entrenarModelo():
    repeticiones = 1000

    modelo = crearModelo()
    datasetEntrenamiento = cargarDatasetEntrenamiento()
    datasetValidacion = cargarDatasetValidacion()
   
    # Se guarda unicamente el mejor modelo
    mejorModelo = ModelCheckpoint(
        # Nombre del modelo
        "modeloPeces.keras",
        # Lo que se va a monitorear           
        monitor="val_accuracy",    
        # Unicamente el maximo  
        mode="max",           
        # Que se guarde el mejor solamente            
        save_best_only=True,           
        # Que los logs sean especificos
        verbose=1                         
    )

    #Se entrena el modelo con el dataset de entrenamiento, el dataset de validacion, las repeticiones y el callback
    historial = modelo.fit(
        datasetEntrenamiento,
        validation_data=datasetValidacion,
        epochs=repeticiones,
        callbacks=[mejorModelo]
    )


    precision = historial.history["accuracy"]

    precisionValidacion = historial.history["val_accuracy"]
 
    perdida = historial.history["loss"]
   
    perdidaValidacion = historial.history['val_loss']

    rangoRepeticiones = range(repeticiones)
  
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(rangoRepeticiones, precision, label="Precision entrenamiento")
    plt.plot(rangoRepeticiones, precisionValidacion, label="Precision Validacion")
    plt.legend(loc="lower right")
    plt.title("Precision de entrenamiento y validacion")
  
    plt.subplot(1, 2, 2)
    plt.plot(rangoRepeticiones, perdida, label="Perdida entrenamiento")
    plt.plot(rangoRepeticiones, perdidaValidacion, label="Perdida validacion")
    plt.legend(loc="upper right")
    plt.title("Perdida de entrenamiento y validacion")
    plt.show()


 

def medirMetricasModeloEntrenado():
    
    modelo = models.load_model("modeloPeces.keras")
    
    datasetPruebas = cargarDatasetPruebas()




   

    resultados = modelo.evaluate(datasetPruebas)

   

   
    # La precision mide la exactitud de las predicciones del modelo
    print("Precision",  (resultados[1]*100))
    # Perdida significa que tan lejos estuvo el modelo de predecir la clase correcta
    # Como este es un caso de clasificacion mide cuan lejos estuvo la clase predicha de la clase verdadera
    print("Perdida: ", resultados[0])


   

def predicirPez(rutaImagen):
    modelo = models.load_model("modeloPeces.keras")
   
    
    datasetEntrenamiento = cargarDatasetEntrenamiento()

    clases = obtenerClases(datasetEntrenamiento)

  

    # Se carga la imagen
    imagen = tf.keras.utils.load_img(
        rutaImagen, target_size=(obtenerAltoImagen(), obtenerAnchoImagen())
    )   

    

    # Se convierte la imagen a un arreglo de NumPy
    arregloImagen = tf.keras.utils.img_to_array(imagen)

 
    # Se expanden las dimensiones de la imagen, es decir se agrega una dimension al inicio, esto para que el lote de imagenes que el modelo
    # vaya a predecir sea 1, es decir solamente una imagen
    arregloImagen = tf.expand_dims(arregloImagen, 0) 
  
    # Se hace la prediccion de la imagen 
    prediccion = modelo.predict(arregloImagen)
    # La unica fila, ya que es solo una imagen, que retorna la prediccion es la que que contiene las puntuaciones 
    # Se utiliza softmax para la prediccion por que es un problema de clasificacion multiclase
    puntuacion = tf.nn.softmax(prediccion[0])

    puntacionPorcentaje = 100 * np.max(puntuacion)

    if puntacionPorcentaje < 50:
        messagebox.showinfo(           
            "Prediccion Hecha con Exito",                    
            "No se ha podido identificar la especie de la imagen"
        )
    
    if puntacionPorcentaje >= 50:
        messagebox.showinfo(           
            "Prediccion Hecha con Exito",                    
            #np.argmax retorna el indice el elemento con mayor puntuacion
            f"Esta imagen pertenece a la clase {clases[np.argmax(puntuacion)]} con un porcentaje del {puntacionPorcentaje:.2f}"
        )

  
def buscarArchivos():
    rutaImagen = filedialog.askopenfilename(initialdir = "/",
                                          title = "Seleccione una imagen"
                                        )

    if not esImagen(rutaImagen):
        messagebox.showerror(           
            "Error",                    
            "El tipo de archivo no es una imagen"
        )
    else:
        predicirPez(rutaImagen)

    
    


def esImagen(rutaImagen):
    contiene = re.search(".png$|.jpg$|.jpeg$|.webp$", rutaImagen)

    if contiene:
        return True
    return False


def crearInterfaz():
    ventana = Tk()


 

    ventana.title("Adivinar Especie de Pez")
  
    ventana.geometry("800x800")

    labelBuscarArchivo = Label(ventana, 
                            text = "Seleccione imagen de pez",
                            width = 100, height = 4, 
                            fg = "black",
                            font = ("Arial", 25))
  
      
    botonBuscarArchivo = Button(ventana, 
                        width = 30, height = 4, 
                        text = "Buscar Imagenes",
                        font = ("Arial", 18),
                        fg = "white",
                        bg = "black",
                        command = buscarArchivos) 

    labelBuscarArchivo.pack()

    botonBuscarArchivo.pack()
    
    ventana.mainloop()



main()


