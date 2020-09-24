# sia_tp3
Ejecución:
Colocarse en el directorio donde se encuentre el archivo requirements.txt dentro de una terminal e ingresar el comando 'pip install -r requirement.txt'
para instalar las dependencias necesarias.
Luego para ejecutar un script ingresar el comando 'py archivo.py', siendo "archivo" el nombre del script de Python
Siendo nuestros archivos ejecutables: LogicAnd.py, LogicXor.py, DigitsTest.py, XorMIpTest, SimpleLinearDataTest.py y NonLinearDataTest.py

ej1:
learn_factor: el factor de aprendizaje del perceptrón,

ej2:
total_epochs: Cantidad total de épocas sobre las cuales entrenar al perceptrón
epoch_step: Paso en la cantidad de épocas sobre las cuales se entrena al perceptrón
learn_factor: el factor de aprendizaje del perceptrón
k: Cantidad de subdivisiones sobre el conjunto de entrenamiento. Las subdivisiones serán conjuntos de testeo disjuntos, siendo el conjunto de entrenamiento el complemento de cada uno
cross_validation: activar o desactivar la validación cruzada
beta: Valor de la variable Beta en la fórmulas sigmodeas.


Ejemplo de configuración de ejecución:
{
  "ej1": [{
      "limit": "500",
      "learn_factor": "0.1"

  }],

  "ej2": [{
      "total_epochs": "300",
      "epoch_step": "50",
      "learn_factor": "0.01",
      "k": "5",
      "cross_validation": "true",
      "beta": "5"
  }]
}
