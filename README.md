# punctuation_restoring
Puntuación automática de textos en español 
- Los scripts de entrenamiento son bidir_atten_embb2.py y bidir_atten_embb3.py
- En la carpeta Texto se encuentran algunos de los documentos utilizados para el entrenamiento. 
- Para testear los algoritmos se utilizan los scripts bidir_att_embb_2_restore.py y bidir_att_embb_3_restore.py
- La función inputs_test carga los textos que se quieren utilizar como entrada. Estos se pueden encontrar en la carpeta Texto al igual que los de entrenamiento. El path al archivo se establece en texto_procesado = nlp(leer_texto(path)).
NOTA: Puede que algunos de estos textos contengan caracteres especiales no recogidos a la hora de entrenar los modelos.
- En frases_total2.csv.zip se encuentran los csv utilizados para el entrenamiento. Se debe descomprimir la carpeta para utilizar el archivo en la función inputs de los scripts de testeo. 
- Los modelos finales entrenados son model_bidirectional_attention3.h5 y model_bidirectional_attention2.h5
