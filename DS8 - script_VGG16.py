# Script de preprocessing sur Spark (features VGG16)

# Importation des librairies :
from typing import Iterator
import numpy as np
import pandas as pd
from PIL import Image
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col, split
from pyspark.sql.types import StructType, StructField, ArrayType, IntegerType, FloatType
from pyspark.ml.image import ImageSchema
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.ml.feature import PCA
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

# Adresse du compartiment S3 :
path = "s3a://vivianorsprojet8/dataset/"

# Instanciation de Spark :
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()

# Récupération des images :
df = spark.read.format("image").load(path + "*")

# Traitement et redimensionnement des images (à cause d'erreurs mémoire)
schema = StructType(df.select("image.*").schema.fields + [
    StructField("data_array", ArrayType(IntegerType()), True)
])

def preprocess_array_image(data):
    mode = 'RGB' 
    img = Image.frombytes(mode=mode, data=data.data, size=[100, 100])
    img = img.resize([224, 224])
    arr = np.asarray(img)
    arr = arr.reshape([224*224*3])
    return arr

def array_image(dataframe_batch_iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    for dataframe_batch in dataframe_batch_iterator:
        dataframe_batch["data_array"] = dataframe_batch.apply(preprocess_array_image, axis=1)
        yield dataframe_batch

df = df.select("image.*").mapInPandas(array_image, schema)

# Préparation des images et récupération des features :
def normalize_array(arr):
    return preprocess_input(arr.reshape([224,224,3]))

@pandas_udf(ArrayType(FloatType()))
def featurize(iterator: Iterator[pd.Series]) -> Iterator[pd.Series] :
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    for input_array in iterator :
        normalized_input = np.stack(input_array.map(normalize_array))
        preds = model.predict(normalized_input)
        yield pd.Series(list(preds))
df = df.withColumn("cnn_features", featurize("data_array"))

# Vectorisation :
ImageSchema.imageFields
img2vec = udf(lambda l: Vectors.dense(l), VectorUDT())
df = df.withColumn("cnn_vectors", img2vec("cnn_features"))
df.show()

# Réduction de dimension :
pca = PCA(k=10, inputCol="cnn_vectors", outputCol="cnn_vectors_redux")
redux = pca.fit(df)
df = redux.transform(df)

# Réorganisation du Dataframe et ajout de la target :
df = df.select("origin", "cnn_vectors", "cnn_vectors_redux")
df = df.withColumn("target", split(col("origin"), "dataset/").getItem(1))
df = df.withColumn("target", split(col("target"), "/").getItem(0))

# Export :
resultats = "s3://vivianorsprojet8/résultats/"
df.write.parquet(resultats)