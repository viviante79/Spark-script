# Importation des librairies :
from typing import Iterator
import numpy as np
import pandas as pd
from PIL import Image
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, udf, col
from pyspark.sql.types import StructType, StructField, ArrayType, IntegerType
from pyspark.ml.image import ImageSchema
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.ml.feature import PCA

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
    img = img.resize([10, 10])
    arr = np.asarray(img)
    arr = arr.reshape([10*10*3])
    return arr

def array_image(dataframe_batch_iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    for dataframe_batch in dataframe_batch_iterator:
        dataframe_batch["data_array"] = dataframe_batch.apply(preprocess_array_image, axis=1)
        yield dataframe_batch

df = df.select("image.*").mapInPandas(array_image, schema)

# Vectorisation :
ImageSchema.imageFields
img2vec = udf(lambda l: Vectors.dense(l), VectorUDT())
df = df.withColumn("vectors", img2vec("data_array"))

# Standardisation :
scaler = StandardScaler(inputCol="vectors", outputCol="scaled_vectors", withMean=True, withStd=True)
model_std = scaler.fit(df)
df = model_std.transform(df)

# Réduction de dimension :
pca = PCA(k=10, inputCol="scaled_vectors", outputCol="vectors_redux")
redux = pca.fit(df)
df = redux.transform(df)

# Réorganisation du Dataframe et ajout de la target :
df = df.select("origin","vectors_redux")
df = df.withColumn("target", split(col("origin"), "dataset/").getItem(1))
df = df.withColumn("target", split(col("target"), "/").getItem(0))

# Export :
resultats = "s3://vivianorsprojet8/résultats/"
df.write.parquet(resultats)