import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder
import numpy as np
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

spark=SparkSession.builder.appName('wine_quality').getOrCreate()
df=spark.read.option("delimiter", ";").csv('winequality-white.csv', inferSchema=False, header=True)
df.show(10)
df.printSchema()
print df.columns

df2 = df.withColumn('fixed acidity', df['fixed acidity'].cast('double')) \
.withColumn('volatile acidity', df['volatile acidity'].cast('double')) \
.withColumn('citric acid', df['citric acid'].cast('double')) \
.withColumn('residual sugar', df['residual sugar'].cast('double')) \
.withColumn('chlorides', df['chlorides'].cast('double')) \
.withColumn('free sulfur dioxide', df['free sulfur dioxide'].cast('int')) \
.withColumn('total sulfur dioxide', df['total sulfur dioxide'].cast('int')) \
.withColumn('density', df['density'].cast('double')) \
.withColumn('pH', df['pH'].cast('double')) \
.withColumn('sulphates', df['sulphates'].cast('double')) \
.withColumn('alcohol', df['alcohol'].cast('double')) \
.withColumn('quality', df['quality'].cast('int'))

df2.printSchema()

for item in df2.head(5):
    print(item)
    print('\n')

assembler = VectorAssembler(inputCols=['fixed acidity', \
'volatile acidity', \
'citric acid', \
'residual sugar', \
'chlorides', \
'free sulfur dioxide', \
'total sulfur dioxide', \
'density', \
'pH', \
'sulphates', \
'alcohol'], outputCol='features')

output=assembler.transform(df2)
output.select('features', 'quality').show(5)

final_data = output.select('features', 'quality')
training_data,test_data=final_data.randomSplit([0.8, 0.2])
training_data.describe().show()
test_data.describe().show()

wine_lr=LinearRegression(featuresCol='features', labelCol='quality', regParam=0.0, maxIter=300, elasticNetParam=0.0)
trained_wine_model_lr=wine_lr.fit(training_data)
wine_results_lr=trained_wine_model_lr.evaluate(training_data)

print('Rsquared error: ', wine_results_lr.r2)

unlabeled_data=test_data.select('features')
unlabeled_data.show(5)

predictions_lr=trained_wine_model_lr.transform(unlabeled_data)
predictions_lr.show()

rf = RandomForestRegressor(featuresCol='features', labelCol='quality')
trained_rf = rf.fit(training_data)
rf_results = trained_rf.transform(training_data)
evaluator = RegressionEvaluator(labelCol='quality')
print evaluator.evaluate(rf_results, {evaluator.metricName: "r2"})
