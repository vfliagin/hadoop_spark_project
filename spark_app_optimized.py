import findspark
findspark.init()

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

optimize = True

sc = SparkContext.getOrCreate(SparkConf().setMaster('spark://spark-master:7077'))
sc.setLogLevel("INFO")

spark = SparkSession.builder.getOrCreate()

df = spark.read.csv("hdfs://namenode:9001/data/hotel_data/hotel_bookings.csv", header=True, inferSchema=True)

from pyspark.ml.feature import StringIndexer

train, test = df.randomSplit([0.7, 0.3])

if optimize:
	train = train.repartition(8)
	test = test.repartition(8)

inputs = [col[0] for col in df.dtypes if col[1] == 'string']
outputs = [col + '_index' for col in inputs]
indexer = StringIndexer(inputCols=inputs, outputCols=outputs)
model = indexer.fit(train)
train = model.transform(train)
test = model.transform(test)

from pyspark.ml.classification import GBTClassifier, DecisionTreeClassifier, NaiveBayes
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

ignore = ['reservation_status_date_index', 'is_canceled', 'reservation_status_date', 'agent_index', 'company_index', 'country_index', 'reservation_status_index'] + inputs

assembler = VectorAssembler(inputCols=[x for x in train.columns if x not in ignore], outputCol='features')
        
train_data = (assembler.transform(train).select("is_canceled", "features"))
test_data = (assembler.transform(train).select("is_canceled", "features"))

if optimize:
	train_data.cache()

gbt = GBTClassifier(labelCol="is_canceled", featuresCol="features")

evaluator = BinaryClassificationEvaluator(labelCol="is_canceled")
paramGrid = ParamGridBuilder().addGrid(gbt.maxIter, [10, 20, 30]).addGrid(gbt.maxDepth, [3, 4, 5]).build()
crossval = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=6)
model = crossval.fit(train_data)

auc_roc = model.avgMetrics[0]
print("AUC ROC = ", auc_roc)

predictions = model.transform(test_data)

evaluator = MulticlassClassificationEvaluator(labelCol="is_canceled", metricName="accuracy")

accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = {:.2f}".format(accuracy))

