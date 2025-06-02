from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .appName("NASA Web Logs Analysis") \
    .getOrCreate()

# Print Spark version
print(f"Spark Version: {spark.version}")

log_file_path = "access_log_Aug95"
raw_data = spark.read.text(log_file_path)
raw_data.show(5, truncate=False)

#----------------------------------------------------------------------------------------------------------------

from pyspark.sql.functions import regexp_extract

# Regular expressions for parsing
host_pattern = r'^(\S+)'  # Extracts host
timestamp_pattern = r'\[(.*?)\]'  # Extracts timestamp
method_endpoint_pattern = r'\"(GET|POST|HEAD|PUT|DELETE)\s(\S+)\s'  # Extracts HTTP method and endpoint
status_pattern = r'(\d{3})'  # Extracts status code
content_size_pattern = r'(\d+)$'  # Extracts content size

# Parse data into structured columns
logs_df = raw_data.withColumn("host", regexp_extract("value", host_pattern, 1)) \
                  .withColumn("timestamp", regexp_extract("value", timestamp_pattern, 1)) \
                  .withColumn("endpoint", regexp_extract("value", method_endpoint_pattern, 2)) \
                  .withColumn("status", regexp_extract("value", status_pattern, 1).cast("int")) \
                  .withColumn("content_size", regexp_extract("value", content_size_pattern, 1).cast("int"))
logs_df.show(5)

#----------------------------------------------------------------------------------------------------------------

#a
logs_df.selectExpr(
    "AVG(content_size) as avg_size",
    "MIN(content_size) as min_size",
    "MAX(content_size) as max_size"
).show()

#----------------------------------------------------------------------------------------------------------------

#b
logs_df.groupBy("status").count().orderBy("status").show()

#----------------------------------------------------------------------------------------------------------------

# c
from pyspark.sql.functions import desc

logs_df.groupBy("host").count().orderBy(desc("count")).limit(10).show()

#----------------------------------------------------------------------------------------------------------------

#d
logs_df.groupBy("endpoint").count().orderBy(desc("count")).limit(20).show()

#----------------------------------------------------------------------------------------------------------------

# e
logs_df.filter("status >= 400").groupBy("endpoint").count().orderBy(desc("count")).limit(10).show()

#----------------------------------------------------------------------------------------------------------------

# f
unique_hosts = logs_df.select("host").distinct().count()
print(f"Unique hosts: {unique_hosts}")

#----------------------------------------------------------------------------------------------------------------

# g
total_requests = logs_df.count()
avg_requests_per_host = total_requests / unique_hosts
print(f"Average requests per host: {avg_requests_per_host}")

#----------------------------------------------------------------------------------------------------------------

# h
logs_df.filter("status == 404").groupBy("endpoint").count().orderBy(desc("count")).limit(20).show()

#----------------------------------------------------------------------------------------------------------------

# i
logs_df.filter("status == 404").groupBy("host").count().orderBy(desc("count")).limit(20).show()

#----------------------------------------------------------------------------------------------------------------

# j
'''from pyspark.sql.functions import to_date

daily_404 = logs_df.filter("status == 404").withColumn("date", to_date("timestamp", "dd/MMM/yyyy")) \
                   .groupBy("date").count().toPandas()

import matplotlib.pyplot as plt

daily_404.plot(x="date", y="count", kind="line", title="404 Errors Per Day")
plt.show()
'''
from pyspark.sql.functions import to_timestamp

# Set legacy parser policy if needed
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

# Correct timestamp parsing
correct_format = "dd/MMM/yyyy:HH:mm:ss Z"
daily_404 = logs_df.filter("status == 404") \
    .withColumn("date", to_timestamp("timestamp", correct_format)) \
    .groupBy("date").count().toPandas()

import matplotlib.pyplot as plt

# Plot the result
daily_404.plot(x="date", y="count", kind="line", title="404 Errors Per Day")
plt.show()

#----------------------------------------------------------------------------------------------------------------

# k
from pyspark.sql.functions import to_date, desc # Import to_date and desc here as desc is also used

logs_df.filter("status == 404").withColumn("date", to_date("timestamp", "dd/MMM/yyyy")) \
    .groupBy("date").count().orderBy(desc("count")).limit(4).show()

#----------------------------------------------------------------------------------------------------------------

# l
from pyspark.sql.functions import hour

hourly_404 = logs_df.filter("status == 404").withColumn("hour", hour("timestamp")) \
                    .groupBy("hour").count().toPandas()

hourly_404.plot(x="hour", y="count", kind="bar", title="Hourly 404 Errors")
plt.show()

#----------------------------------------------------------------------------------------------------------------
