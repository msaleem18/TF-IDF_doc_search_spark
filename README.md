# TF-IDF_doc_search_spark
Using Spark to implement a TF-IDF document search

#### The task is to create a document search algorithim using TF-IDF

Following code is for pyspark
```python
#import relevant items
import re
from pyspark.sql.functions import *


#function to make all the words lowercase, remove punctuations
def removePunc (words):
	words = words.lower().strip()
	words = re.sub("[^0-9a-zA-Z ]", "", words)
	return words


#read the files - i'm using the files provided in the athletics folder
read_all_files = sc.wholeTextFiles("/user/root/final/athletics/").map(lambda x: (x[0],removePunc(x[1]))).map(lambda x: (x[0],x[1].split()))

#create dataframe and use explode to create row wise items, the new column is name words_explode
df= read_all_files.toDF(['doc_id','words'])
df_explode = df.withColumn("words_explode",explode('words'))

#create Term Frequency dataframe
term_frequency_df = df_explode.groupBy('doc_id','words_explode').agg(count('doc_id').alias('tf'))

doc_frequency_df = df_explode.groupBy('words_explode').agg(countDistinct('doc_id').alias('df'))
total_doc_count = df_explode.agg(countDistinct('doc_id')).collect()[0][0]

idf_df = doc_frequency_df.withColumn('idf',log((total_doc_count + 1) / (doc_frequency_df.df + 1)))

tf_idf_join_df = term_frequency_df.join(idf_df,'words_explode')
tf_idf_join_df = tf_idf_join_df.withColumn('tf_idf', tf_idf_join_df.tf * tf_idf_join_df.idf)

#function to search words and return top N docs
def search_word(word_from_user,top_n_docs):
	#remove puncutations, make lower case and tokenize the words
	tokenized = removePunc(word_from_user).split()
	N = len(tokenized)

	#search the words in the dataframe and aggregate by combining the TFIDF scode and multiplying it with the weight of the words founds
	results = tf_idf_join_df.filter(tf_idf_join_df.words_explode.isin(tokenized)).groupBy('words_explode','doc_id').agg(sum('tf_idf').alias('sum_tfidf'),count('words_explode').alias('count_tfidf'))
	results = results.withColumn('score',results.sum_tfidf * (results.count_tfidf/N))

	#sort descending and limit to top N
	results = results.sort(desc('score')).limit(top_n_docs)

	return results.select('doc_id','score')
```
