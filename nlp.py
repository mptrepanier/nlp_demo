from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover, CountVectorizer, RegexTokenizer
from pyspark.ml.clustering import LDA
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark import SparkFiles
import nltk
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import string
import re

# gcloud dataproc jobs submit pyspark --cluster cow nlp.py

def word_filter(word):
    """
    Boolean check leveraged by remove_features(). Checks if a word is alphanumeric and fits
    the length requirements.
    Input: String
    Output: Boolean
    """

    if (re.match('^[a-zA-z]*$', word) and len(word) > 2 and len(word) < 25):
        return True
    return False

def remove_features(text):
        """
        Filters out any words that do not meet word_filters() requirements.
        Input: String
        Output: String
        """
        return ' '.join([word for word in re.split('\\s+', text) if word_filter(word)])

def lemmatize(text):
    # Courtesy: https://mingchen0919.github.io/learning-apache-spark/nlpC.html
    """
    Runs NLTK's lemmatizer on the input text.
    Input: String
    Output: String
    """
    list_pos = 0
    cleaned_str = ''
    lemmatizer = WordNetLemmatizer()
    split_text = text.split()
    tagged_words = pos_tag(split_text)

    for word in tagged_words:
        if 'v' in word[1].lower():
            lemma = lemmatizer.lemmatize(word[0], pos='v')
        else:
            lemma = lemmatizer.lemmatize(word[0], pos='n')
        if list_pos == 0:
            cleaned_str = lemma
        else:
            cleaned_str = cleaned_str + ' ' + lemma
        list_pos += 1
    return cleaned_str

def indices_mapping(indices, vocab):
    return [vocab[i] for i in indices]

def make_indices_mapping(vocab):
    schema = ArrayType(StringType())
    return udf(lambda indices: indices_mapping(indices, vocab), schema)


def length_filter(tokens):
    """
    Returns a list of all tokens with lengths between 7 and 25, exclusive.
    Note that this is applied prior to lemmatization, so lemmas not matching this
    filter may be included in the final result.

    Input: List[String]
    Output: List[String]
    """
    return [token for token in tokens if (len(token) > 7 and len(token) < 25)]

def plot_topic_descriptions(topics_with_words):
    """
    Plots the word weights of the first topic.
    """
    topics_with_words_pdf = topics_with_words.toPandas()
    first_topic = topics_with_words_pdf.iloc[0]
    min_first_topic = min(first_topic.termWeights)
    max_first_topic = max(first_topic.termWeights)

    for i in range(len(first_topic.terms)):
        plt.bar(0, (first_topic.termWeights[i] - min_first_topic) / (max_first_topic - min_first_topic) * 100,
                label=first_topic.terms[i])

    ax = plt.axes()
    rects = ax.patches
    rectangle_tops = []
    for rect in rects:
        rectangle_tops.append(rect.get_height())

    rectangle_tops.append(0.0)
    i = 1
    for (rect, label) in zip(rects, first_topic.terms):
        height = rect.get_height()
        width = rect.get_width()
        plt.text(rect.get_x() + width / 2, (height + rectangle_tops[i]) / 2, label, ha='center', va='center', size=14)
        i = i + 1

    # plt.legend(bbox_to_anchor=(1.05, 1), loc="best", ncol=1)
    plt.xlabel("Topic 0", size=14)
    plt.ylabel("Normalized Weight per Word", size=14)
    plt.tick_params(
        axis='x',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off'
    )

    plt.figure(figsize=(100, 100), dpi=100)
    plt.show()
    plt.close()

def plot_document_descriptions(topic_distibutions):
    topic_distribution_pds = model.transform(feature_vectors).select("topicDistribution").toPandas()
    first_document = topic_distribution_pds.iloc[0]
    second_document = topic_distribution_pds.iloc[5421]

    labels = ['Topic 0', 'Topic 1', 'Topic2']
    plt.pie(first_document.topicDistribution, labels=labels, autopct='%1.1f%%')
    plt.title("Topic Distribution for the First Document", size=14)
    plt.show()
    plt.close()


if __name__ == "__main__":

    # Initializing the SparkContext and importing the data.
    spark = SparkSession.builder.appName("LDA").getOrCreate()
    bucket_location = "gs://dataproc-04d7eda2-db56-484f-aba4-5db51f8b3d84-us/"
    file_location = "mimic/NOTEEVENTS_DEMO_PROCESSED.csv"
    # Raw data is of form |rowid|subjectid|hadmid|chartid|charttime|storetime|category|description|cgid|iserror|text|
    # Please see the NOTEEVENTS table at https://mimic.physionet.org/about/mimic/ for more info.
    notes_df = (spark.read.format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .option("delimiter", ",")
                .load(bucket_location + file_location).na.fill(-1))


    # Creating the UDFs.
    remove_features_udf = udf(remove_features, StringType())
    lemmatize_udf = udf(lemmatize, StringType())

    # Applying the UDFs to generate a DataFrame of the format |rowid|lemmatized|.
    # Note that the feature to easily convert UDFs into ML Pipeline transformations is not yet in Spark.
    filtered_notes_df = (notes_df.filter(notes_df['iserror'] != 1)
                           .select(notes_df['rowid'], remove_features_udf(notes_df['text'].cast("string")).alias('text')))

    lemmatized_notes_df = filtered_notes_df.withColumn("lemmatized", lemmatize_udf(filtered_notes_df['text']))

    # Defining our ML pipeline.
    tokenizer = RegexTokenizer(inputCol='lemmatized', outputCol='tokens', pattern='\\W')
    common_words = ['admission', 'discharge'] + StopWordsRemover().getStopWords()
    remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), stopWords=common_words,outputCol='tokens_filtered')
    cv = CountVectorizer(inputCol=remover.getOutputCol(),outputCol='features')
    lda = LDA().setK(3)
    pipeline = Pipeline(stages=[tokenizer, remover, cv, lda])

    # Fitting our pipeline.
    model = pipeline.fit(lemmatized_notes_df)

    # Here we access the last stage of the model, as this is where we applied the LDA.
    lda_model = model.stages[-1]
    cv_model = model.stages[2]

    topics = lda_model.describeTopics()
    topics_with_words = (topics.select(topics["topic"],
                                       make_indices_mapping(cv_model.vocabulary)(topics["termIndices"])
                                       .alias("terms"), topics["termWeights"]))
    topics_with_words.show(truncate=False)
    # Produces a DataFrame of schema |rowid|text|lemmatized|tokens|tokens_filtered|features|topicDistribution|
    # Note that the fitted model is being called on the input. (Albeit this is a clustering algorithm,
    # so cross-validation need not apply.)
    model.transform(lemmatized_notes_df).show()

    # plot_topic_descriptions(topics_with_words)
    # plot_topic_descriptions(model.transform(lemmatized_notes_df).select("topicDistribution").toPandas()


    spark.stop()



















