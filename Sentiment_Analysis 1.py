!pip install schedule
!pip install azure-eventhub==5.11.5

import numpy as np
import pandas as pd
import requests
import re
import torch
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaModel, RobertaTokenizer
import openai
import schedule
import time
from azure.eventhub import EventHubProducerClient, EventData
import json
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql import functions as F
from openai.error import RateLimitError
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from datetime import datetime
from dateutil.parser import parse, isoparse

spark = SparkSession.builder \
    .appName('App') \
    .getOrCreate()

# Set up Azure OpenAI API credentials
openai.api_version = "**"
openai.api_base = "**"
openai.api_type = "**"
openai.api_key = "**"
deployment_id = "**"

# Set up FinBERT
tokenizer_finbert = AutoTokenizer.from_pretrained('ProsusAI/finbert')
finbert_model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')

# Set up RoBERTa
tokenizer_roberta = AutoTokenizer.from_pretrained('roberta-base')
roberta_model = AutoModelForSequenceClassification.from_pretrained('roberta-base')

def send_to_eventhub(csv_data):
    # Your Azure Event Hub connection string and event hub name
    connection_str = "**"
    eventhub_name = "**"

    producer = EventHubProducerClient.from_connection_string(connection_str, eventhub_name=eventhub_name)

    with producer:
        event_data_batch = producer.create_batch()
        event_data_batch.add(EventData(csv_data))
        producer.send_batch(event_data_batch)
    print("Data sent to Azure Event Hub successfully.")

def finbert_model_scores(article_text):
    inputs = tokenizer_finbert(article_text, padding=True, truncation=True, return_tensors='pt')
    outputs = finbert_model(**inputs)

    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive = predictions[:, 0].item()
    negative = predictions[:, 1].item()
    # Calculate the sentiment score
    sentiment_score = positive - negative

    return sentiment_score

def roberta_model_scores(article_text):
    # Tokenize the input text
    inputs = tokenizer_roberta(article_text, return_tensors="pt", max_length=512, truncation=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Perform sentiment analysis
    inputs = {key: inputs[key].to(device) for key in inputs}
    outputs = roberta_model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    positive_prob = probabilities[0, 1].item()  # Access the probability for positive sentiment
    negative_prob = probabilities[0, 0].item()  # Access the probability for negative sentiment
    sentiment_score = positive_prob - negative_prob

    return sentiment_score

def get_openai_response(article_text, stock_symbol):
    chatgpt_input = f"Please provide a sentiment score for the following news article related to stock {stock_symbol}: /n/n {article_text}/n/n Provide a single float value as the sentiment score based on the following guideline: If there is no discernible effect of the news on the stock, return 0. If the news has a negative effect on the stock, return a float value between -1 and 0. If the news has a positive effect on the stock, return a float value between 0 and 1."
    try:
        response = openai.ChatCompletion.create(
            deployment_id=deployment_id,
            messages=[
                {"role": "user", "content": chatgpt_input}
            ]
        )
        assistant_response = response['choices'][0]['message']['content']
        try:
            # Try to convert the response to a float
            sentiment_score = float(assistant_response)
        except ValueError:
            # If conversion fails (e.g., response is not a valid number), set sentiment_score to 0
            sentiment_score = 0
    except RateLimitError as e:
        print("Rate limit exceeded. Waiting for 6 seconds before retrying.")
        time.sleep(6)
        sentiment_score = get_openai_response(article_text, stock_symbol)
    return sentiment_score

# Function to fetch news articles and perform sentiment analysis
def fetch_and_print_sentiment_scores():
    kustoQuery = "['NewsSentimentScoreIngest'] | take 5000"
    kustoUri = "**"
    database = "StockDatabase"
    accessToken = mssparkutils.credentials.getToken(kustoUri)
    kustoDf  = spark.read\
        .format("com.microsoft.kusto.spark.synapse.datasource")\
        .option("accessToken", accessToken)\
        .option("kustoCluster", kustoUri)\
        .option("kustoDatabase", database)\
        .option("kustoQuery", kustoQuery).load()

    if not isinstance(kustoDf, pyspark.sql.dataframe.DataFrame):
        print("Error: kustoDf is not a DataFrame.")
        return

    if 'URL' not in kustoDf.columns:
        print("Error: 'URL' column not found in kustoDf.")
        return
    # List of stock symbols
    stock_symbols = ['CSCO', 'ADSK', 'INTC', 'ADBE', 'WMT', 'NFLX', 'DIS' , 'KO', 'AMZN', 'META']
    # News API URL and API token
    api_token = "**"
    url = f"https://api.marketaux.com/v1/news/all?filter_entities=true&language=en&api_token={api_token}&country=us"

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error in response from news API: Status Code {response.status_code}")
        return
    data = response.json()
    articles = data.get('data', [])
    if not articles:
        print("No articles found in the API response.")
        return

    for article in articles:
        article_url = article.get('url', '')
        published_at = article.get('published_at', '')

        if published_at.endswith('Z'):
            published_at = published_at[:-1]
        try:

            published_datetime = isoparse(published_at)
        except ValueError as e:
            print(f"Error parsing published datetime for article: {e}")
            continue

        formatted_published_at = published_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Check if the URL has already been processed
        if article_url in kustoDf.select('URL').rdd.flatMap(lambda x: x).collect():
            print('Article already processed:', article_url)
            continue  # Skip processing this article and move to the next one
         # Process the new URL
        response_article = requests.get(article_url)
        soup_article = BeautifulSoup(response_article.content, 'html.parser')
        article_text = ' '.join([p.get_text() for p in soup_article.find_all('p')])
        data_list = []  # List to store data for each article
        # Perform sentiment analysis
        for stock_symbol in stock_symbols:
            finbert_score = finbert_model_scores(article_text)
            roberta_score = roberta_model_scores(article_text)
            chatgpt_score = get_openai_response(article_text, stock_symbol)

            data_list.append({
                'Headline': article['title'],
                'Description': article['description'],
                'Source': article['source'],
                'Published At': published_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                'URL': article_url,
                'Stock Symbol': stock_symbol,
                'FinBERT Score': finbert_score,
                'RoBERTa Score': roberta_score,
                'ChatGPT Score': float(chatgpt_score),  # Explicitly cast to DoubleType
                'Combined Score': 0.60 * float(chatgpt_score) + 0.20 * float(finbert_score) + 0.20 * float(roberta_score)
            })

        # Create DataFrame
        schema = StructType([
            StructField('Headline', StringType(), True),
            StructField('Description', StringType(), True),
            StructField('Source', StringType(), True),
            StructField('Published At', StringType(), True),
            StructField('URL', StringType(), True),
            StructField('Stock Symbol', StringType(), True),
            StructField('FinBERT Score', DoubleType(), True),
            StructField('RoBERTa Score', DoubleType(), True),
            StructField('ChatGPT Score', DoubleType(), True),
            StructField('Combined Score', DoubleType(), True)
        ])
        df = spark.createDataFrame(data_list, schema=schema)

        # Convert DataFrame to JSON
        df_pandas = df.toPandas()

        # Convert Pandas DataFrame to CSV format
        csv_data = df_pandas.to_csv(index=False)

        # Send CSV data to Azure Event Hub
        send_to_eventhub(csv_data)

        # Create DataFrame with the same schema as kustoDf
        schema = kustoDf.schema
        new_urls_df = spark.createDataFrame([], schema=schema)

        # Add processed URL to new_urls_df
        for _ in range(len(stock_symbols)):
            new_row = (None,) * len(schema.fields)  # Create a row with None values for each column
            new_urls_df = new_urls_df.union(spark.createDataFrame([new_row], schema=schema))

        # Union operation
        kustoDf = kustoDf.union(new_urls_df)

    # Display the DataFrame
    display(kustoDf)
    return df

fetch_and_print_sentiment_scores()