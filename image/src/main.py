import numpy as np
import spacy 
import spacy
#from flask import Flask, request, jsonify, abort
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
from hdbcli import dbapi
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
import os
from datetime import datetime
import os
import json
import boto3

region = 'eu-central-1'

# load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
sm_client = boto3.client('secretsmanager',region )

#Getting the spaCy model
nlp = spacy.load('en_core_web_md') 
#app = Flask(__name__)
print("STARTING111")
def exception_handler(e):
    status_code = 404

    return {
        'statusCode': status_code,
        'body': json.dumps(str(e))
    }
def best_match_word_per_chunk(chunk, word_list, fuzzy_threshold):

    # Extract best matches for each word from the paragraph based on pre specified threshold
    filtered_candidates = process.extractBests(chunk, word_list, scorer = fuzz.token_set_ratio, score_cutoff = fuzzy_threshold, limit = len(word_list))   

    #updating wordlist to have filtered words and if not meeting threshold, aborts with code 404 along a suitable message
    if filtered_candidates != []:
        word_list = filtered_candidates
    else:
        raise Exception("No suitable candidate found")

    #preparing for spacy processing
    chunk = preprocessing(chunk)    
    match_score_list = []

    # Iterating over each of the filtered out words and getting a combined similarity score obtained for each word which consists of score generated from fuzzywuzzy and sentence transformer based cosine similarity
    for word, fuzzy_score in word_list:
        word= preprocessing(word)
        spacy_score = similarity(chunk, word)
        combined_score = ((fuzzy_score/100) * 0.55) + (spacy_score * 0.45)
        #print(f'''Fuzzy Score for {word} is: {round(fuzzy_score, 4)}\nSpacy Score for {word} is: {round(spacy_score, 4)}\nCombined Score for {word} is: {round(combined_score, 4)}\n''')      
        match_score_list.append(round(combined_score, 4))

    # The list of scores are arranged in the descending order and the top three scores are chosen
    temp_score_list = sorted(match_score_list, reverse = True)
    temp_score_list = temp_score_list[:3]

    # Preparing a list of tuples of the top three (word, score) pairs
    result_list = []
    for i in range(len(temp_score_list)):
        result_list.append((word_list[match_score_list.index(temp_score_list[i])][0], temp_score_list[i]))

    return(result_list)
def preprocessing (text):
    doc = nlp(text)

    # Clean and process text
    cleaned_text = ' '.join([element.lemma_.lower() for element in doc if not element.is_stop and not element.is_punct])
    return cleaned_text

def similarity(text1, text2):
    embedding1 = model.encode(text1)
    embedding2 = model.encode(text2)

    return (util.cos_sim(embedding1, embedding2)).item()
def handler(event, context):

    print("Printing Event:\n", event)
    paragraph = event["queryStringParameters"]["paragraph"]
    print("Printing paragraph:\n", paragraph)
    arr = np.random.randint(0,10,(3,3))

    get_secret_value_response = sm_client.get_secret_value(SecretId='Hana-cloud-dev-secrets')

    secret = get_secret_value_response['SecretString']
    db_secrets = json.loads(secret)
    dbusername = db_secrets["dbusername"]
    dbpassword = db_secrets["dbpassword"]
    dbip = db_secrets["dbip"]
    dbport = db_secrets["dbport"]
    
    #Making the database connection
    conn = dbapi.connect(
            address=dbip, 
            port=dbport, 
            user=dbusername, 
            password=dbpassword
        )

    # Fetching Partner IDs and Partner Names from the specified table
    print(f'''Selection query start time: {datetime.now}''' )
    query = '''SELECT ID, FULL_NAME FROM "696480533D0A45F5895621B1DDA6A3D2".ATOM_DB_MDM_CRM_INSUREDS'''
    cursor = conn.cursor()
    cursor.execute(query)
    print(f'''Selection query end time: {datetime.now}''')
    data = cursor.fetchall()

    df = pd.DataFrame(data, columns = ['ID', 'FULL_NAME'])

    #Creating a dictionary for index identification
    partner_dict = {}
    for id, name in data:
        partner_dict[name] = id

    #Adding all the the names into a list for search
    name_list = []
    for nam in partner_dict:
        name_list.append(nam)

    fuzzy_threshold = 70

    chunks = re.split(r'\s+and/or\s+', paragraph)
    # If many chunks have been obtianed, only the first one is chosen to perform the matching
    if len(chunks) != 1:
        try:
            final_list = best_match_word_per_chunk(chunks[0], name_list, fuzzy_threshold)
        except Exception as e:
            return exception_handler(e)
    # If there is no 'and/or' in the supplied input, the search is done on the first 100 characteres of the input
    else:
        try:
            final_list = best_match_word_per_chunk(chunks[0][:100], name_list, fuzzy_threshold)
        except Exception as e:
            return exception_handler(e)

    result = {'Best matching partners': [], 'Best Scores': [], 'Best Partner IDS': []}

    no_of_additional_mathches = -1
    for i in range(min(len(final_list), 3)):
        result['Best matching partners'].append(final_list[i][0])
        result['Best Scores'].append(final_list[i][1])
        result['Best Partner IDS'].append(partner_dict[final_list[i][0]])
        no_of_additional_mathches += 1
    
    other_matches_list = []
    for j in range(1, (no_of_additional_mathches + 1)):
        other_matches_list.append({'BP_ID': result['Best Partner IDS'][j], 'BP_NAME': result['Best matching partners'][j], 'SCORE': result['Best Scores'][j]})
    
    output = {'BP_ID': result['Best Partner IDS'][0], 'BP_NAME': result['Best matching partners'][0], 'SCORE': result['Best Scores'][0], 'other_matches' :other_matches_list}
    
    return {
        "statusCode":200, "body": json.dumps(output)
    }
    