import pandas as pd
from pymongo import MongoClient
#import dnstwist
import re
import math
from collections import Counter
import sklearn.feature_extraction
import numpy as np
from feature_extractor import *
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
import os



client = MongoClient('####', ssl=True)
dbname = client['phishing']
dataset = dbname["phishing"]

datasetpd = pd.DataFrame(list(dataset.find()))
#print(datasetpd["html_content"].head())

def domain(url):
  a = str(re.findall(r'^(?:http:\/\/|www\.|https:\/\/)([^\/]+)', url))[2:-2]
  if a == '':
    a = url
  return a


datasetpd["domain"] = datasetpd["url"].apply(domain)
datasetpd = datasetpd.rename(columns={"html_content": "content"})


##### URL FEATURES #####

urldf = pd.DataFrame(columns= ['domain','hyphen', 'dot', 'at','digit','equal','longer than 7', 'mimics','u_susp words','u_length','extension', 'entropy',
                                   'alexa_ngram_count','dict_ngram_count','squatting','phishing'])



for index, row in datasetpd.iterrows():
    if row["label"] == "benign":
        urldf.loc[len(urldf.index)] = [domain(row['url']), ct_hyphen(row['url']), ct_dot(row['url']), ct_at(row['url']), ct_digit(row['url']),
                                    ct_equal(row['url']),longer_than_7(row['url']),False,susp_words_func(row['url']),length(row['url']),
                                    extension(row['url']),entropy(row['url']),alexa_ngram_count(row['url']),dict_ngram_count(row['url']),
                                    False,False] 
    else:
        urldf.loc[len(urldf.index)] = [domain(row['url']), ct_hyphen(row['url']), ct_dot(row['url']), ct_at(row['url']), ct_digit(row['url']),
                                    ct_equal(row['url']),longer_than_7(row['url']),False,susp_words_func(row['url']),length(row['url']),
                                    extension(row['url']),entropy(row['url']),alexa_ngram_count(row['url']),dict_ngram_count(row['url']),
                                    squatting(row['url']),True] 



##### HTML CONTENT FEATURES #####


contentdf = pd.DataFrame(columns= ['domain','intlink', 'extlink', '.js','a href','a','meta', 'popup','iframe','link','button', 'label', 'emptyhref',
                                   'link href','script','div','ul','li','.php','h','img','href','form','p',
                                   'input','len','tags','suspicious_words', 'input_pass', 'input_button', 'input_checkbox', 'input_date',
                                   'input_email', 'input_image', 'input_submit', 'input_text', 'input_tel', 'input_radio', 'input_reset', 'input_search',
                                   'display_none', 'visibility_none', 'script_type_js', 'svg', 'social media', 'hidden', 'embed','aspx', 'addEventListener', 
                                   'source', 'blockquote', 'hasAttribute', 'getEntriesByName', 'media', 'accesskey', 'click', 'getAttribute', 
                                   'innerHTML', 'innerText', 'removeAttribute', 'removeEventListener', 'setAttribute', 'querySelectorAll', 'freeze', 'throwError', 
                                   'call', 'phishing'])


for index, row in datasetpd.iterrows():
    if row["label"] == "benign":
      contentdf.loc[len(contentdf.index)] = [domain(row['url']), ct_intlink(row['content'],domain(row['url'])), ct_extlink(row['content'],domain(row['url'])), 
                                            ct_js(row['content']), ct_a_href(row['content']),ct_a(row['content']),ct_meta(row['content']),ct_popup(row['content']),
                                            ct_iframe(row['content']),ct_link(row['content']),ct_button(row['content']),ct_label(row['content']),
                                            ct_empty_href(row['content']),ct_link_href(row['content']),ct_script(row['content']),ct_div(row['content']),ct_ul(row['content']),
                                            ct_li(row['content']),ct_php(row['content']),ct_h(row['content']),ct_img(row['content']),
                                            ct_href(row['content']),ct_form(row['content']),ct_p(row['content']),ct_input(row['content']),length(row['content']),
                                            totaltags(row['content']),ct_susp_words(row['content']), ct_input_password(row['content']), ct_input_button(row['content']),
                                            ct_input_checkbox(row['content']),ct_input_date(row['content']),ct_input_email(row['content']),ct_input_image(row['content']), 
                                            ct_input_submit(row['content']), ct_input_text(row['content']), ct_input_tel(row['content']), ct_input_radio(row['content']),
                                            ct_input_reset(row['content']), ct_input_search(row['content']), ct_display_none(row['content']), ct_visibility_none(row['content']),
                                            ct_script_type_js(row['content']), ct_svg(row['content']), ct_facebook(row['content']) + ct_twitter(row['content']) +
                                            ct_instagram(row['content']) + ct_youtube(row['content']) + ct_pinterest(row['content']) + ct_linkedin(row['content']),
                                            ct_hidden(row['content']), ct_embed(row['content']), ct_aspx(row['content']), ct_addEvent(row['content']), ct_source(row['content']),
                                            ct_blockquote(row['content']), ct_hasAttribute(row['content']),ct_getEntriesByName(row['content']), ct_media(row['content']), 
                                            ct_accessKey(row['content']), ct_click(row['content']), ct_getAttribute(row['content']), ct_innerHTML(row['content']), 
                                            ct_innerText(row['content']), ct_removeAttribute(row['content']), ct_removeEventListener(row['content']), 
                                            ct_setAttribute(row['content']), ct_querySelectorAll(row['content']), ct_freeze(row['content']), ct_throwError(row['content']), 
                                            ct_call(row['content']),False] 

    else:
      contentdf.loc[len(contentdf.index)] = [domain(row['url']), ct_intlink(row['content'],domain(row['url'])), ct_extlink(row['content'],domain(row['url'])), 
                                            ct_js(row['content']), ct_a_href(row['content']),ct_a(row['content']),ct_meta(row['content']),ct_popup(row['content']),
                                            ct_iframe(row['content']),ct_link(row['content']),ct_button(row['content']),ct_label(row['content']),
                                            ct_empty_href(row['content']),ct_link_href(row['content']),ct_script(row['content']),ct_div(row['content']),ct_ul(row['content']),
                                            ct_li(row['content']),ct_php(row['content']),ct_h(row['content']),ct_img(row['content']),
                                            ct_href(row['content']),ct_form(row['content']),ct_p(row['content']),ct_input(row['content']),length(row['content']),
                                            totaltags(row['content']),ct_susp_words(row['content']), ct_input_password(row['content']), ct_input_button(row['content']),
                                            ct_input_checkbox(row['content']),ct_input_date(row['content']),ct_input_email(row['content']),ct_input_image(row['content']), 
                                            ct_input_submit(row['content']), ct_input_text(row['content']), ct_input_tel(row['content']), ct_input_radio(row['content']),
                                            ct_input_reset(row['content']), ct_input_search(row['content']), ct_display_none(row['content']), ct_visibility_none(row['content']),
                                            ct_script_type_js(row['content']), ct_svg(row['content']), ct_facebook(row['content']) + ct_twitter(row['content']) +
                                            ct_instagram(row['content']) + ct_youtube(row['content']) + ct_pinterest(row['content']) + ct_linkedin(row['content']),
                                            ct_hidden(row['content']), ct_embed(row['content']), ct_aspx(row['content']), ct_addEvent(row['content']), ct_source(row['content']),
                                            ct_blockquote(row['content']), ct_hasAttribute(row['content']),ct_getEntriesByName(row['content']), ct_media(row['content']), 
                                            ct_accessKey(row['content']), ct_click(row['content']), ct_getAttribute(row['content']), ct_innerHTML(row['content']), 
                                            ct_innerText(row['content']), ct_removeAttribute(row['content']), ct_removeEventListener(row['content']), 
                                            ct_setAttribute(row['content']), ct_querySelectorAll(row['content']), ct_freeze(row['content']), ct_throwError(row['content']), 
                                            ct_call(row['content']),True] 


training_df = pd.merge(urldf, contentdf, left_index=True, right_index=True)
training_df = training_df.drop('domain_y',axis='columns')
training_df = training_df.rename(columns={'domain_x': 'domain'})
training_df = training_df.drop('phishing_x',axis='columns')
training_df = training_df.rename(columns={'phishing_y': 'phishing'})                                           


print(training_df.isnull().values.any())  # No missing values  -  False
print(training_df.dtypes)


ord_enc = OrdinalEncoder()
training_df["longer than 7"] = ord_enc.fit_transform(training_df[["longer than 7"]])
training_df["squatting"] = ord_enc.fit_transform(training_df[["squatting"]])
training_df["u_susp words"] = ord_enc.fit_transform(training_df[["u_susp words"]])
training_df["mimics"] = ord_enc.fit_transform(training_df[["mimics"]])
training_df["phishing"] = ord_enc.fit_transform(training_df[["phishing"]])




for index, row in training_df.iterrows():
  if row['extension'].isnumeric() :
    training_df['extension'][index] = '0'

training_df["extension"] = ord_enc.fit_transform(training_df[["extension"]])
print(len(training_df['extension'].unique()))  # Removing extensions that are integer


domains = training_df['domain']   # object -> float conversion
training_df = training_df.loc[:, training_df.columns != 'domain'].astype(float)

# min-max scalar
# sc = MinMaxScaler()
# training_df1 = sc.fit_transform(training_df.loc[:, training_df.columns != 'domain'])

print(training_df.columns)

print(training_df.head())
print(len(training_df))

training_df2 = training_df.drop(['input_reset','innerText','embed','input_text','accesskey','at','equal','input_checkbox','input_search','getEntriesByName', 'input_image', 'blockquote', 'removeAttribute', 'input_date', 'input_radio',
                                 'input_tel', 'hasAttribute', 'freeze', 'longer than 7', 'aspx', 'throwError','source','getAttribute',
                                  'innerHTML', 'input_email', 'visibility_none','display_none','input_submit','media','popup','addEventListener','script_type_js','input_pass', 'form','input_button','svg','label','querySelectorAll',
                                                             'call','click','removeEventListener','u_susp words','button', 'social media','hidden','input','ul','a href'], axis=1)

