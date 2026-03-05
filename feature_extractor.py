import pandas as pd
from pymongo import MongoClient
#import dnstwist
import re
import math
from collections import Counter
import sklearn.feature_extraction
import numpy as np
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




"""
squatting = [] 
def squatting_list (domain1): 
  data = dnstwist.run(domain=domain1, registered=True, format='null')
  for i in range(len(data)):
    if data[i]['fuzzer'] == 'bitsquatting' or data[i]['fuzzer'] == 'homoglyph' or data[i]['fuzzer'] == 'hyphenation' or data[i]['fuzzer'] == 'insertion':
      squatting.append(data[i]['domain'])
    if data[i]['fuzzer'] == 'omission' or  data[i]['fuzzer'] == 'repetition' or data[i]['fuzzer'] == 'replacement' or data[i]['fuzzer'] == 'subdomain' :
      squatting.append(data[i]['domain'])
    if data[i]['fuzzer'] == 'transposition' or data[i]['fuzzer'] == 'vowel-swap' or data[i]['fuzzer'] == 'various':
      squatting.append(data[i]['domain'])

for i in range(50):
  squatting_list(datasetpd[datasetpd["label"] == "benign"]['url'][i])

with open('./sq_list.txt', 'w') as sq:
  for word in squatting:
    sq.write(word)
    sq.write('\n')
"""



##### URL FEATURES #####

textfile = open(r'./suspwords.txt')
susp_words = []
for line in textfile:
  susp_words.append(line[:-1])

sq_list = pd.read_csv(r'./sq_list.txt', header=None, names=['squatting'])

def squatting(domain):
  for i in sq_list['squatting']:
    if i[:i.rfind('.')] in domain and len(i[:i.rfind('.')]) > 3:
      return True
  return False

def urls(url):
  return url

def ct_hyphen (url):
  return url.count('-')

def ct_dot (url):
  return url.count('.')

def ct_equal (url):
  return url.count('=')

def ct_digit(url):
  count = 0
  for i in url:
    if i.isdigit():
      count+=1
  return count

def ct_at(url):
  return url.count('@')

def length(url):
  return len(url)

def longer_than_7(url):
  if len(url) > 7:
    return True
  else:
    return False

def mimics(url):       # combosquatting
  url = url.lower()
  if url == 'bt' or url == 't':
      return True
  for i in datasetpd[datasetpd["label"] == "benign"]['domain']:
    i = i.split('.')[0]
    if len(i) > 2 :
      if i in url and i != 't':
        return True
  return False

def domain(url):
  a = str(re.findall(r'^(?:http:\/\/|www\.|https:\/\/)([^\/]+)', url))[2:-2]
  if a == '':
    a = url
  return a

def extension(url): 
    b = str(url).split('.')[-1]
    return b

def susp_words_func (url): # purchase, buy, accept, enable, 
  url = url.lower()
  for i in range(len(susp_words)):
    if susp_words[i] in url:
       return True
  return False


def entropy(s):
    p, lns = Counter(s), float(len(s))
    return( -sum( count/lns * math.log(count/lns, 2) for count in p.values()))


def alexa_ngram_count(domain):
    alexa_match = alexa_counts * alexa_vc.transform([domain]).T 
    return float(alexa_match)


def dict_ngram_count(domain):
    dict_match = dict_counts * dict_vc.transform([domain]).T
    return float(dict_match)


alexa_vc = sklearn.feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(3,5), min_df=1e-4, max_df=1.0)
counts_matrix = alexa_vc.fit_transform(datasetpd[datasetpd["label"] == "benign"]['domain'])
alexa_counts = np.log10(counts_matrix.sum(axis=0).getA1())

word_dataframe = pd.read_csv(
    './words.txt',
    names=['word'],
    header=None,
    dtype={'word': str},
    encoding='utf-8',
    sep=',',  # Use the correct delimiter if needed
    on_bad_lines='skip'  # Skip malformed lines
)

word_dataframe = word_dataframe[word_dataframe['word'].map(lambda x: str(x).isalpha())]
word_dataframe = word_dataframe.applymap(lambda x: str(x).strip().lower())
word_dataframe = word_dataframe.dropna()
word_dataframe = word_dataframe.drop_duplicates()

dict_vc = sklearn.feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(3,5), min_df=1e-5, max_df=1.0)
counts_matrix1 = dict_vc.fit_transform(word_dataframe['word'])
dict_counts = np.log10(counts_matrix1.sum(axis=0).getA1())


##### HTML CONTENT FEATURES #####


def ct_js(text):
  return text.count(".js")

def ct_a_href(text):
  return text.count("<a href")

def ct_a(text):
  return text.count("<a")

def ct_meta(text):
  return text.count("<meta")

def ct_popup(text):
  return text.count("popup") + text.count("Popup")

def ct_iframe(text):
  return text.count("<iframe")

def ct_link(text):
  return text.count("<link")

def ct_link_href(text):
  return text.count("<link href")

def ct_script(text):
  return text.count("<script")

def ct_div(text):
  return text.count("<div")

def ct_ul(text):
  return text.count("<ul") 

def ct_li(text):
  return text.count("<li") 

def ct_php(text):
  return text.count(".php") 

def ct_h(text):
  ct = text.count("<h1") + text.count("<h2") + text.count("<h3") + text.count("<h4") 
  return ct

def ct_img(text):
  return text.count("<img") 

def ct_href(text):
  ct = str(re.findall(r'(href=".*?")', text)).split(',')
  if ct == ['[]']:
    return 0
  return len(ct)

def ct_form(text):
  return text.count("<form action") 

def ct_p(text):
  return text.count("<p") 

def ct_input(text):
  return text.count("<input") 
  
def length(text):
  return len(text)

def totaltags(text):
  return text.count("</")

def ct_susp_words(text): 
  text = text.lower()
  ct = 0
  for i in range(len(susp_words)):
    if susp_words[i] in text:
       ct = ct+1
  return ct

def ct_aspx(text):
  return text.count(".aspx")

def ct_embed(text):
  return text.count("<embed")

def ct_button(text):
  return text.count("<button")

def ct_label(text):
  return text.count("<label")

def ct_input_password(text):
  return text.count("type=\"password\"")

def ct_input_button(text):
  return text.count("type=\"button\"")

def ct_input_checkbox(text):
  return text.count("type=\"checkbox\">")

def ct_input_date(text):
  return text.count("type=\"date\"")

def ct_input_email(text):
  return text.count("type=\"email\"")

def ct_input_image(text):
  return text.count("type=\"image\"")

def ct_input_submit(text):
  return text.count("type=\"submit\"")

def ct_input_text(text):
  return text.count("type=\"text\"")

def ct_input_tel(text):
  return text.count("type=\"tel\"")

def ct_input_radio(text):
  return text.count("type=\"radio\"")

def ct_input_reset(text):
  return text.count("type=\"reset\"")

def ct_input_search(text):
  return text.count("type=\"search\"")

def ct_display_none(text):
  return text.count("display: none;")

def ct_visibility_none(text):
  return text.count("visibility: hidden;")

def ct_script_type_js(text):
  return text.count("<script type=\"text/javascript\"")

def ct_svg(text):
  return text.count("</svg>")

def ct_facebook(text):
  return text.lower().count("facebook")

def ct_twitter(text):
  return text.lower().count("twitter")

def ct_instagram(text):
  return text.lower().count("instagram")

def ct_youtube(text):
  return text.lower().count("youtube")

def ct_pinterest(text):
  return text.lower().count("pinterest")

def ct_linkedin(text):
  return text.lower().count("linkedin")

def ct_hidden(text):
  return text.count('hidden') + text.count(".hidden") + text.count("#hidden") + text.count("\"hidden") + text.count("*[visibility=\"none\"]") + text.count("*[display=\"none\"]")

def ct_empty_href(text):
  return text.count("href=\"#") + text.count("href=\"#skip\"") + text.count("href=\"#content\"") + text.count("href=\"javascript:;") + text.count("href=\"javascript::void(0);") + text.count("href=\"javascript:void(0);") + text.count("href=\"\\\"") + text.count("href=''")

def ct_intlink(text, domain1):
  counter = 0
  if ct_href(text) == 0:
    return 0
  a = str(re.findall(r'(href=".*?")', text))
  a = a.split(',')
  b = []
  for i in range(len(a)):
    link = str(re.findall(r'(".*?")', a[i]))
    link = link[3:-3]
    b.append(link)
  for i in range(len(b)):
    if domain(b[i]) == domain1 or domain(b[i]) == '':
      counter = counter+1
  return counter

def ct_extlink(text,domain1):
   return ct_href(text) - ct_intlink(text,domain1)

def ct_addEvent(text):
  return text.count('addEventListener')

def ct_source(text):
  return text.count("<source")

def ct_blockquote(text):
  return text.count("<blockquote")

def ct_hasAttribute(text):
  return text.count('hasAttribute') + text.count('hasAttributes')

def ct_getEntriesByName(text):
  return text.count('getEntriesByName')

def ct_media(text):
  return text.count('@media')

def ct_accessKey(text):
  return text.count('accessKey')

def ct_click(text):
  return text.count('click(')

def ct_getAttribute(text):
  return text.count('getAttribute(')

def ct_innerHTML(text):
  return text.count('innerHTML')

def ct_innerText(text):
  return text.count("innerText")

def ct_removeAttribute(text):
  return text.count("removeAttribute")

def ct_removeEventListener(text):
  return text.count("removeEventListener")

def ct_setAttribute(text):
  return text.count("setAttribute")

def ct_querySelectorAll(text):
  return text.count("querySelectorAll") + text.count("querySelector")

def ct_freeze(text):
  return text.count("freeze")

def ct_throwError(text):
  return text.count("throw Error")

def ct_call(text):
  return text.count("call(")

