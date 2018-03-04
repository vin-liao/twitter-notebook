import pandas as pd
import numpy as np
import random
import zipfile
from sklearn.model_selection import train_test_split
from keras.preprocessing import text, sequence
from html import unescape


def preprocess(dataset):
	#preprocess dataset
	eyes = r"[8:=;]"
	nose = r"['`\-]?"

	#decode html entities
	dataset.text = dataset.text.apply(lambda x: unescape(x))

	dataset['text'] = dataset['text']\
	.str.replace(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*', '<url>')\
	.str.replace(r'@\w+', '<user>')\
	.str.replace(r'{}{}[)dD]+|[)dD]+{}{}'.format(eyes, nose, nose, eyes), '<smile>')\
	.str.replace(r'{}{}p+'.format(eyes, nose), '<lolface>')\
	.str.replace(r'{}{}\(+|\)+{}{}'.format(eyes, nose, nose, eyes), '<sadface>')\
	.str.replace(r'{}{}[\/|l*]'.format(eyes, nose), '<neutralface>')\
	.str.replace(r'/',' / ')\
	.str.replace(r'<3','<heart>')\
	.str.replace(r'[-+]?[.\d]*[\d]+[:,.\d]*', '<number>')\
	.str.replace(r'#\S+', '<hashtag>')\
	.str.replace(r'([!?.]){2,}', r'\1 <repeat>')\
	.str.replace(r'\b(\S*?)(.)\2{2,}\b', r'\1\2 <elong>')

	return dataset

def load_data(fraction=1, filepath='../data/Sentiment-Analysis-Dataset.zip'):
	zip_ref = zipfile.ZipFile(filepath, 'r')
	zip_ref.extractall('../data/')

	
	dataset = pd.read_csv('../data/Sentiment Analysis Dataset.csv', error_bad_lines=False, encoding='utf-8')

	dataset.dropna(axis=0, inplace=True)
	dataset = dataset.rename(index=str, columns={"SentimentText": "text", "Sentiment": "sentiment"})
	dataset = dataset.sample(frac=fraction, random_state=42)

	return dataset

def generate_data():
	dataset = load_data()
	dataset = preprocess(dataset)

	token = text.Tokenizer(filters='!"#$%&()*+,-./:;=?@[\]^_`{|}~\t\n')
	max_len = dataset['text'].str.len().max()

	#learn the vocabulary from all the text
	token.fit_on_texts(list(dataset['text']))
	vocab_size = len(token.word_index) + 1

	x_train, x_test, y_train, y_test = train_test_split(dataset['text'], dataset['sentiment'], test_size=0.015, shuffle=False, random_state=42)

	y_train = pd.get_dummies(y_train)
	y_test = pd.get_dummies(y_test)

	#encode
	x_train_enc = token.texts_to_sequences(x_train)
	x_test_enc = token.texts_to_sequences(x_test)


	#add zero padding
	x_train_enc_pad = sequence.pad_sequences(x_train_enc, maxlen=max_len)
	x_test_enc_pad = sequence.pad_sequences(x_test_enc, maxlen=max_len)

	return x_train_enc_pad, y_train, x_test_enc_pad, y_test