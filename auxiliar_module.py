from nltk.stem import WordNetLemmatizer
import nltk.tokenize
from nltk.corpus import stopwords
import numpy as np

stopwords = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

f=open('emojis','r')
emojis=f.read().split(',')
emojis=[emojis[0].split(' '),emojis[1].split(' ')]
f.close()

def remove_emojis(tweet,i):
	aux=tweet
	emoji_score=0
	for emoji in emojis[i]:
		if aux.startswith(emoji) or aux.endswith(emoji) or (' '+emoji+' ') in aux:
			emoji_score+=1
		aux = aux.replace(emoji,'')
	if i==0:
		emoji_score*=-1
	return aux,emoji_score

def clean_tweet(tweet,rmquery=''):
	aux=tweet.lower()
	if rmquery:
		aux=aux.replace(rmquery.lower(),'')
	if tweet.startswith('RT '):
		aux=aux[3:]
	usus=aux.split('@')
	aux=usus[0]
	for w in usus[1:]:
		try:
			aux+=w[w.index(' ')+1:]
		except(ValueError):
			pass
	tweet_words=[]
	for w in nltk.word_tokenize(aux):
		splittedw = w.split('-')
		is_composed = len(splittedw)>1
		if is_composed:
			for splittedpart in splittedw:
				is_composed = is_composed and splittedpart.isalpha()
		if (w.isalpha() or is_composed) and w not in stopwords and 'http' not in w and len(w)>1:
			lemma=lemmatizer.lemmatize(w)
			tweet_words.append(lemma)
	return tweet_words


def rellenar_arrays(t,pals):
	tset=np.zeros((len(t),len(pals)),dtype=bool)
	for i in range(len(t)):
		auxset=set(t[i])
		for w in auxset:
			try:
				ind=pals.index(w)
				tset[i,ind]=True
			except(ValueError):
				pass
	return tset