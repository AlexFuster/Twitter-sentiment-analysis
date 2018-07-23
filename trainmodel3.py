import nltk
from nltk.corpus import twitter_samples
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import numpy as np
import random
import auxiliar_module
import pickle

ids=twitter_samples.fileids()
clasifiers=[MultinomialNB(),LinearSVC()]

def obtain_words(i):
	tweets=twitter_samples.strings(ids[i])
	tweets_words=[]
	all_words=[]
	for tweet in tweets:
		aux,_ = auxiliar_module.remove_emojis(tweet,i)
		aux=auxiliar_module.clean_tweet(aux)
		if len(aux)>0:
			tweets_words.append(aux)
			all_words+=aux

	return all_words,tweets_words

if __name__=='__main__':
	all_words,neg=obtain_words(0)
	all_words2,pos=obtain_words(1)
	all_words+=all_words2
	talla=min(len(neg),len(pos))
	dist=nltk.FreqDist(all_words)
	pals=sorted(dist.keys(), key= lambda x: dist[x], reverse=True)[:3000]
	save_pals=open('pals.pickle','wb')
	pickle.dump(pals,save_pals)
	save_pals.close()
	random.shuffle(neg)
	random.shuffle(pos)
	tr=neg[:talla]+pos[:talla]
	trainset=auxiliar_module.rellenar_arrays(tr,pals)
	trlabels=np.ones(2*talla,dtype=int)
	trlabels[:talla]=-1
	for i in range(len(clasifiers)):
		clasifiers[i].fit(trainset,trlabels)
		save_clasifier=open('clasifier'+str(i)+'.pickle','wb')
		pickle.dump(clasifiers[i],save_clasifier)
		save_clasifier.close()
		