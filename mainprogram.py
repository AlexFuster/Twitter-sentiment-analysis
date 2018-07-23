from nltk.corpus import wordnet
from os import listdir
import pickle
import auxiliar_module
import tweepy
import numpy as np
import matplotlib.pyplot as plt

ckey = 'XXXXXXXXX'
csecret = 'XXXXXXXXX'
auth = tweepy.OAuthHandler(consumer_key=ckey, consumer_secret=csecret)
api = tweepy.API(auth)

chartlabels = ['Negative','Positive']
colors=[(1,0,47/255),(0,1,47/255)]
f=open('AFINN-111.txt','r')
afin=f.read().split('\n')
afindic={}
for w in afin:
	aux = w.split('\t')
	afindic[aux[0]]=int(aux[1])
f.close()


def obtain_score(tweet_words):
	cont=0
	score=0
	for w in tweet_words:
		if w in afindic.keys():
			score+=afindic[w]
			cont+=1
		
		else:
			found=False
			for syn in wordnet.synsets(w):
				for l in syn.lemmas():
					if not found and l.name() in afindic.keys():
						score+=afindic[l.name()]
						cont+=1
						found=True
	
	return score

if __name__=='__main__':
	clasifiers=[]
	clasifier_files = [f for f in listdir() if f.startswith('clasifier') and f.endswith('.pickle') ]
	for clasifier_file in clasifier_files:
		load_clasifier=open(clasifier_file,'rb')
		clasifiers.append(pickle.load(load_clasifier))
		load_clasifier.close()
	load_pals=open('pals.pickle','rb')
	pals=pickle.load(load_pals)
	load_pals.close()
	weights=[1.5,1]
	while True:
		query=input('query: ')
		if query=='':
			break
		rmquery=input('remove query from tweets? [Y/N]: ').lower()
		
		if rmquery.startswith('y'):
			rmquery=query.replace('"','')
		else:
			rmquery=''

		show_tweets=input('show tweets? [Y/N]: ').lower().startswith('y')
		
		ts=[]
		tweets=[]
		emoji_scores=[]

		for tweet_info in tweepy.Cursor(api.search, q=query, lang = 'en', tweet_mode='extended').items(100):
			if 'retweeted_status' in dir(tweet_info):
				tweet=tweet_info.retweeted_status.full_text
			else:
				tweet=tweet_info.full_text
			tweet1,emoji_neg_score = auxiliar_module.remove_emojis(tweet,0)
			tweet1,emoji_pos_score = auxiliar_module.remove_emojis(tweet1,1)
			tweet_words=auxiliar_module.clean_tweet(tweet1,rmquery)
			if len(tweet_words)>0:
				ts.append(tweet_words)
				emoji_scores.append(emoji_neg_score+emoji_pos_score)
				if show_tweets:
					tweets.append(tweet)
		testset=auxiliar_module.rellenar_arrays(ts,pals)
		
		emoji_scores=3*np.sign(np.array(emoji_scores))
		predictions=[weights[i]*clasifiers[i].predict(testset) for i in range(len(clasifiers))]
		predictions.append(np.sign(np.array([obtain_score(tsitem) for tsitem in ts])))
		predictions.append(emoji_scores)
		result=np.sign(sum(predictions))
		neg_rate=np.sum(result==-1)/len(ts)
		pos_rate= 1-neg_rate
		if show_tweets:
			for i in range(len(tweets)):
				print('<<'+tweets[i]+'>>',result[i])
				#print('<<'+tweets[i]+'>>',ts[i],predictions[0][i],predictions[1][i],predictions[2][i],emoji_scores[i],result[i])
				
		print('Tweets analized: ',len(ts))
		print('negtive rate,positive rate: ',neg_rate,pos_rate)
		fig1, ax1 = plt.subplots()
		ax1.pie([neg_rate,pos_rate], labels=chartlabels, colors=colors ,autopct='%1.1f%%')
		ax1.set_title(query)
		ax1.axis('equal')  
		plt.show()
