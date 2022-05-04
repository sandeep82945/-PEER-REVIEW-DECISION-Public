from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
d={'positive':'pos','negative':'neg'}
def get_score(sentence,polarity):    
    vs = analyzer.polarity_scores(sentence)
    return vs[d[polarity]]



    