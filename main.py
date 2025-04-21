from sklearn.metrics.pairwise import cosine_similarity as CS
from sklearn.feature_extraction.text import CountVectorizer as CVR
import pandas as pd

moviedata={
    "title":["Transformers","Avengers Endgames","Interstellers","Dark Knight","Openheimer"],
    "genre":["sci fi Action Thriller","Thriller","Action Thriller","Tragedy Drama","Historical Drama Documentary"]
}

Moviedata=pd.DataFrame(moviedata)
#print(Moviedata)

vectorizer=CVR()
genre_matrix=vectorizer.fit_transform(Moviedata["genre"])
#print(genre)

similarity=CS(genre_matrix)
movieindex=Moviedata[Moviedata["title"]=="Openheimer"].index[0]
#print(movieindex)

score=list(enumerate(similarity[movieindex]))
score=sorted(score, key=lambda x :x[1],reverse=True)[1:4]
#print(score)


recommended_movie=[Moviedata.iloc([A[0]]) for A in score]
print(recommended_movie)