import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_csv("/Users/Dejan Belusevic/Desktop/Programiranje/projects/book_data.csv")
df = df[["book_title", "book_desc", "book_rating_count"]]

data = df.sort_values(by = "book_rating_count", ascending = False)
top_5 = data.head()

import plotly.express as px

labels = top_5["book_title"]
values = top_5["book_rating_count"]
colors = ['gold','lightgreen']

fig = px.pie(top_5, values = values, names = labels, hole = 0.3)
fig.update_layout(title_text="Top 5 Rated Books")
fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=30,
                  marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

data = data.dropna()
feature = data["book_desc"].tolist()
vectorizer = TfidfVectorizer(max_df = 0.5, stop_words= "english")
tfidf_matrix = vectorizer.fit_transform(feature)
similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(data = data.index, index = data["book_title"]).drop_duplicates()

def book_recommendation(title, similarity = similarity):
    index = indices[title]
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[0:5]
    bookindices = [i[0] for i in similarity_scores]
    return data['book_title'].iloc[bookindices]

print(book_recommendation("Letters to a Secret Lover"))