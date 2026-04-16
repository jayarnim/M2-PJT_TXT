import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel


def build_coherence_model(texts, vectorizer, lda, top_k):
    # gensim dictionary
    dictionary = Dictionary(texts)

    # extract words @ vectorizer
    feature_names = vectorizer.get_feature_names_out()

    # extract top N words @ each topics
    topics = []
    for topic in lda.components_:
        top_idx = topic.argsort()[-top_k:][::-1]
        top_words = [feature_names[i] for i in top_idx]
        topics.append(top_words)

    # coherence model definition
    kwargs = dict(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
    )
    coherence_model = CoherenceModel(**kwargs)

    return coherence_model


def engine(texts, dtm, vectorizer, date, num_topics, top_k, seed):
    kwargs = dict(
        n_components=num_topics,
        random_state=seed,
    )
    lda = LatentDirichletAllocation(**kwargs)
    doc_topic_mat = pd.DataFrame(
        data=lda.fit_transform(dtm),
        index=date,
        columns=[f"topic_{i+1}" for i in range(num_topics)],
    )

    kwargs = dict(
        texts=texts,
        vectorizer=vectorizer,
        lda=lda,
        top_k=top_k,
    )
    coh = build_coherence_model(**kwargs)
    score = coh.get_coherence()
    score_per_topic = coh.get_coherence_per_topic()

    return dict(
        lda=lda,
        doc_topic_mat=doc_topic_mat,
        coh=coh,
        score=score,
        score_per_topic=score_per_topic,
    )


def main(
    df, 
    min_df=0.01,
    max_df=0.5,
    num_topics=range(10,31), 
    top_k=10, 
    token_col="words",
    seed=42,
):
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=True)
    date = df.index

    kwargs = dict(
        min_df=min_df,
        max_df=max_df,
    )
    vectorizer = CountVectorizer(**kwargs)

    docs = (
        df[token_col]
        .apply(lambda x: " ".join(x))
        .tolist()
    )
    dtm = vectorizer.fit_transform(docs)

    result = dict()

    for k in num_topics:
        kwargs = dict(
            texts=df[token_col].tolist(), 
            dtm=dtm, 
            vectorizer=vectorizer, 
            date=date,
            num_topics=k, 
            top_k=top_k, 
            seed=seed,
        )
        result[k] = engine(**kwargs)

        print(
            f"NUM TOPIC: {k}",
            f"COH SCORE: {result[k]['score']:.4f}",
            sep="\t",
        )

    print("LDA FINISHED")

    return result, dtm, vectorizer