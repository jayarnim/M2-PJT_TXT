import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def plot_coh(result, figsize):
    Y = [obj["score"] for k, obj in result.items()]
    X = range(len(result))
    K_LIST = list(result.keys())

    fig = plt.figure(figsize=figsize)

    plt.plot(X, Y, marker='o')

    for x, y in zip(X, Y):
        kwargs = dict(
            x=x, 
            y=y+0.001, 
            s=f"{y:.4f}", 
            ha="center", 
            rotation=50,
        )
        plt.text(**kwargs)

    plt.xticks(ticks=X, labels=K_LIST)
    
    plt.xlabel("number of topics")
    plt.ylabel("coherence score")
    plt.title("coherence score by number of topics")
    
    plt.grid(True)

    plt.tight_layout()

    return fig


def plot_coh_per_topic(result, num_topics, figsize):
    Y = result[num_topics]["score_per_topic"]
    X = range(num_topics)
    TOPIC_LABELS = [f"T{i+1}" for i in range(num_topics)]

    fig = plt.figure(figsize=figsize)

    plt.bar(x=X, height=Y)
    plt.axhline(y=0.6, color="red")

    for x, y in zip(X, Y):
        kwargs = dict(
            x=x, 
            y=y+0.01, 
            s=f"{y:.2f}", 
            ha="center", 
            rotation=50,
        )
        plt.text(**kwargs)

    plt.xticks(ticks=X, labels=TOPIC_LABELS, rotation=45)
    
    plt.xlabel("topic number")
    plt.ylabel("coherence score")
    plt.title(f"coherence score per topic (k={num_topics})")
    
    plt.tight_layout()

    return fig


def plot_topic_similarity(result, num_topics):
    lda = result[num_topics]["lda"]
    topic_word_mat = lda.components_
    similarity = np.corrcoef(topic_word_mat)

    TOPIC_LABELS = [f"T{i+1}" for i in range(num_topics)]
    COLOR_MAP = "coolwarm"

    fig = plt.figure(figsize=(10, 8))

    im = plt.imshow(X=similarity, cmap=COLOR_MAP)

    plt.colorbar(mappable=im, label="correlation")

    plt.xticks(ticks=range(num_topics), labels=TOPIC_LABELS, rotation=45)
    plt.yticks(ticks=range(num_topics), labels=TOPIC_LABELS)

    plt.xlabel("topic number")
    plt.ylabel("topic number")
    plt.title(f"topic similarity heatmap (k={num_topics})")

    plt.tight_layout()
    
    return fig


def top_words_per_topic(result, vectorizer, num_topics, top_k):
    topic_word_mat = result[num_topics]["lda"].components_
    feature_names = vectorizer.get_feature_names_out()

    for i, topic in enumerate(topic_word_mat):
        top_words = [
            feature_names[j] 
            for j in topic.argsort()[-top_k:][::-1]
        ]
        print(i+1, top_words)


def wordcloud_per_topic(result, vectorizer, num_topics, nrows, ncols, figsize):
    lda = result[num_topics]["lda"]
    topic_word_mat = lda.components_

    FEATURE_NAMES = vectorizer.get_feature_names_out()
    FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
    BACKGROUND_COLOR = "white"

    fig, axes = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=figsize,
        squeeze=False,
    )

    for ax, topic_idx in zip(axes.flatten(), range(num_topics)):
        word_freq = {
            FEATURE_NAMES[i]: topic_word_mat[topic_idx][i]
            for i in range(len(FEATURE_NAMES))
        }
        kwargs = dict(
            font_path=FONT_PATH,
            background_color=BACKGROUND_COLOR,
        )
        wc = WordCloud(**kwargs).generate_from_frequencies(word_freq)

        ax.imshow(wc)
        ax.axis("off")
        ax.set_title(f"Topic {topic_idx+1}")

    for ax in axes.flatten()[num_topics:]:
        ax.axis("off")

    plt.tight_layout()
    
    return (fig, axes)


def plot_annual_freq_per_topic(result, num_topics, top_k, nrows, ncols, figsize):
    doc_topic_mat = result[num_topics]["doc_topic_mat"]
    
    # date column
    doc_topic_mat.index = pd.to_datetime(doc_topic_mat.index)
    doc_topic_mat = doc_topic_mat.sort_index(ascending=True)
    date = doc_topic_mat.index

    YEAR_MIN = date.year.min()
    YEAR_MAX = date.year.max()

    fig, axes = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=figsize,
        squeeze=False,
    )

    for ax, topic_idx in zip(axes.flatten(), range(num_topics)):
        TOP_IDX = doc_topic_mat.iloc[:, topic_idx].argsort()[::-1][:top_k]
        TOP_DATE = date[TOP_IDX].year
        COUNTS = TOP_DATE.value_counts().sort_index()

        ax.plot(COUNTS.index, COUNTS.values, marker='o')
        ax.set_xlim(YEAR_MIN, YEAR_MAX)
        ax.set_title(f'Topic {topic_idx+1}')
        ax.grid(True)

    for ax in axes.flatten()[num_topics:]:
        ax.axis("off")

    TITLE = f'yearly counts of top {top_k} topic-dominant documents (k={num_topics})'
    plt.suptitle(t=TITLE, fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    return (fig, axes)