from pathlib import Path
from tqdm import tqdm
from IPython.display import clear_output
import pandas as pd
import re
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords


def tokenizer(txt):
    PATTERN = r"(@[A-Za-z0-9]+)|(https?:\/\/\S+)|([^가-힣 \t])"
    p = re.compile(PATTERN)
    analyzer = Kiwi(typos='basic_with_continual')
    
    result = p.sub('', txt)
    result = analyzer.space(result)
    result = analyzer.split_into_sents(result)
    result = [
        analyzer.tokenize(sent.text, normalize_coda=True)
        for sent in result
    ]
    
    return result


def cleanser(txt):
    stopwords = Stopwords()

    result = []

    for sent in txt:
        tokens = []

        for token in sent:
            # 일반명사, 고유명사, 외래어를 제외한 토큰 제거
            COND_TAG = token.tag in {"NNG", "NNP", "SL"}
            # 길이가 1 이하인 토큰 제거
            COND_LEN = len(token.form) >= 2
            # 불용어 제거
            COND_STOPWORDS = (token.form, token.tag) not in stopwords.stopwords

            if COND_TAG and COND_LEN and COND_STOPWORDS:
                tokens.append(token.form)

        result.extend(tokens)
    
    return result


def engine(file, save_dir, txt_col, token_col, date_col):
    FILE_NAME = Path(file).stem

    # df
    df = pd.read_csv(file)
    df[token_col] = None
    
    # tqdm obj    
    kwargs = dict(
        iterable=df.iterrows(),
        desc=f"{FILE_NAME} PREPROCESS",
        total=len(df),
    )

    # 전처리
    for i, row in tqdm(**kwargs):
        txt = row[txt_col]
        tokenized = tokenizer(txt)
        filtered = cleanser(tokenized)
        df.at[i, token_col] = filtered

    # 불필요한 칼럼 제거
    df = df.drop(columns=[txt_col])

    # 시계열 정렬
    df[date_col] = pd.to_datetime(df[date_col])
    df = (
        df
        .sort_values(by=date_col, ascending=True)
        .reset_index(drop=True)
    )
    
    # 추출된 텍스트를 연도별로 .csv 저장
    SAVE_PATH = Path(save_dir) / f"{FILE_NAME}.pkl"
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(SAVE_PATH)


def main(
    load_dir="./data/txt", 
    save_dir="./data/pre", 
    txt_col="speech", 
    token_col="words", 
    date_col="date",
):
    # 텍스트가 저장된 년도별 .csv 파일 리스트 불러오기
    FILES = list(Path(load_dir).glob("*.csv"))

    for file in FILES:
        kwargs = dict(
            file=file, 
            save_dir=save_dir, 
            txt_col=txt_col, 
            token_col=token_col,
            date_col=date_col,
        )
        engine(**kwargs)
        clear_output(wait=False)

    print("TXT PREPROCESSING FINISHED")