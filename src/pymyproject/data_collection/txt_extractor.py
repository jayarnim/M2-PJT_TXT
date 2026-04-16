from pathlib import Path
import requests
import pdfplumber
from io import BytesIO
import time
from tqdm import tqdm
import pandas as pd
from IPython.display import clear_output


def pdf_buffer(url):
    kwargs = dict(
        url=url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response = requests.get(**kwargs)
    buffer = BytesIO(response.content)
    return buffer


def txt_extractor(buffer):
    texts = []
    
    with pdfplumber.open(buffer) as pdf:
        for page in pdf.pages:
            WIDTH = page.width
            HEIGHT = page.height
            BBOX_LEFT = (0, 0, WIDTH/2, HEIGHT)
            BBOX_RIGHT = (WIDTH/2, 0, WIDTH, HEIGHT)

            for bbox in [BBOX_LEFT, BBOX_RIGHT]:
                part = page.crop(bbox=bbox)
                text = part.extract_text()
                if text:
                    texts.append(text)
    
    return " ".join(texts)


def engine(file, save_dir, url_col, txt_col, date_col):
    FILE_NAME = Path(file).stem    

    # df
    df = pd.read_csv(file)
    df[txt_col] = None

    # tqdm obj    
    kwargs = dict(
        iterable=df.iterrows(),
        desc=f"{FILE_NAME} EXTRACTION",
        total=len(df),
    )

    # pdf 파일을 임시 열람하여 텍스트 추출
    for i, row in tqdm(**kwargs):
        buffer = pdf_buffer(row[url_col])
        txt = txt_extractor(buffer)
        df.at[i, txt_col] = txt
        print(f"{i+1}/{len(df)} completed")
        time.sleep(0.3)

    # 불필요한 칼럼 제거
    df = df.drop(columns=[url_col])

    # 시계열 정렬
    df["date"] = pd.to_datetime(df[date_col])
    df = (
        df
        .sort_values(by=date_col, ascending=True)
        .reset_index(drop=True)
    )

    # 추출된 텍스트를 연도별로 .csv 저장
    SAVE_PATH = Path(save_dir) / f"{FILE_NAME}.csv"
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAVE_PATH, index=False)


def main(
    load_dir="./data/url", 
    save_dir="./data/txt", 
    url_col="url", 
    txt_col="speech", 
    date_col="date",
):
    # url 정보가 저장된 년도별 .csv 파일 리스트 불러오기
    FILES = list(Path(load_dir).glob("*.pkl"))

    for file in FILES:
        kwargs = dict(
            file=file, 
            save_dir=save_dir, 
            url_col=url_col, 
            txt_col=txt_col, 
            date_col=date_col,
        )
        engine(**kwargs)
        clear_output(wait=False)

    print("TXT EXTRACTION FINISHED")