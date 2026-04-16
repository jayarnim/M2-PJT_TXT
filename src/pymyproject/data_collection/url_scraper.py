import requests
import xml.etree.ElementTree as ET
import pandas as pd
import math
import time
from IPython.display import clear_output
from .const import ASSEMBLY, URL


def _counter(params):
    response = requests.get(url=URL, params=params)
    root = ET.fromstring(response.text)
    num_contents = int(root.findtext(".//list_total_count") or 0)
    return num_contents


def _scraper(params, num_pages, conf_col, date_col, url_col):
    data = []

    for page in range(1, num_pages+1):
        params["pIndex"] = page
        response = requests.get(url=URL, params=params)
        root = ET.fromstring(response.text)
        
        for row in root.iter("row"):
            sample = {
                conf_col: row.findtext("CONFER_NUM"),
                date_col: row.findtext("CONF_DATE"),
                url_col: row.findtext("PDF_LINK_URL"),
            }
            data.append(sample)

    return data


def engine(key, format, p_size, conf_col, date_col, url_col):
    data = []

    for DAE, YEARS in ASSEMBLY.items():
        for YEAR in YEARS:
            params = {
                "KEY": key,
                "Type": format,
                "pIndex": 1,
                "pSize": p_size,
                "DAE_NUM": str(DAE),
                "CONF_DATE": str(YEAR),
            }
            NUM_CONTENTS = _counter(params)

            if NUM_CONTENTS==0:
                continue
            else:
                kwargs = dict(
                    params=params,
                    num_pages=math.ceil(NUM_CONTENTS / p_size),
                    conf_col=conf_col, 
                    date_col=date_col, 
                    url_col=url_col,
                )
                data.extend(_scraper(**kwargs))
                print(f"{DAE} th (year: {YEAR}) completed")
            
            time.sleep(0.3)

        clear_output(wait=False)
    
    return data


def main(
    key, 
    format="xml", 
    p_size=100, 
    save_dir="./data/url", 
    conf_col="conf_num", 
    date_col="date", 
    url_col="url",
):
    # 데이터 수집
    kwargs = dict(
        key=key, 
        format=format, 
        p_size=p_size, 
        conf_col=conf_col, 
        date_col=date_col, 
        url_col=url_col,
    )
    data = engine(**kwargs)
    
    # 데이터프레임 생성
    df = pd.DataFrame(data)
    
    # 콘텐츠가 회의 단위가 아니라 회의의 섹션 단위로 구성되어 있으나
    # 섹션별 회의록이 나뉘어 있지 않고 동일한 url 을 중복하여 제공하고 있으므로 제거
    df = df.drop_duplicates(subset=[url_col])
    
    # 시계열 정렬
    df["date"] = pd.to_datetime(df[date_col])
    df = (
        df
        .sort_values(by=date_col, ascending=True)
        .reset_index(drop=True)
    )

    # 연도별 파싱하여 저장
    df["year"] = df[date_col].dt.year
    for year, group in df.groupby("year"):
        SAVE_PATH = f"{save_dir}/{year}.csv"
        group.to_csv(SAVE_PATH, index=False)
    
    print("API SCRAPING FINISHED")