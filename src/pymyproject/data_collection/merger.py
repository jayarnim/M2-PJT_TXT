import pandas as pd
from pathlib import Path


def main(
    load_dir="./data/pre", 
    save_dir="./data",
    save_file="national_assembly", 
    token_col="words",
    date_col="date", 
):
    # .csv 불러오기
    FILES = list(Path(load_dir).glob("*.pkl"))

    # 데이터프레임 병합
    df_list = [pd.read_csv(file) for file in FILES]
    merged_df = pd.concat(df_list, ignore_index=True)

    # 시계열 정렬
    merged_df[date_col] = pd.to_datetime(merged_df[date_col])
    merged_df = (
        merged_df
        .sort_values(by=date_col, ascending=True)
        .reset_index(drop=True)
    )

    # 칼럼 선별
    SAVE_COLS = [date_col, token_col]
    merged_df = merged_df[SAVE_COLS]

    # 저장
    SAVE_PATH = Path(save_dir) / f"{save_file}.pkl"
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_pickle(SAVE_PATH)

    print("MERGE FINISHED")