"""
Run this from project root path

python bin/sparkapp.py korean_job --config_path=./configs/korean_job.yaml
"""

import os
import yaml
from pyspark import SparkContext
from pyspark.rdd import RDD

from dps.spark.prep.korean_prep import (
    korean_word_ratio_filter,
    reduce_emoticon,
    replace_korean_pii,
    spam_words_filter,
    remove_html_tags,
    bad_words_filter,
    make_compat,
)
from dps.spark.prep.lang_agnostic_prep import (
    doc_len_filter,
    mean_word_len_filter,
    symbol_to_word_ratio_filter,
    bullet_ellipsis_filter,
    remove_whitespace,
    process_html_and_uri_text,
    replace_email_and_url,
    remove_repeated_text,
)
from dps.spark.spark_session import spark_session, spark_session_for_cluster
from dps.spark.utils.io_utils import read_line, to_json


def preprocess_text(input_text: str):
    processing_function_list = [
        process_html_and_uri_text, # br태그 -> \n
        reduce_emoticon, # 반복 이모티콘 2개로 줄임
        remove_whitespace, # 중복 공백 "[^\S\r\n\t\v\f]+” 을 모두 " " 로 변경
        replace_email_and_url,# email, url 가리기
        replace_korean_pii, # 카드번호, 주민번호, 전화번호, 통장번호 가리기
        spam_words_filter,# “공유하기”, “~~~기자였습니다”, “OOO뉴스 ~~~입니다”, “사진=OO뉴스” 형태 제거
        remove_html_tags, # “javascript”, “/”, “#”, “*”, … “뉴스”, “카카오”, “네이버”, “티스토리”, “저작권”, “구독”, “검색어” 등 제거
        remove_repeated_text, # 3회 이상 반복 단어 or 구문 제거
    ]

    for func in processing_function_list:
        input_text = func(input_text)

    if isinstance(input_text, str):
        processed_text = input_text
    else:
        processed_text = " ".join(input_text)

    return processed_text


def korean_job(config_path):
    with open(config_path) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    if conf['targets'] == ['all']:
        input_paths = f'{conf["base_dir"]}/*/*.jsonl'
    else:
        input_paths = ','.join([f'{conf["base_dir"]}/{t}/*.jsonl' for t in conf["targets"]])
    session_fn = spark_session_for_cluster if conf["is_cluster"] else spark_session

    with session_fn("korean text processing job") as spark:
        sc: SparkContext = spark.sparkContext

        # set heap memorty
        sc._conf.set('spark.driver.memory', '15g')

        print(sc.getConf().getAll())
        proc_rdd: RDD = (
            sc.textFile(input_paths).repartition(conf["n_dist"]) # pyspark dataframe을 n_dist만큼 분할 
            .flatMap(read_line) 
            .map(
                lambda x: dict(
                    text=make_compat( # NFC형태로 변경 및 단일 자음/모음 제거
                        x["text"],
                    ),
                )
            )
            .filter(
                lambda x: bad_words_filter( # /utils/korean_utils.py의 BAD_WORDS 목록으로 필터
                    x["text"],
                )
            )
            .filter(
                lambda x: doc_len_filter( #도큐먼트 길이로 필터 [50, 100000]
                    x["text"],
                    conf["min_doc_len"],
                    conf["max_doc_len"],
                )
            )
            .filter(
                lambda x: mean_word_len_filter( # 단어 평균 길이로 필터 [3, 10]
                    x["text"],
                    conf["min_mean_word_len"],
                    conf["max_mean_word_len"],
                )
            )
            .filter(
                lambda x: symbol_to_word_ratio_filter( # "…", ". . ."", "#"의 비율이 0.1 이상이면 필터 
                    x["text"],
                    conf["symbol_to_word_ratio"],
                )
            )
            .filter(
                lambda x: bullet_ellipsis_filter( # bullet으로 시작하는 문장이나 ellipsis로 끝나는 문장 비율이 0.9, 0.3 이상이면 필터
                    x["text"],
                    conf["bullet_point_ratio"],
                    conf["ellipsis_ratio"],
                )
            )
            .filter(
                lambda x: korean_word_ratio_filter( # 한글 글자 비율이 0.25 이하면 필터
                    x["text"],
                    conf["korean_word_ratio"],
                )
            )
            .map(
                lambda x: dict(
                    text=preprocess_text(
                        x["text"],
                    )
                )
            )
            # one more length filter
            # to exclude "" after preprocess_text()
            .filter(
                lambda x: doc_len_filter(
                    x["text"],
                    conf["min_doc_len"],
                    conf["max_doc_len"],
                )
            )
        )
        proc_rdd.repartition(conf["n_output"]).flatMap(to_json).saveAsTextFile(
            conf["output_dir"]
        )
