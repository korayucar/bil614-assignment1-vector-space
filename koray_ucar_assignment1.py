import json
import time
import logging

logging.basicConfig(level=logging.INFO)


def main():
    documents = parse_input_file()
    documents = filter_out_entities_without_tr_field(documents)
    documents = remove_stop_words(documents)

    pass


def time_usage(func):
    def wrapper(*args, **kwargs):
        logging.info(func.__name__ + " started. ")
        beg_ts = time.time()
        retval = func(*args, **kwargs)
        end_ts = time.time()
        logging.info(func.__name__ + " finished.(took: %f seconds)" % (end_ts - beg_ts))
        return retval
    return wrapper


@time_usage
def parse_input_file():
    # parse input file
    with open('thesisjson.txt') as json_data:
        return json.load(json_data)


@time_usage
def filter_out_entities_without_tr_field(documents):
    return [x for x in documents if 'tr' in x]


@time_usage
def remove_stop_words(documents):
    stoplist = set(
        unicode("de da gibi ki ve bir ben sen o bu \u015fu ile hangi tez \uxf6zet", 'utf-8').split())
    for document in documents:
        document['trWithoutStopWord'] = [word for word in document['tr'].lower().split() if word not in stoplist]
    return documents


if __name__ == "__main__": main()
