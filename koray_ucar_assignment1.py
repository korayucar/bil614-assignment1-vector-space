import json
import math
import random
import time
import logging
from inspect import getouterframes, currentframe

FIELD_NAME_CONTENT_OF_INTEREST = "tr"
FIELD_NAME_CONTENT_WITHOUT_STOPWORDS = "trWithoutStopWord"
FIELD_NAME_WORD_COUNT_MAP = "wordCountsInDocument"
FIELD_NAME_TF_IDF_MAP = "tfIdf"
FIELD_NAME_DOCUMENT_MAGNITUDE = "documentMagnitude"
FIELD_NAME_CLOSEST_CENTROID = "closestCentroidIndex"
EPSILON_OF_CENTROID_EQUALITY = 0.001
MAX_ITERATIONS_UNTIL_CONVERGENCE = 20
# INPUT_FILE_NAME = "thesisjson.txt"
INPUT_FILE_NAME = "fast_subset.txt"

idfCache = dict()

logging.basicConfig(format='%(asctime)-15s%(message)s  ',level=logging.INFO)


def main():
    documents = parse_input_file()
    documents = filter_out_entities_without_tr_field(documents)
    remove_stop_words(documents)
    count_word_occurences(documents)
    number_of_documents_terms_exists_in = calculate_commonality_of_terms_by_counting_number_of_documents_it_exist(
        documents)
    calculate_tf_idf(documents, number_of_documents_terms_exists_in)
    pre_calculate_document_vector_magnitudes(documents)
    normalize_documents_to_unit_sphere(documents)
    k_means_for_range_of_k(documents, 1,22)
    pass


def time_usage(func):
    def wrapper(*args, **kwargs):
        logging_prefix = calculate_logging_prefix_for_better_visualization()
        logging.info(logging_prefix + func.__name__ + " started. ")
        beg_ts = time.time()
        retval = func(*args, **kwargs)
        end_ts = time.time()
        logging.info(logging_prefix + func.__name__ + " finished.(took: %f seconds)" % (end_ts - beg_ts))
        return retval

    return wrapper


def calculate_logging_prefix_for_better_visualization():
    return ("   " * sum([1 for stack in getouterframes(currentframe(1)) if stack[3] == 'wrapper']))


@time_usage
def normalize_documents_to_unit_sphere(documents):
    for doc in documents:
        for key, value in doc[FIELD_NAME_TF_IDF_MAP].items():
            doc[FIELD_NAME_TF_IDF_MAP][key] = value / doc[FIELD_NAME_DOCUMENT_MAGNITUDE]
    pass

@time_usage
def pre_calculate_document_vector_magnitudes(documents):
    for doc in documents:
        doc[FIELD_NAME_DOCUMENT_MAGNITUDE] = math.sqrt(sum([v * v for k, v in doc[FIELD_NAME_TF_IDF_MAP].items()]))
    pass


@time_usage
def calculate_tf_idf(documents, number_of_documents_terms_exists_in):
    for doc in documents:
        doc[FIELD_NAME_TF_IDF_MAP] = dict()
        for term, count in doc[FIELD_NAME_WORD_COUNT_MAP].iteritems():
            doc[FIELD_NAME_TF_IDF_MAP][term] = tf(count, doc) * idf(documents, number_of_documents_terms_exists_in,
                                                                    term)


def tf(count, doc):
    return float(count) / len(doc[FIELD_NAME_CONTENT_WITHOUT_STOPWORDS])


def idf(documents, number_of_documents_terms_exists_in, term):
    # Caching triples the performance as logarithm is expensive. Better optimizations available but is not worthwhile.
    if term not in idfCache:
        idfCache[term] = math.log(float(len(documents)) / (1 + number_of_documents_terms_exists_in[term]))
    return idfCache[term]


@time_usage
def calculate_commonality_of_terms_by_counting_number_of_documents_it_exist(documents):
    number_of_documents_terms_exists_in = dict()
    for doc in documents:
        for k, v in doc[FIELD_NAME_WORD_COUNT_MAP].iteritems():
            number_of_documents_terms_exists_in[k] = number_of_documents_terms_exists_in.get(k, 0) + 1
    return number_of_documents_terms_exists_in


@time_usage
def count_word_occurences(documents):
    for doc in documents:
        doc[FIELD_NAME_WORD_COUNT_MAP] = dict()
        for i in doc[FIELD_NAME_CONTENT_WITHOUT_STOPWORDS]:
            doc[FIELD_NAME_WORD_COUNT_MAP][i] = doc[FIELD_NAME_WORD_COUNT_MAP].get(i, 0) + 1


@time_usage
def parse_input_file():
    with open(INPUT_FILE_NAME) as json_data:
        return json.load(json_data)


@time_usage
def filter_out_entities_without_tr_field(documents):
    return [x for x in documents if 'tr' in x]


@time_usage
def remove_stop_words(documents):
    stoplist = set(unicode("de da gibi ki ve bir ben sen o bu ile hangi tez", 'utf-8').split())
    for document in documents:
        document[FIELD_NAME_CONTENT_WITHOUT_STOPWORDS] = [word for word in
                                                          document[FIELD_NAME_CONTENT_OF_INTEREST].lower().split() if
                                                          word not in stoplist]


# @time_usage
def calculate_document_distance(doc1, doc2):
    # mutual_terms = doc1[FIELD_NAME_TF_IDF_MAP].viewkeys() & doc2[FIELD_NAME_TF_IDF_MAP].viewkeys()
    smaller_doc = doc1
    bigger_doc = doc2
    if len(doc2[FIELD_NAME_TF_IDF_MAP]) < len(doc1[FIELD_NAME_TF_IDF_MAP]):
        bigger_doc = doc1
        smaller_doc = doc2
    if doc1[FIELD_NAME_DOCUMENT_MAGNITUDE] == 0 or doc2[FIELD_NAME_DOCUMENT_MAGNITUDE] == 0:
        return 1.0
    sum = 0.0
    for term, tfidf in smaller_doc[FIELD_NAME_TF_IDF_MAP].iteritems():
        sum += bigger_doc[FIELD_NAME_TF_IDF_MAP].get(term, 0) * tfidf
    return abs(1.0 - (sum))


def distance_to_closest_centoroid_squared(doc, centers):
    min_distance = min([calculate_document_distance(doc, center) for center in centers])
    return min_distance * min_distance


@time_usage
def select_initial_centers_by_kmeans_plus_plus(documents, k):
    centers = []
    for k in range(1, k + 1):
        if k == 1:
            centers.append(random.choice(documents))
        else:
            sum_of_minimum_centoroid_distances_squared = sum(
                [distance_to_closest_centoroid_squared(doc, centers) for doc in documents])
            random_float = random.uniform(0.0, sum_of_minimum_centoroid_distances_squared)
            squared_sum = 0.0
            for doc in documents:
                squared_sum += distance_to_closest_centoroid_squared(doc, centers)
                if squared_sum > random_float:
                    centers.append(doc)
                    break
    # logging.info("calculated "+ str(len(centers))+ " initial centoroids by k-means++ algorithm for k value:" + str(k))
    return centers


@time_usage
def k_means_for_range_of_k(documents, startingK, endingK):
    for i in range(startingK, endingK + 1,2):
        centers = select_initial_centers_by_kmeans_plus_plus(documents, i)
        k_means(documents, centers)
    pass


@time_usage
def find_closest_centroid_for_each_document(documents, centroids):
    for doc in documents:
        distances = [calculate_document_distance(centroid, doc) for centroid in centroids]
        doc[FIELD_NAME_CLOSEST_CENTROID] = distances.index(min(distances))
    logging.info(calculate_logging_prefix_for_better_visualization() +
                 "sum of squared errors: " + str(
        sum([distance_to_closest_centoroid_squared(doc, centroids) for doc in documents])))


def continue_iterating_k_means(oldCentroids, centroids, iterations):
    if iterations >= MAX_ITERATIONS_UNTIL_CONVERGENCE:
        return False
    if not oldCentroids:
        return True
    for i in range(0, len(oldCentroids)):
        if calculate_document_distance(oldCentroids[i], centroids[i]) > EPSILON_OF_CENTROID_EQUALITY:
            return True
    return False


@time_usage
def compute_next_centoroids(documents, k):
    centoroids = []
    for i in range(k):
        cent = dict()
        cent[FIELD_NAME_TF_IDF_MAP]= dict()
        centoroids.append(cent)
    for doc in documents:
        centoroid_id = doc[FIELD_NAME_CLOSEST_CENTROID]
        centoroid = centoroids[centoroid_id]
        number_of_elements_assigned_to_centroid = centoroid.get('numberOfDocsInCentroid', 0)
        tf_idf_vector_of_centroid = centoroid.get(FIELD_NAME_TF_IDF_MAP, {})
        for k, v in doc[FIELD_NAME_TF_IDF_MAP].iteritems():
            existing_weight = centoroid[FIELD_NAME_TF_IDF_MAP].get(k, 0.0)
            centoroid[FIELD_NAME_TF_IDF_MAP][k] = (existing_weight + v)
    for doc in centoroids:
        doc[FIELD_NAME_DOCUMENT_MAGNITUDE] = math.sqrt(sum([v * v for k, v in doc[FIELD_NAME_TF_IDF_MAP].items()]))
        for k,v in doc[FIELD_NAME_TF_IDF_MAP].items():
            doc[FIELD_NAME_TF_IDF_MAP][k] = doc[FIELD_NAME_TF_IDF_MAP][k]/doc[FIELD_NAME_DOCUMENT_MAGNITUDE]
        doc[FIELD_NAME_DOCUMENT_MAGNITUDE] =1.0

    return centoroids


@time_usage
def find_closest_centroid_for_each_document_and_calculate_squared_distance(documents, centroids):
    find_closest_centroid_for_each_document(documents, centroids)
    logging.info(calculate_logging_prefix_for_better_visualization() +
                 "sum of squared errors: " + str(
        sum([distance_to_closest_centoroid_squared(doc, centroids) for doc in documents])))
    pass


@time_usage
def k_means(documents, initial_centoroids):
    logging.info(calculate_logging_prefix_for_better_visualization() + "k means started with k value "+str(len(initial_centoroids)))
    centroids = initial_centoroids
    iterations = 0
    oldCentroids = None
    while continue_iterating_k_means(oldCentroids, centroids, iterations):
        iterations += 1
        logging.info(calculate_logging_prefix_for_better_visualization() + "iteration "+str(iterations)+ " starting")
        oldCentroids = centroids
        find_closest_centroid_for_each_document(documents, centroids)
        centroids = compute_next_centoroids(documents, len(centroids))
    find_closest_centroid_for_each_document_and_calculate_squared_distance(documents, centroids)
    return centroids
    pass


if __name__ == "__main__": main()
