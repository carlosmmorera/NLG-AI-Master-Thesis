from credentials import ENRON, INITS
from mongodb_client import init_mongodb
from output import printProgressBar
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as text
from hyperparams import *
from collections import deque
import pickle


def get_documents(col, myfilter, doc_to_tuple=lambda doc: doc, output=True):
    num_docs = None
    if output:
        num_docs = col.count_documents(myfilter)
    cursor = col.find(myfilter)
    if output:
        print('Start copying cursor')
    collection_data = deque()
    i = 0
    if output:
        printProgressBar(i, num_docs, {}, prefix='Progress:', suffix='Complete', length=50, printEnd='')
    for doc in cursor:
        collection_data.append(doc_to_tuple(doc))
        i += 1
        if output and i % 100 == 0:
            printProgressBar(i, num_docs, {}, prefix='Progress:', suffix='Complete', length=50, printEnd='')

    if output:
        if i%100 != 0:
            printProgressBar(i, num_docs, {}, prefix='Progress:', suffix='Complete', length=50, printEnd='')
        print('Cursor copied')
    return collection_data, num_docs


def generate_dataset_row(body, inits_data, num_inits):
    row = []
    for j in range(MAX_INITS):
        s = []
        v = []
        o = []
        if j < num_inits:
            s, v, o = inits_data.pop()
            s = s.split('¿')
            v = v.split('¿')
            o = o.split('¿')
        for part, max_length in zip([s, v, o], MAX_LENGTH_SVO):
            for k in range(max_length):
                if k < len(part):
                    row.append(part[k])
                else:
                    row.append('')
    row.append(body)
    return row


def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, shuffle=True, shuffle_size=10000):
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=7)

    train_size = int(train_split * ds_size)

    train_ds = ds.take(train_size)
    test_ds = ds.skip(train_size)

    return train_ds, test_ds


def create_datalist():
    data_list = []
    size_dataset = 0

    for enron_cluster, inits_cluster in zip(ENRON, INITS):
        print(f'\nWorking with {enron_cluster.split("@")[1]}')
        print('----------------------------------------------')
        enron_client, _, enron_col = init_mongodb(enron_cluster)
        init_client, _, init_col = init_mongodb(inits_cluster, init=True)

        enron_filter = {
            "$and": [{"Number_of_Inits": {"$lte": MAX_INITS}}, {"Number_of_Words": {"$lte": max(MAX_WORDS)}}]}
        enron_data, num_enron = get_documents(enron_col, enron_filter,
                                              lambda doc: (doc['Message-ID'], doc['Body'],
                                                           doc['Number_of_Inits'], doc['Number_of_Words']))
        with open("./dataset_checkpoint/enron.pkl", "wb") as fp:
            pickle.dump(enron_data, fp)
        with open("./dataset_checkpoint/datalist.pkl", "wb") as fp:
            pickle.dump(data_list, fp)

        print('Starting process...')
        printProgressBar(0, num_enron, {}, prefix='Progress:', suffix='Complete', length=50, printEnd='')
        for i in range(num_enron):
            msg_id, body, num_inits, num_words = enron_data.pop()

            if num_words <= MAX_WORDS[num_inits - 1] and msg_id not in REMOVED_MSGS:
                inits_data, _ = get_documents(init_col, {"Message-ID": msg_id},
                                              lambda doc: (doc['Subject'], doc['Verb'], doc['Object']),
                                              False)
                if len(inits_data) == num_inits:
                    data_list.append(generate_dataset_row(body, inits_data, num_inits))
                    size_dataset += 1

            if i % 10000 == 0:
                with open("./dataset_checkpoint/enron.pkl", "wb") as fp:
                    pickle.dump(enron_data, fp)
                with open("./dataset_checkpoint/datalist.pkl", "wb") as fp:
                    pickle.dump(data_list, fp)
            if i % 50 == 0:
                printProgressBar(i, num_enron, {}, prefix='Progress:', suffix='Complete', length=50, printEnd='')
        if i%50 != 0:
            printProgressBar(num_enron, num_enron, {}, prefix='Progress:', suffix='Complete', length=50, printEnd='')
        enron_client.close()
        init_client.close()

    with open("./dataset_checkpoint/datalist.pkl", "wb") as fp:
        pickle.dump(data_list, fp)
    print(f'\n\nDataList created with {size_dataset} rows.')

    return data_list, size_dataset


def tokenize_svo_and_body(tokenizers, svo_and_body):
    t = tokenizers.en.tokenize(svo_and_body).to_tensor()
    return (tf.reshape(t[:-1], [-1]), t[-1])


def make_batches(ds, buffer_size, batch_size, tokenizers):
    return (
      ds
      .cache()
      .shuffle(buffer_size)
      .map(lambda x: tokenize_svo_and_body(tokenizers, x), num_parallel_calls=tf.data.AUTOTUNE)
      .batch(batch_size)
      .prefetch(tf.data.AUTOTUNE))


def create_dataset_from_datalist():
    dl, size_ds = create_datalist()
    ds = tf.data.Dataset.from_tensor_slices(dl)
    tf.data.experimental.save(ds, './dataset_from_dl/')
    print("Created Dataset")
    return ds, size_ds

def create_dataset(ds = None, size_ds = 0, saved_ds = False):
    model_name = "ted_hrlr_translate_pt_en_converter"
    tf.keras.utils.get_file(
        f"{model_name}.zip",
        f"http://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
        cache_dir='.', cache_subdir='', extract=True
    )
    tokenizers = tf.saved_model.load(model_name)

    if saved_ds and ds is None:
        ds = tf.data.experimental.load('./dataset_from_dl/')
        size_ds = 231222
    if ds is None:
        ds, size_ds = create_dataset_from_datalist()

    train_ds, test_ds = get_dataset_partitions_tf(ds, size_ds, 0.9, True, size_ds + 1)
    print("Dataset divided into train and test")

    train_batches = make_batches(train_ds, size_ds, 256, tokenizers)
    test_batches = make_batches(test_ds, size_ds, 256, tokenizers)
    tf.data.experimental.save(train_batches, './train_batches/')
    tf.data.experimental.save(test_batches, './test_batches/')
    print("Created train and test batches")
    return train_batches, test_batches, tokenizers