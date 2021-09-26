from credentials import *
from get_mongodocs import NLP, get_numwords, get_inits
from output import *
from mongodb_client import *
import datetime

print('Imported libraries')

SIGNATURE_PAT = '______________________________________________________________________________'


def get_documents(col, filter, doc_to_tuple=lambda doc: doc):
    num_docs = col.count_documents(filter)
    cursor = col.find(filter)
    print('Copying cursor...')
    collection_data = []
    i = 0
    printProgressBar(i, num_docs, {}, prefix='Progress:', suffix='Complete', length=50, printEnd='')
    for doc in cursor:
        collection_data.append(doc_to_tuple(doc))
        i += 1
        if i % 100 == 0:
            printProgressBar(i, num_docs, {}, prefix='Progress:', suffix='Complete', length=50, printEnd='')
    printProgressBar(i, num_docs, {}, prefix='Progress:', suffix='Complete', length=50)
    print('Cursor copied')
    return collection_data


def update_documents(num_enron, deleted_messages, updated=0):
    client, db, col = init_mongodb(ENRON[num_enron])
    inclient, _, incol = init_mongodb(INITS[num_enron], True)

    enr2, _, col2 = init_mongodb(ENRON2)
    inclient2, _, incol2 = init_mongodb(INIT2, True)

    print(f'\nWorking with {ENRON[num_enron].split("@")[1]}')
    print('----------------------------------------------')
    print(f'Starting process at {datetime.datetime.now()}')
    collection_data = get_documents(col, {}, lambda doc: (doc['Message-ID'], doc['Body']))
    num_collection = len(collection_data)
    reviewed = 0
    printProgressBar(reviewed, num_collection, deleted_messages, prefix='Progress:', suffix='Complete', length=50)
    for msg_id, body in collection_data:
        if SIGNATURE_PAT in body:
            l = [doc for doc in col.find({"Message-ID": msg_id})]
            new_msg = {'Message-ID': msg_id, 'From': l[0]['From'], 'To': l[0]['To'], 'Subject': l[0]['Subject'],
                       'Body': body.split(SIGNATURE_PAT)[0]}
            doc = NLP(new_msg['Body'])
            n = get_numwords(doc)

            if n == 0:
                if 'no-words' not in deleted_messages:
                    deleted_messages['no-words'] = 0
                deleted_messages['no-words'] += 1
            else:
                new_msg['Number_of_Words'] = n
                inits = get_inits(doc)
                if len(inits) == 0:
                    if 'no-inits' not in deleted_messages:
                        deleted_messages['no-inits'] = 0
                    deleted_messages['no-inits'] += 1
                else:
                    new_msg['Number_of_Inits'] = len(inits)
                    col2.insert_one(new_msg)
                    for init in inits:
                        incol2.insert_one({'Message-ID': new_msg['Message-ID'],
                                          'Subject': init[0], 'Verb': init[1],
                                          'Object': init[2]})
                    updated += 1
            col.delete_many({"_id": l[0]['_id']})
            incol.delete_many({"Message-ID": msg_id})
        reviewed += 1
        printProgressBar(reviewed, num_collection, deleted_messages, prefix='Progress:', suffix='Complete', length=50)

    client.close()
    inclient.close()
    enr2.close()
    inclient2.close()
    return deleted_messages, updated


deleted_messages = {}
updated = 0
for i in range(len(ENRON)):
    deleted_messages, updated = update_documents(i, deleted_messages, updated)

print(f'Updated documents: {updated}')
myprint(str(deleted_messages), 'update.log')
myprint(f'Updated documents: {updated}', 'update.log')
print('Process finished')
