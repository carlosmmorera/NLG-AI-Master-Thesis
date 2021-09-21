from credentials import *
from get_mongodocs import get_mongodocs
from output import *
from mongodb_client import *
import os

print('Imported libraries')

MAX_bytes = 508 * 1024 * 1024 #508Mb
PATH = os.getcwd() + '\\enron\\maildir'
NUM_MSGS = 517401

deleted_messages = {}
n_inserted = 0

print('Defined variables')

def insert_messages(num_enron, num_inserted):
    """
    Insert the local e-mails in the MongoDB database given by the URL

    Parameters
    ----------
    num_enron: int
        Number of Enron project.
    num_inserted: int
        Number of inserted messages.

    Returns
    -------
    bool
        True if at least an e-mail has been inserted.

    """
    inserted = False

    client, db, col = init_mongodb(ENRON[num_enron])
    inclient, _, incol = init_mongodb(INITS[num_enron], True)
    actual_size = get_database_size(db)
    users = list(os.walk(PATH))[0][1]

    saved, last_created_id = get_saved_users()
    if num_enron > last_created_id:
        enron_n = set()
    else:
        enron_n = get_users_of_enron_n(last_created_id)

    directories = [{'Path': os.path.join(PATH, user),
                    'Size': get_directory_size(os.path.join(PATH, user))}
                   for user in users
                   if os.path.join(PATH, user) not in saved]
    directories.sort(key=lambda x: x['Size'], reverse=True)

    for user in directories:
        if actual_size + user['Size'] < MAX_bytes:
            inserted = True

            with open("tmpdirs.txt", 'w') as f:
                f.write("")
            for subdir, _, files in os.walk(user['Path']):
                with open("tmpdirs.txt", 'a') as f:
                    f.write(str(subdir) + '\n')
                for file in files:
                    msg, inits = get_mongodocs(os.path.join(subdir, file))
                    if msg is None:
                        if inits not in deleted_messages:
                            deleted_messages[inits] = 0
                        deleted_messages[inits] += 1
                    else:
                        col.insert_one(msg)
                        for init in inits:
                            incol.insert_one({'Message-ID': msg['Message-ID'],
                                              'Subject': init[0], 'Verb': init[1],
                                              'Object': init[2]})
                    num_inserted += 1
                    printProgressBar(num_inserted, NUM_MSGS, deleted_messages, prefix='Progress:', suffix='Complete',
                                     length=50)

            actual_size += user['Size']
            enron_n.add(user['Path'])
            with open(f'enron{max(last_created_id, num_enron)}.set', 'w') as f:
                f.write(str(enron_n))

    client.close()
    inclient.close()

    return inserted, num_inserted

print('Starting process...')
printProgressBar(n_inserted, NUM_MSGS, deleted_messages, prefix='Progress:', suffix='Complete', length=50)
for i in range(len(ENRON)):
    cont, n_inserted = insert_messages(i, n_inserted)
    while cont:
        cont, n_inserted = insert_messages(i, n_inserted)

myprint(str(deleted_messages))
print('Process finished')