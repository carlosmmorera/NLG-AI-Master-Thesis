# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 00:53:48 2021

@author: Carlos Moreno Morera
"""
from pymongo import MongoClient
import os, email
import ast

NUM_PROJECTS = 5


def get_database_size(db):
    """
    Obtains the size of the MongoDB database in bytes.

    Parameters
    ----------
    db : pymongo.database
        MongoDB Database.

    Returns
    -------
    int
        Size of the database in bytes.

    """
    call = db.command("dbstats")
    return call['dataSize']


def get_directory_size(path):
    """
    If the given path is a file, it returns its size. If is a directory, obtains
    the sum of the sizes of all the files it and its subdirectories contain.

    Parameters
    ----------
    path : str
        Path of the file or directory that we want to know its size.

    Returns
    -------
    int
        Size in bytes.

    """
    if os.path.isfile(path):
        return os.path.getsize(path)
    else:
        return sum([get_directory_size(os.path.join(path, f))
                    for f in os.listdir(path)])


def get_users_of_enron_n(n):
    """
    Obtains the stored users in Enron database with the given id.

    Parameters
    ----------
    n : int
        Id of the Enron database.

    Returns
    -------
    saved : set
        Users stored.

    """
    if os.path.isfile(f'enron{n}.set'):
        with open(f'enron{n}.set', 'r') as f:
            saved = ast.literal_eval(f.read())
    else:
        saved = set()

    return saved


def get_saved_users():
    """
    Obtains all the users previously saved.

    Returns
    -------
    saved : set
        Users stored in an Enron database.
    big_ind : int
        Identifier of the last Enron database created.

    """
    saved = set()
    big_ind = 0
    for i in range(NUM_PROJECTS):
        if os.path.isfile(f'enron{i}.set'):
            big_ind = i
            with open(f'enron{i}.set', 'r') as f:
                saved = saved.union(ast.literal_eval(f.read()))

    return saved, big_ind


def init_mongodb(url, init = False):
    """
    Initialize a MongoDB connection

    Parameters
    ----------
    url : str
        URL.

    Returns
    -------
    client : pymongo.mongo_client
        Client connection.
    db : pymongo.database
        MongoDB database.
    col : pymongo.collection
        Main collection of the database.

    """
    client = MongoClient(url, ssl=True, ssl_cert_reqs='CERT_NONE')
    db = client.EnronDB
    if not(init):
        col = db.emails
    else:
        col = db.init
    return client, db, col