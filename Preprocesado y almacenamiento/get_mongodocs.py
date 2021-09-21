import email
import spacy
import textacy
import re

REPLY_PAT = '-----Original Message-----'
FORWARD_PAT = '---------------------- Forwarded by'
TOKEN_SEPARATOR = '¿'
MAXIMUM_LEN = 1000000

NLP = spacy.load("en_core_web_lg")


def get_msg_body(msg):
    """
    Extracts the body of the given e-mail.

    Parameters
    ----------
    msg : email.message
        E-mail.

    Returns
    -------
    str
        Body of the e-mail.

    """
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append(part.get_payload())

    return ''.join(parts)


def clean_body_message(body):
    return re.sub(' +', ' ', body.split(REPLY_PAT)[0].split(FORWARD_PAT)[0].replace('\n', ' '))


def create_msg_from_string(string):
    """
    Given the raw e-mail in string format, obtains the following information
    about the message:
        - Message-ID: unique identifier of the e-mail.
        - From
        - To
        - Subject
        - Body

    Parameters
    ----------
    string : str
        Raw e-mail.

    Returns
    -------
    dict
        Dictionary with all the mentioned fields.

    """
    msg = email.message_from_string(string)
    body = clean_body_message(get_msg_body(msg))
    return {'Message-ID': msg.get('Message-ID'), 'From': msg.get('From'),
            'To': msg.get('To'), 'Subject': msg.get('Subject'), 'Body': body}


def get_numwords(doc):
    num_words = 0
    for token in doc:
        if not (token.is_punct or token.is_space):
            num_words += 1
    return num_words


def get_texts_svo(init):
    return (TOKEN_SEPARATOR.join([token.text for token in init.subject]),
     TOKEN_SEPARATOR.join([token.text for token in init.verb]),
     TOKEN_SEPARATOR.join([token.text for token in init.object]))


def get_inits(doc):
    inits = textacy.extract.subject_verb_object_triples(doc)
    if inits:
        return [get_texts_svo(init) for init in list(inits)]
    else:
        return []


def get_mongodocs(path):
    with open(path) as f:
        msg = create_msg_from_string(f.read())

    if len(msg['Body']) > MAXIMUM_LEN:
        return None, 'over-length'

    doc = NLP(msg['Body'])
    n = get_numwords(doc)

    if n == 0:
        return None, 'no-words'

    msg['Number_of_Words'] = n
    inits = get_inits(doc)

    if len(inits) == 0:
        return None, 'no-inits'

    msg['Number_of_Inits'] = len(inits)
    return msg, inits
