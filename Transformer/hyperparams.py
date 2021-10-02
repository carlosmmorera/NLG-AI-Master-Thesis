# Transformer hyperparameters
num_layers = 6
d_model = 512
dff = 2048
num_heads = 8
dropout_rate = 0.1
EPOCHS = 50

# Dataset hyperparameters
MAX_INITS = 6
MAX_WORDS = [103.5, 158.5, 205.5, 247.5, 283.0, 336.0]
REMOVED_MSGS = {'<13628103.1075854036066.JavaMail.evans@thyme>', '<7717017.1075854041870.JavaMail.evans@thyme>',
                '<10927207.1075854047255.JavaMail.evans@thyme>',
                '<30229870.1075854441633.JavaMail.evans@thyme>', '<15495868.1075861473989.JavaMail.evans@thyme>',
                '<18112782.1075854449243.JavaMail.evans@thyme>',
                '<12421080.1075851639178.JavaMail.evans@thyme>', '<1429151.1075840948969.JavaMail.evans@thyme>',
                '<26014676.1075854474765.JavaMail.evans@thyme>',
                '<29393146.1075854491245.JavaMail.evans@thyme>'}

# Tokenizer hyperparameters
MAX_LENGTH_SVO = (39, 5, 91)
MAX_TOK_SUBJECT = 100
MAX_TOK_VERB = 64
MAX_TOK_OBJECT = 129
MAX_TOK_MSG = 2172
