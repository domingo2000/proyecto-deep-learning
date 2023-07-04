#########################
## Training Parameters ##
#########################
# DATSET_PATH = ["SCAN", "simple_split", "tasks_train_simple.txt"]
DATSET_PATH = ["data", "tasks_toy.txt"]
LEARNING_RATE = 1e-5
EPOCHS = 120
EPOCH_N_METRICS = 10
BATCH_LOGGING_N = 100
GRADIENT_CLIPPING = 0.5

######################
## Model Parameters ##
######################

D_MODEL = 32  # 512  # Original from Attention is All You need paper describe the dimension of the embeddings (input, output, positional)
D_FEED_FORWARD = 256
N_HEADS = 4
N_ENCODER_LAYERS = 2
N_DECODER_LAYERS = 2
DROPOUT = 0.0
# MAX sequence of commands OR actions are 479 so 512 is enough for the task
CONTEXT_LENGTH = 512
MAX_LENGTH = 512


###################################
## Dataset Vocabulary parameters ##
###################################
INPUT_VOCABULARY = [
    "and",  # 0
    "after",  # 1
    "twice",  # 2
    "thrice",  # 3
    "opposite",  # 4
    "around",  # 5
    "left",  # 6
    "right",  # 7
    "turn",  # 8
    "walk",  # 9
    "look",  # 10
    "run",  # 11
    "jump",  # 12
    "<SOS>",  # 13
    "<EOS>",  # 14
    "<PAD>",  # 15
]

OUTPUT_VOCABULARY = [
    "I_WALK",  # 0
    "I_LOOK",  # 1
    "I_RUN",  # 2
    "I_JUMP",  # 3
    "I_TURN_LEFT",  # 4
    "I_TURN_RIGHT",  # 5
    "<SOS>",  # 6
    "<EOS>",  # 7
    "<PAD>",  # 8
]

INPUT_VOCAB_SIZE = len(INPUT_VOCABULARY)
OUTPUT_VOCAB_SIZE = len(OUTPUT_VOCABULARY)
