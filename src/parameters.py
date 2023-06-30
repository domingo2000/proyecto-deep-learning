#########################
## Training Parameters ##
#########################

LEARNING_RATE = 1e-3
EPOCHS = 200_000
BATCH_SIZE = 32

######################
## Model Parameters ##
######################

D_MODEL = 32  # 512  # Original from Attention is All You need paper describe the dimension of the embeddings (input, output, positional)
N_HEADS = 2
N_ENCODER_LAYERS = 2
N_DECODER_LAYERS = 2
# MAX sequence of commands OR actions are 479 so 512 is enough for the task
CONTEXT_LENGTH = 10


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
