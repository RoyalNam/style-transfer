class Config:
    SOURCE_URL = 'https://drive.google.com/file/d/1tzLRhLautKQv32pF4sQ4-CCbZHqhnYj9/view?usp=sharing'
    ROOT_DIR = 'data'
    OUTPUT_PATH = "data/dataset.zip",
    DIR_A = 'data/trainA'
    DIR_B = 'data/trainB'
    CHECKPOINT_GEN_PATH = ''
    CHECKPOINT_DISC_A_PATH = ''
    CHECKPOINT_DISC_B_PATH = ''
    GEN_SAVED_PATH = 'saved_models/gen.pth'
    DISC_A_SAVED_PATH = 'saved_models/disc_A.pth'
    DISC_B_SAVED_PATH = 'saved_models/disc_B.pth'

    N_EPOCHS = 5
    DIM_A = 3
    DIM_B = 3
    BATCH_SIZE = 1
    LR = 2e-4
    IMG_SIZE = 256
    PRETRAINED = False
    BETA1 = 0.5
    BETA2 = 0.999
    LAMBDA_IDENTITY = 0.1
    LAMBDA_CYCLE = 10
