import pandas as pd
import numpy as np
import numpy as np
from skip_thoughts import configuration
from skip_thoughts import encoder_manager

train_size = 2000
test_size = 500

def load_data():
    print('Loading data')
    train_stories = pd.read_csv("story_generation_dataset/ROCStories_train.csv", encoding="utf8")
    test_stories = pd.read_csv("story_generation_dataset/ROCStories_test.csv", encoding="utf8")
    val_stories = pd.read_csv("story_generation_dataset/ROCStories_val.csv", encoding="utf8")
    train_stories = train_stories.append(val_stories)
    train_stories = train_stories[:train_size]
    test_stories = train_stories[:test_size]
    train_array = train_stories.values[:,1:].reshape(-1).tolist()
    test_array = test_stories.values[:,1:].reshape(-1).tolist()
    return train_array, test_array

def get_bidirectional_encoder():
    print('Getting bidirectional encoder')
    # bidirectional skip thoughts
    VOCAB_FILE = "skip_thoughts/pretrained/skip_thoughts_bi_2017_02_16/vocab.txt"
    EMBEDDING_MATRIX_FILE = "skip_thoughts/pretrained/skip_thoughts_bi_2017_02_16/embeddings.npy"
    CHECKPOINT_PATH = "skip_thoughts/pretrained/skip_thoughts_bi_2017_02_16/model.ckpt-500008"
    encoder = encoder_manager.EncoderManager()
    encoder.load_model(configuration.model_config(bidirectional_encoder=True),
                    vocabulary_file=VOCAB_FILE,
                    embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                    checkpoint_path=CHECKPOINT_PATH)
    return encoder

def generate_vectors(encoder, train_array, test_array):
    print('Generating train vectors')
    vectors_train = encoder.encode(train_array)
    np.save("data/train_vectors", vectors_train)
    np.save("data/train_sentences", train_array)
    print('Generating test vectors')
    vectors_test = encoder.encode(test_array)
    np.save("data/test_vectors", vectors_test)
    np.save("data/test_sentences", test_array)

def get_nearest(qvec, vectors, array, k=5):
    scores = np.dot(qvec, vectors.T).flatten()
    sorted_args = np.argsort(scores)[::-1]
    sentences = [array[a] for a in sorted_args[:k]]
    print ('DATA: ',end="")
    for i, s in enumerate(sentences):
        print (s,sorted_args[i])

def output_data_and_label():
    test_vectors = np.load("data/test_vectors.npy")
    test_sentences = np.load("data/test_sentences.npy")
    for i,j in enumerate(test_vectors):
        get_nearest(test_vectors[i], test_vectors, test_sentences, k=1)
        print("LABEL:", test_sentences[i+1], i+1, end="\n\n")

if __name__ == '__main__':
    train_array, test_array = load_data()
    encoder = get_bidirectional_encoder()
    generate_vectors(encoder, train_array, test_array)
    output_data_and_label()

