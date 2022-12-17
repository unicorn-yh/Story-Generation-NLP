import numpy as np
import pandas as pd
from keras.models import Sequential 
from keras.layers import LSTM 
from random import randint

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
    train_array = train_stories.values[:12000,1:].reshape(-1).tolist()
    test_array = test_stories.values[:,1:].reshape(-1).tolist()
    return train_array, test_array

def get_nearest(qvec, vectors, array, k=5):
    sentences = []
    qvec /= np.linalg.norm(qvec)
    vectors /= np.linalg.norm(vectors)
    scores = np.dot(qvec, vectors.T).flatten()
    sorted_args = np.argsort(scores)[::-1]
    for i in range(k):
        sentences.append(array[sorted_args[i]])
    for i, s in enumerate(sentences):
        print (sorted_args[i],'\t',s)

def define_X_and_Y():
    vecfile = np.load("data/train_vectors.npy")
    vector = vecfile.tolist()
    _,v1,v2,v3,v4,v5 = vector[::6],vector[1::6],vector[2::6],vector[3::6],vector[4::6],vector[5::6]
    v0 = np.zeros(2400).tolist()
    X,Y = [],[]
    for i in range(len(v1)):
        X.append([v0 , v0 , v0 , v1[i]])
        Y.append(v2[i])
        X.append([v0, v0, v1[i], v2[i]])
        Y.append(v3[i])
        X.append([v0, v1[i], v2[i], v3[i]])
        Y.append(v4[i])
        X.append([v1[i], v2[i], v3[i], v4[i]])
        Y.append(v5[i])
    X = np.asarray(X)   # X.shape = (samples, timesteps, features)
    Y = np.asarray(Y)
    return X, Y, vecfile

def lstm_model(X,Y):
    model = Sequential() 
    model.add(LSTM(2400, dropout=0.2, recurrent_dropout=0.2, input_shape=(4,2400))) 
    model.compile(loss='mean_squared_error', optimizer='rmsprop') 
    model.fit(X, Y, batch_size=16, epochs=10)
    return model

def define_test_index():
    test_index = []
    for i in range(10):
        test_index.append(randint(0,12000))
    np.savetxt("test-index.txt",test_index,fmt="%s")

def generate_train_story(model,X,Y,vecfile,train_array):
    test_ls = np.loadtxt("test-index.txt")
    for i in range(20):
        j = int(test_ls[i])
        print("Original Story:")
        get_nearest(X[j:j+1,3,:].squeeze().tolist(), vecfile, train_array, k=1)
        print("Actual Story:")
        get_nearest(Y[j:j+1,:].squeeze().tolist(), vecfile, train_array, k=1)
        print("Predicted Story:")
        pred = model.predict(X[j:j+1,:,:])
        get_nearest(pred.squeeze().tolist(), vecfile, train_array)
        print("Similarity:",np.dot(pred.squeeze(), Y[j:j+1,:].squeeze().T),end="\n\n")
        #Should be close to 1, that means they are same (1 when normalized)

def generate_test_story(model,X,Y,vecfile,test_array):
    for j in range(10):
        print("Original Story:")
        get_nearest(X[j:j+1,3,:].squeeze().tolist(), vecfile, train_array, k=1)
        print("Predicted Story:")
        pred = model.predict(X[j:j+1,:,:])
        get_nearest(pred.squeeze().tolist(), vecfile, test_array)
        print("Similarity:",np.dot(pred.squeeze(), Y[j:j+1,:].squeeze().T))
        #Should be close to 1, that means they are same (1 when normalized)

if __name__ == "__main__":
    train_array, test_array = load_data()
    X, Y, vecfile = define_X_and_Y()
    model = lstm_model()
    define_test_index()
    generate_train_story(model, X, Y, vecfile, train_array)
    generate_test_story(model, X, Y, vecfile, test_array)



