import glob
import numpy as np
import re
import sys
import codecs
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    atheism_examples_train = []
    for doc in glob.glob('./datasets/20newsgroup/train/alt.atheism/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            atheism_examples_train.append(f.read())
    atheism_examples_test = []
    for doc in glob.glob('./datasets/20newsgroup/test/alt.atheism/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            atheism_examples_test.append(f.read())
    #print("length of 1st",len(atheism_examples_train))
    # second class
    graphics_examples_train = []
    for doc in glob.glob('./datasets/20newsgroup/train/comp.graphics/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            graphics_examples_train.append(f.read())
    graphics_examples_test = []
    for doc in glob.glob('./datasets/20newsgroup/test/comp.graphics/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            graphics_examples_test.append(f.read())
    #print("length of 2",len(graphics_examples_train))
    # 3 class
    windows_examples_train = []
    for doc in glob.glob('./datasets/20newsgroup/train/comp.os.ms-windows.misc/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            windows_examples_train.append(f.read())
    
    windows_examples_test = []
    for doc in glob.glob('./datasets/20newsgroup/test/comp.os.ms-windows.misc/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            windows_examples_test.append(f.read())
    #print("length of 3",len(windows_examples_train))
    # 4 class 
    hardware_examples_train = []
    for doc in glob.glob('./datasets/20newsgroup/train/comp.sys.ibm.pc.hardware/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            hardware_examples_train.append(f.read())
    hardware_examples_test = []
    for doc in glob.glob('./datasets/20newsgroup/test/comp.sys.ibm.pc.hardware/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            hardware_examples_test.append(f.read())
    #print("length of 4",len(hardware_examples_train))
    # 5 class 
    mac_hardware_examples_train = []
    for doc in glob.glob('./datasets/20newsgroup/train/comp.sys.mac.hardware/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            mac_hardware_examples_train.append(f.read())
    mac_hardware_examples_test = []
    for doc in glob.glob('./datasets/20newsgroup/test/comp.sys.mac.hardware/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            mac_hardware_examples_test.append(f.read())
    #print("length of 5",len(mac_hardware_examples_train))
    # 6 class 
    windows_x_examples_train = []
    for doc in glob.glob('./datasets/20newsgroup/train/comp.windows.x/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            windows_x_examples_train.append(f.read())
    windows_x_examples_test = []
    for doc in glob.glob('./datasets/20newsgroup/test/comp.windows.x/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            windows_x_examples_test.append(f.read())
    #print("length of 6",len(windows_x_examples_train))
    # 7 class 
    forsale_examples_train = []
    for doc in glob.glob('./datasets/20newsgroup/train/misc.forsale/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            forsale_examples_train.append(f.read())
    forsale_examples_test = []
    for doc in glob.glob('./datasets/20newsgroup/test/misc.forsale/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            forsale_examples_test.append(f.read())
    #print("length of 7",len(forsale_examples_train))
    # 8 class 
    autos_examples_train = []
    for doc in glob.glob('./datasets/20newsgroup/train/rec.autos/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            autos_examples_train.append(f.read())
    autos_examples_test = []
    for doc in glob.glob('./datasets/20newsgroup/test/rec.autos/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            autos_examples_test.append(f.read())
    #print("length of 8",len(autos_examples_train))
    # 9 class 
    motorcycles_examples_train = []
    for doc in glob.glob('./datasets/20newsgroup/train/rec.motorcycles/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            motorcycles_examples_train.append(f.read())
    motorcycles_examples_test = []
    for doc in glob.glob('./datasets/20newsgroup/test/rec.motorcycles/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            motorcycles_examples_test.append(f.read())
    #print("length of 9",len(motorcycles_examples_train))
    # 10 class 
    baseball_examples_train = []
    for doc in glob.glob('./datasets/20newsgroup/train/rec.sport.baseball/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            baseball_examples_train.append(f.read())
    baseball_examples_test = []
    for doc in glob.glob('./datasets/20newsgroup/test/rec.sport.baseball/*'):
        with codecs.open(doc,"r",encoding='utf-8', errors='ignore') as f:
            baseball_examples_test.append(f.read())
    #print("length of 10",len(baseball_examples_train))
    # 11 class
    hockey_examples_train = []
    for doc in glob.glob('./datasets/20newsgroup/train/rec.sport.hockey/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            hockey_examples_train.append(f.read())
    hockey_examples_test = []
    for doc in glob.glob('./datasets/20newsgroup/test/rec.sport.hockey/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            hockey_examples_test.append(f.read())
    #print("length of 11",len(hockey_examples_train))
    # 12 class
    crypt_examples_train = []
    for doc in glob.glob('./datasets/20newsgroup/train/sci.crypt/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            crypt_examples_train.append(f.read())
    crypt_examples_test = []
    for doc in glob.glob('./datasets/20newsgroup/test/sci.crypt/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            crypt_examples_test.append(f.read())
    #print("length of 12",len(crypt_examples_train))
    # 13 class
    electronics_examples_train = []
    for doc in glob.glob('./datasets/20newsgroup/train/sci.electronics/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            electronics_examples_train.append(f.read())
    
    electronics_examples_test = []
    for doc in glob.glob('./datasets/20newsgroup/test/sci.electronics/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            electronics_examples_test.append(f.read())
    #print("length of 13",len(electronics_examples_train))
    # 14 class 
    med_examples_train = []
    for doc in glob.glob('./datasets/20newsgroup/train/sci.med/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            med_examples_train.append(f.read())
    med_examples_test = []
    for doc in glob.glob('./datasets/20newsgroup/test/sci.med/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            med_examples_test.append(f.read())
    #print("length of 14",len(med_examples_train))
    # 15 class 
    space_examples_train = []
    for doc in glob.glob('./datasets/20newsgroup/train/sci.space/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            space_examples_train.append(f.read())
    space_examples_test = []
    for doc in glob.glob('./datasets/20newsgroup/test/sci.space/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            space_examples_test.append(f.read())
    #print("length of 15",len(space_examples_train))
    # 16 class 
    christian_examples_train = []
    for doc in glob.glob('./datasets/20newsgroup/train/soc.religion.christian/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            christian_examples_train.append(f.read())
    christian_examples_test = []
    for doc in glob.glob('./datasets/20newsgroup/test/soc.religion.christian/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            christian_examples_test.append(f.read())
    #print("length of 16",len(christian_examples_train))
    # 17 class 
    guns_examples_train = []
    for doc in glob.glob('./datasets/20newsgroup/train/talk.politics.guns/*'):
        with codecs.open(doc,"r",encoding='utf-8', errors='ignore') as f:
            guns_examples_train.append(f.read())
    guns_examples_test = []
    for doc in glob.glob('./datasets/20newsgroup/test/talk.politics.guns/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            guns_examples_test.append(f.read())
    #print("length of 17",len(guns_examples_train))
    # 18 class 
    mideast_examples_train = []
    for doc in glob.glob('./datasets/20newsgroup/train/talk.politics.mideast/*'):
        with codecs.open(doc,"r",encoding='utf-8', errors='ignore') as f:
            mideast_examples_train.append(f.read())
    mideast_examples_test = []
    for doc in glob.glob('./datasets/20newsgroup/test/talk.politics.mideast/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            mideast_examples_test.append(f.read())
    #print("length of 18",len(mideast_examples_train))
    # 19 class 
    politics_misc_examples_train = []
    for doc in glob.glob('./datasets/20newsgroup/train/talk.politics.misc/*'):
        with codecs.open(doc,"r",encoding='utf-8', errors='ignore') as f:
            politics_misc_examples_train.append(f.read())
    politics_misc_examples_test = []
    for doc in glob.glob('./datasets/20newsgroup/test/talk.politics.misc/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            politics_misc_examples_test.append(f.read())
    #print("length of 19",len(politics_misc_examples_train))
    # 20 class 
    religion_misc_examples_train = []
    for doc in glob.glob('./datasets/20newsgroup/train/talk.religion.misc/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            religion_misc_examples_train.append(f.read())
    religion_misc_examples_test = []
    for doc in glob.glob('./datasets/20newsgroup/test/talk.religion.misc/*'):
        with codecs.open(doc, "r",encoding='utf-8', errors='ignore') as f:
            religion_misc_examples_test.append(f.read())
    #print("length of 20",len(religion_misc_examples_train))
    # Split by words
    x_train = atheism_examples_train + graphics_examples_train + windows_examples_train + hardware_examples_train + mac_hardware_examples_train + windows_x_examples_train + forsale_examples_train + autos_examples_train + motorcycles_examples_train + baseball_examples_train + hockey_examples_train + crypt_examples_train + electronics_examples_train + med_examples_train + space_examples_train + christian_examples_train + guns_examples_train + mideast_examples_train + politics_misc_examples_train + religion_misc_examples_train
    #print("train examples.........-",len(x_train))

    x_train = [clean_str(sent) for sent in x_train]
    #print("train examples-",len(x_train))
    #### test example
    x_test = atheism_examples_test + graphics_examples_test + windows_examples_test + hardware_examples_test + mac_hardware_examples_test + windows_x_examples_test + forsale_examples_test + autos_examples_test + motorcycles_examples_test + baseball_examples_test + hockey_examples_test + crypt_examples_test + electronics_examples_test + med_examples_test + space_examples_test + christian_examples_test +     guns_examples_test + mideast_examples_test + politics_misc_examples_test + religion_misc_examples_test
    #print("test examples-",len(x_test))
    x_test = [clean_str(sent) for sent in x_test]
    #print(len(x_train))
    #print(len(x_test))
    # Generate labels for training examples
    atheism_labels_train =  [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in atheism_examples_train]
    graphics_labels_train = [[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in graphics_examples_train]
    windows_labels_train =  [[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in windows_examples_train]
    hardware_labels_train=  [[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]for _ in hardware_examples_train]
    mac_labels_train=  [[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]for _ in mac_hardware_examples_train]
    windowx_labels_train =  [[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in windows_x_examples_train]
    forsale_labels_train = [[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in forsale_examples_train]
    autos_labels_train =  [[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0] for _ in autos_examples_train]
    motor_labels_train=  [[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]for _ in motorcycles_examples_train]
    baseball_labels_train=  [[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]for _ in baseball_examples_train ]
    hockey_labels_train =  [[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0] for _ in hockey_examples_train]
    crypt_labels_train = [[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0] for _ in crypt_examples_train ]
    electronics_labels_train =  [[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0] for _ in electronics_examples_train]
    med_labels_train=  [[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]for _ in med_examples_train]
    space_labels_train=  [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]for _ in space_examples_train]
    christ_labels_train =  [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0] for _ in christian_examples_train]
    guns_labels_train = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0] for _ in guns_examples_train]
    mideast_labels_train =  [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0] for _ in  mideast_examples_train]
    poli_labels_train=  [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]for _ in politics_misc_examples_train]
    rel_labels_train=  [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]for _ in religion_misc_examples_train]

    

    y_train = np.concatenate([atheism_labels_train, graphics_labels_train,windows_labels_train,hardware_labels_train,mac_labels_train,windowx_labels_train, forsale_labels_train,autos_labels_train,motor_labels_train,baseball_labels_train, hockey_labels_train ,crypt_labels_train,electronics_labels_train , med_labels_train,space_labels_train , christ_labels_train, guns_labels_train , mideast_labels_train, poli_labels_train, rel_labels_train], 0)
    #print("total labels",len(y_train))
    # generate labels for test data
    atheism_labels_test =  [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in atheism_examples_test]
    graphics_labels_test = [[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in graphics_examples_test]
    windows_labels_test =  [[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in windows_examples_test]
    hardware_labels_test=  [[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]for _ in hardware_examples_test]
    mac_labels_test=  [[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]for _ in mac_hardware_examples_test]
    windowx_labels_test =  [[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in windows_x_examples_test]
    forsale_labels_test = [[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in forsale_examples_test]
    autos_labels_test =  [[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0] for _ in autos_examples_test]
    motor_labels_test =  [[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]for _ in motorcycles_examples_test]
    baseball_labels_test =  [[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]for _ in baseball_examples_test]
    hockey_labels_test =  [[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0] for _ in hockey_examples_test]
    crypt_labels_test = [[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0] for _ in crypt_examples_test]
    electronics_labels_test =  [[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0] for _ in electronics_examples_test]
    med_labels_test =  [[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]for _ in med_examples_test]
    space_labels_test =  [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]for _ in space_examples_test]
    christ_labels_test  =  [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0] for _ in christian_examples_test]
    guns_labels_test = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0] for _ in guns_examples_test]
    mideast_labels_test =  [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0] for _ in  mideast_examples_test]
    poli_labels_test=  [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]for _ in politics_misc_examples_test]
    rel_labels_test=  [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]for _ in religion_misc_examples_test]
    y_test = np.concatenate([atheism_labels_test,    graphics_labels_test , windows_labels_test ,hardware_labels_test , mac_labels_test , windowx_labels_test , forsale_labels_test , autos_labels_test , motor_labels_test , baseball_labels_test ,   hockey_labels_test  , crypt_labels_test , electronics_labels_test , med_labels_test , space_labels_test , christ_labels_test , guns_labels_test , mideast_labels_test , poli_labels_test , rel_labels_test], 0)
    #print("total labels for test",len(y_test))
    return [x_train, y_train, x_test, y_test]

"""
def batch_iter(data, batch_size, num_epochs, shuffle=True):
   
    #Generates a batch iterator for a dataset.
 
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
"""
