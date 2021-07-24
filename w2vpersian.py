import tqdm
import io
import os
import csv
import re
import string
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Optimizer, Adam
from tensorflow.keras.preprocessing.sequence import make_sampling_table, skipgrams
from tensorflow.keras.layers import Dot, Flatten, Embedding
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


class DataSet:

    def __init__(self, csv_path: str, txt_path: str):
        self.txt_path = txt_path
        with open(csv_path, "r", encoding="utf-8") as csv_file:
            with open(txt_path, "w", encoding="utf-8") as txt_file:
                [txt_file.write(" ".join(row) + '\n') for row in csv.reader(csv_file)]
        csv_file.close()
        txt_file.close()
        self.preprocess()

    def preprocess(self):
        with open(self.txt_path, "r", encoding="utf-8") as pre_file:
            text = pre_file.read()
        pre_file.close()
        os.remove(self.txt_path)

        with open(self.txt_path, "w", encoding="utf-8") as file:
            new_text = re.sub("\d+|\d+::\d+:\d+|\d+//\d+/\d+|\d+/\d+|(\d+)|[a-z]"
                              , "", text)
            test = new_text.split('\n')
            new_data = []
            for i in range(len(test)):
                if test[i] == '/// ' or test[i] == '::: ':
                    continue
                elif test[i] == '':
                    continue
                else:
                    new_data.append(test[i])
            file.write(''.join(row + "\n" for row in new_data))
        file.close()

    def get_total_vocab_size(self):
        vocab = []
        with open(self.txt_path, "r", encoding="utf-8") as file:
            texts = file.read()
        file.close()
        for word in tqdm.tqdm(texts.split(" ")):
            if word not in vocab:
                vocab.append(word)
        vocab_size = len(vocab)
        return vocab_size, vocab

DataSet("92121483696289.csv", "Dataset.txt")

def standardization(data):
    return tf.strings.regex_replace(data,
                                    '[%s]' % re.escape(string.punctuation), '')


def generate_data(sequences, window_size, negative_samples, vocab_size, seed):
    target_words, context_words, labels = [], [], []

    sampling_table = make_sampling_table(vocab_size)

    for seq in tqdm.tqdm(sequences):

        positive_skip_gram, _ = skipgrams(
            seq,
            window_size=window_size,
            sampling_table=sampling_table,
            vocabulary_size=vocab_size,
            negative_samples=0
        )

        for target_word, context_word in positive_skip_gram:
            context_word = tf.expand_dims(tf.constant([context_word], dtype='int64'), 1)

            negative_sampling, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_word,
                num_true=1,
                unique=True,
                range_max=vocab_size,
                num_sampled=negative_samples,
                seed=seed
            )

            negative_sampling = tf.expand_dims(negative_sampling, 1)

            context = tf.concat([context_word, negative_sampling], 0)

            label = tf.constant([1] + [0] * negative_samples, dtype='int64')

            target_words.append(target_word)
            context_words.append(context)
            labels.append(label)

    return target_words, context_words, labels


def get_vocab_size(path):
    vocab = []
    with open(path, "r", encoding="utf-8") as file:
        texts = file.read()
    file.close()
    for word in tqdm.tqdm(texts.split(" ")):
        if word not in vocab:
            vocab.append(word)
    vocab_size = len(vocab)
    return vocab_size


def get_train_data_file(num_data, file_name):
    with open(file_name, "w") as train_file:
        with open("Dataset.txt", "r") as ds_file:
            texts = ds_file.read()[:num_data]
            train_file.write(texts)
    ds_file.close()
    train_file.close()

    return get_vocab_size(file_name)


train_ds = "train_ds.txt"

vocab_size = get_train_data_file(500000, train_ds)
sequence_length = 20
batch_size = 1024
buffer_size = 10000
autotune = tf.data.AUTOTUNE
num_negative_samples = 4
seed = 1

vectorize_layer = TextVectorization(
    standardize=standardization,
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length
)

dataset = tf.data.TextLineDataset(train_ds).filter(lambda x: tf.cast(tf.strings.length(x), bool))

vectorize_layer.adapt(dataset.batch(batch_size))

vectorized_dataset = dataset.batch(batch_size).prefetch(autotune).map(vectorize_layer).unbatch()

sequences = list(vectorized_dataset.as_numpy_iterator())

targets, contexts, labels = generate_data(sequences,
                                          window_size=2,
                                          negative_samples=num_negative_samples,
                                          vocab_size=vocab_size,
                                          seed=seed)
words_dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
words_dataset = words_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
words_dataset = words_dataset.cache().prefetch(buffer_size=autotune)


class PersianWord2Vec(Model):
    def __init__(self, vocab_size, embedding_dim):
        super(PersianWord2Vec, self).__init__()

        self.target_embedding = Embedding(vocab_size, embedding_dim,
                                          input_length=1,
                                          name="target_embedding")
        self.context_embedding = Embedding(vocab_size, embedding_dim,
                                           input_length=num_negative_samples + 1)

        self.dot = Dot(axes=(3, 2))

        self.flat = Flatten()

    def call(self, target, context):
        target_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)

        dot = self.dot([context_emb, target_emb])

        flat = self.flat(dot)

        return flat


def compute_loss(logits, labels):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)


train_accuracy = CategoricalAccuracy()


@tf.function
def train_step(model: Model, optimizer: Optimizer, target, context, label):
    with tf.GradientTape() as tape:
        logits = model(target, context)
        loss = compute_loss(logits, tf.cast(label, tf.float32))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_accuracy(tf.cast(label, tf.float32), logits)

    return tf.reduce_mean(loss)


pr_w2v = PersianWord2Vec(vocab_size, embedding_dim=128)
optimizer = Adam()


def train(epochs):
    for epoch in range(epochs):
        train_accuracy.reset_state()
        for (targets, contexts), labels in words_dataset:
            loss = train_step(pr_w2v, optimizer, tf.expand_dims(targets, -1), contexts,
                              labels)
            print("Epoch {} ---> loss: {:.2f}, accuracy: {:.2f}".format(epoch + 1, loss, train_accuracy.result() * 100))


train(8)

vocab = vectorize_layer.get_vocabulary()
weights = pr_w2v.get_layer("target_embedding").get_weights()[0]
vectors = io.open("vectors.tsv", "w", encoding="utf-8")
metadata = io.open("metadata.tsv", "w", encoding="utf-8")

for index, word in enumerate(vocab):
    if index == 0:
        continue
    emb_weight = weights[index]
    vectors.write('\t'.join([str(x) for x in emb_weight]) + "\n")
    metadata.write(word + "\n")
vectors.close()
metadata.close()
