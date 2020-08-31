from flask import Flask, render_template, redirect, request
import re
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from enc_dec import Encoder, BahdanauAttention , Decoder

app = Flask(__name__)

def preprocess_sentence(w):
  w = re.sub(r"([?.!,¿])", r" \1 ", w)# creating a space between a word and the punctuation following it
  w = re.sub(r'[" "]+', " ", w)
  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)# replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")

  w = w.strip()
  w = '<start> ' + w + ' <end>'
  return w

file = open('inp_lang.json', 'r')
inp_lang = file.read()
file.close()
inp_lang = tokenizer_from_json(inp_lang) 

file = open('targ_lang.json', 'r')
targ_lang = file.read()
file.close()
targ_lang = tokenizer_from_json(targ_lang)

BATCH_SIZE = 64
embedding_dim = 256
units = 1024
vocab_input_size = len(inp_lang.index_word) + 1
vocab_target_size = len(targ_lang.index_word) + 1
max_length_inp = 11
max_length_targ = 18

encoder = Encoder(vocab_input_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_target_size, embedding_dim, units, BATCH_SIZE)

encoder.load_weights('encoder.h5')
decoder.load_weights('decoder.h5')

def evaluate(sentence):
  sentence = [preprocess_sentence(sentence)]
  inputs = inp_lang.texts_to_sequences(sentence)
  inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen = max_length_inp, padding='post')

  hidden = tf.zeros((1, units))
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
  
  result = ''

  for t in range(max_length_targ):
    predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

    predicted_id = tf.argmax(predictions[0]).numpy()

    if targ_lang.index_word[predicted_id] == '<end>':
      return result

    result += targ_lang.index_word[predicted_id] + ' '


    dec_input = tf.expand_dims([predicted_id], 0)

  return result



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        if request.method == 'POST':
            eng_sentence = request.form['eng']
            trans = evaluate(eng_sentence)
            return render_template('index.html', spa = trans)

if __name__ == "__main__":
    app.run(debug=True)