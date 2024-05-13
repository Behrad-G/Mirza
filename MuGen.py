
from tensorflow import keras
from keras import layers
from keras.layers import Dense, Dropout, LSTM, Activation
from keras import Sequential
import numpy as np
import random

Notes_ids=[26,	30,	34,	38,	34,	30,	34,	30,	34,	35,	34,	28,	39,	34,	28,	39,	34,	28,	43,	38,	32,	39,	34,	28,	42,	43,	38,	32,	38,	39,	34,	28,	35,	39,	42,	43,	38,	32,	31,	35,	38,	39,	34,	29,	39,	34,	30,	39,	34,	30,	32,	27,	27,	27,	30,	31,	27,	30,	27,	30,	29,	41,	28,	60,	27,	28,	33,	37,	35,	28,	33,	37,	35,	28,	33,	37,	43,	36,	33,	33,	39,	34,	28,	39,	34,	28,	43,	38,	32,	39,	34,	28,	42,	43,	38,	32,	38,	39,	34,	28,	35,	39,	42,	43,	38,	32,	31,	35,	38,	39,	34,	29,	39,	34,	30,	32,	27,	27,	27,	30,	31,	27,	30,	27,	30,	29,	41,	28,	60,	26,	38,	34,	38,	34,	38,	33,	37,	33,	37,	34,	38,	43,	38,	32,	30,	34,	30,	34,	30,	34,	29,	33,	29,	33,	30,	39,	34,	28,	39,	35,	39,	43,	43,	39,	34,	34,	29,	39,	35,	39,	43,	43,	39,	34,	34,	29,	39,	35,	39,	43,	47,	43,	47,	47,	43,	39,	35,	39,	43,	43,	39,	34,	34,	29,	39,	35,	39,	43,	47,	43,	47,	47,	43,	47,	47,	43,	47,	47,	43,	37,	35,	39,	43,	43,	39,	43,	43,	39,	43,	43,	39,	33,	35,	39,	35,	39,	43,	43,	39,	43,	43,	39,	42,	43,	39,	35,	31,	34,	34,	28,	60,	26,	30,	34,	38,	42,	47,	43,	38,	39,	42,	40,	42,	46,	42,	38,	34,	30,	24,	26,	30,	34,	38,	42,	47,	43,	38,	39,	42,	40,	43,	46,	51,	47,	43,	46,	46,	47,	43,	39,	35,	39,	43,	43,	39,	43,	43,	39,	42,	42,	37,	32,	38,	37,	38,	41,	42,	46,	41,	41,	46,	41,	42,	45,	45,	41,	38,	42,	38,	41,	41,	36,	54,	49,	50,	45,	46,	50,	45,	41,	46,	41,	42,	45,	45,	41,	38,	42,	38,	41,	41,	36,	38,	46,	46,	40,	38,	42,	42,	36,	38,	50,	50,	44,	38,	46,	46,	40,	38,	42,	42,	36,	38,	55,	51,	55,	51,	51,	47,	51,	47,	47,	43,	47,	43,	43,	39,	43,	39,	39,	35,	39,	35,	35,	31,	32,	39,	35,	39,	42,	47,	39,	42,	47,	39,	42,	47,	39,	42,	47,	43,	47,	43,	51,	47,	43,	39,	47,	43,	39,	35,	43,	39,	35,	31,	34,	28,	60,	42,	42,	36,	32,	47,	42,	42,	36,	32,	35,	39,	43,	43,	39,	43,	43,	39,	43,	43,	39,	43,	43,	39,	43,	51,	43,	39,	51,	39,	35,	51,	43,	39,	51,	39,	35,	51,	43,	39,	51,	39,	35,	51,	31,	35,	39,	39,	31,	35,	39,	39,	31,	35,	39,	39,	35,	39,	43,	43,	39,	43,	43,	39,	41,	43,	39,	35,	31,	34,	34,	28,	36,	38,	42,	46,	42,	46,	42,	46,	42,	46,	47,	43,	37,	34,	38,	42,	38,	42,	38,	42,	38,	42,	43,	39,	33,	31,	35,	39,	39,	31,	35,	39,	39,	31,	35,	39,	39,	35,	39,	43,	43,	39,	43,	43,	39,	43,	43,	39,	41,	43,	39,	35,	31,	34,	34,	28,	60,	50,	51,	44,	47,	51,	47,	51,	47,	51,	47,	51,	47,	51,	47,	51,	47,	51,	47,	51,	47,	51,	47,	51,	47,	51,	47,	51,	47,	51,	55,	51,	55,	51,	55,	51,	55,	51,	55,	51,	55,	51,	55,	51,	55,	51,	55,	51,	55,	51,	52,	54,	55,	51,	47,	51,	47,	51,	47,	51,	47,	51,	47,	51,	47,	51,	47,	51,	47,	51,	47,	51,	47,	51,	47,	48,	50,	51,	47,	43,	47,	43,	47,	43,	47,	43,	47,	43,	47,	43,	47,	43,	47,	43,	47,	43,	46,	46,	40,	42,	50,	51,	46,	45,	45,	42,	50,	51,	46,	45,	45,	43,	55,	50,	55,	51,	46,	51,	47,	49,	45,	47,	51,	47,	51,	47,	51,	47,	50,	55,	51,	47,	43,	46,	46,	40,	43,	46,	43,	46,	43,	46,	43,	54,	50,	46,	43,	50,	46,	42,	39,	46,	42,	38,	35,	42,	38,	35,	31,	35,	31,	35,	31,	35,	31,	34,	34,	28,	60,	22,	23,	16,	18,	26,	22,	18,	22,	18,	14,	18,	12,	14,	22,	18,	14,	18,	14,	10,	14,	8,	10,	18,	14,	10,	14,	10,	6,	10,	4,	6,	14,	10,	6,	10,	6,	3,	6,	6,	0,	60,	11,	15,	19,	19,	18,	18,	18,	18,	19,	15,	19,	22,	23,	18,	13,	11,	15,	15,	15,	14,	14,	14,	14,	15,	11,	15,	18,	19,	14,	9,	11,	15,	19,	19,	18,	18,	18,	18,	19,	15,	19,	22,	23,	18,	14,	15,	19,	23,	23,	22,	22,	22,	23,	19,	23,	26,	27,	22,	18,	19,	15,	19,	22,	23,	18,	13,	15,	11,	15,	18,	19,	14,	8,	19,	15,	10,	19,	15,	10,	19,	14,	23,	18,	27,	23,	19,	23,	19,	15,	19,	15,	10,	14,	10,	14,	10,	14,	10,	19,	14,	23,	19,	15,	19,	15,	10,	14,	10,	14,	10,	19,	14,	14,	14,	14,	60,	26,	39,	35,	39,	35,	39,	33,	37,	33,	37,	34,	38,	43,	38,	32,	30,	34,	30,	34,	29,	33,	29,	33,	30,	39,	34,	28,	39,	35,	39,	43,	43,	39,	43,	35,	39,	31,	35,	31,	35,	31,	35,	31,	34,	34,	29,	39,	35,	39,	43,	43,	39,	43,	43,	39,	43,	43,	39,	43,	35,	39,	31,	35,	31,	35,	31,	35,	31,	34,	34,	29,	39,	35,	39,	41,	37,	41,	39,	41,	37,	41,	39,	43,	47,	43,	46,	47,	42,	43,	37,	46,	47,	43,	39,	35,	39,	47,	39,	41,	42,	43,	39,	35,	39,	35,	39,	35,	38,	43,	38,	43,	38,	42,	38,	34,	30,	27,	30,	28,	60,	39,	39,	35,	39,	43,	38,	43,	38,	43,	38,	43,	38,	38,	42,	42,	38,	34,	30,	26,	28,	38,	42,	46,	46,	42,	38,	34,	37,	34,	38,	42,	42,	38,	34,	30,	26,	28,	39,	39,	35,	39,	43,	37,	43,	37,	43,	37,	43,	37,	43,	37,	39,	43,	47,	43,	47,	43,	46,	46,	41,	43,	37,	39,	43,	47,	47,	43,	47,	43,	47,	43,	47,	43,	46,	46,	41,	43,	37,	35,	39,	43,	43,	39,	43,	39,	43,	39,	43,	39,	42,	42,	37,	39,	33,	30,	35,	39,	39,	34,	30,	35,	39,	39,	34,	30,	35,	39,	39,	35,	39,	39,	35,	39,	39,	35,	39,	43,	39,	43,	47,	43,	46,	47,	43,	39,	35,	39,	47,	39,	42,	42,	43,	39,	35,	39,	35,	39,	35,	38,	43,	38,	43,	38,	42,	38,	34,	30,	27,	30,	28,	39,	39,	35,	39,	43,	37,	43,	37,	43,	38,	43,	38,	38,	42,	46,	51,	45,	47,	41,	43,	46,	47,	41,	43,	36,	46,	47,	43,	39,	43,	47,	43,	39,	43,	47,	43,	39,	43,	47,	43,	39,	43,	47,	42,	43,	46,	47,	41,	43,	37,	50,	51,	47,	43,	47,	51,	47,	43,	47,	51,	47,	43,	47,	51,	47,	43,	47,	51,	46,	47,	50,	51,	45,	47,	41,	46,	47,	43,	39,	43,	47,	43,	39,	43,	47,	43,	39,	43,	47,	43,	39,	43,	47,	42,	43,	46,	47,	41,	43,	37,	42,	43,	39,	35,	39,	43,	39,	35,	39,	43,	39,	35,	39,	43,	39,	35,	38,	38,	38,	38,	38,	38,	39,	33,	28,	39,	39,	35,	39,	43,	37,	35,	31,	25,	39,	39,	35,	39,	43,	37,	35,	31,	25,	34,	38,	42,	38,	42,	38,	42,	38,	43,	37,	35,	31,	26,	34,	38,	42,	38,	42,	38,	42,	38,	42,	46,	47,	43,	39,	43,	47,	43,	39,	43,	47,	43,	39,	43,	47,	43,	39,	43,	47,	42,	43,	46,	47,	41,	39,	42,	43,	37,	50,	51,	47,	43,	47,	51,	47,	43,	47,	51,	47,	43,	47,	51,	47,	43,	47,	51,	46,	47,	50,	51,	45,	43,	46,	47,	41,	39,	42,	43,	37,	35,	38,	38,	38,	38,	38,	38,	39,	33,	28,	39,	35,	39,	41,	37,	41,	39,	41,	37,	41,	39,	41,	37,	41,	39,	43,	47,	43,	46,	47,	42,	43,	37,	35,	39,	43,	39,	42,	43,	38,	39,	34,	30,	38,	42,	45,	43,	38,	38,	38,	38,	38,	38,	46,	41,	38,	42,	42,	38,	34,	30,	35,	27,	30,	29,	42,	42,	28,	60,	39,	35,	39,	42,	41,	47,	43,	39,	42,	41,	47,	42,	47,	38,	47,	42,	47,	38,	47,	42,	47,	39,	47,	42,	47,	38,	47,	42,	47,	39,	47,	42,	47,	38,	47,	42,	47,	39,	47,	42,	47,	39,	47,	42,	47,	39,	47,	42,	47,	39,	47,	43,	47,	39,	47,	43,	47,	39,	47,	43,	47,	39,	47,	43,	47,	39,	47,	43,	47,	43,	47,	43,	47,	43,	47,	39,	43,	39,	43,	35,	39,	35,	39,	31,	35,	30,	34,	34,	28,	31,	31,	35,	35,	31,	35,	39,	39,	35,	39,	43,	43,	39,	43,	47,	47,	43,	47,	51,	51,	47,	51,	46,	47,	49,	51,	47,	43,	47,	43,	47,	43,	47,	43,	46,	46,	41,	47,	43,	43,	39,	43,	43,	47,	43,	43,	39,	43,	43,	35,	39,	39,	39,	43,	43,	47,	43,	43,	39,	43,	43,	35,	39,	39,	39,	43,	43,	47,	43,	43,	39,	43,	43,	35,	39,	39,	39,	43,	43,	51,	47,	43,	39,	43,	43,	35,	39,	39,	39,	43,	43,	55,	51,	47,	43,	47,	43,	51,	47,	43,	39,	43,	39,	35,	39,	35,	31,	35,	31,	35,	39,	35,	39,	43,	39,	43,	47,	43,	47,	51,	47,	49,	51,	47,	43,	47,	43,	47,	43,	46,	46,	40,	60,	50,	46,	50,	46,	50,	46,	50,	46,	49,	42,	46,	42,	46,	42,	46,	42,	45,	42,	50,	46,	49,	42,	46,	42,	45,	42,	54,	50,	26,	50,	54,	50,	53,	54,	55,	51,	47,	43,	46,	47,	49,	45,	47,	51,	47,	51,	47,	51,	47,	50,	55,	51,	47,	43,	46,	46,	40,	60,	42,	39,	43,	47,	49,	44,	50,	51,	47,	43,	47,	43,	46,	46,	40,	42,	39,	43,	47,	49,	44,	42,	39,	43,	47,	49,	44,	42,	46,	49,	45,	42,	46,	49,	45,	46,	50,	52,	54,	55,	51,	47,	43,	46,	47,	49,	45,	47,	51,	47,	51,	47,	51,	47,	50,	55,	51,	47,	43,	46,	46,	40,	52,	50,	54,	51,	53,	50,	54,	51,	53,	50,	54,	51,	55,	50,	54,	51,	55,	50,	54,	51,	55,	46,	50,	47,	51,	46,	50,	47,	51,	42,	46,	43,	47,	43,	47,	50,	47,	51,	47,	51,	54,	59,	54,	59,	54,	59,	55,	59,	55,	51,	55,	51,	47,	51,	47,	49,	46,	47,	51,	47,	51,	47,	50,	55,	51,	47,	41,	35,	39,	41,	35,	39,	41,	42,	43,	39,	35,	30,	35,	38,	39,	35,	31,	26,	38,	39,	35,	30,	34,	38,	42,	46,	51,	51,	47,	47,	43,	43,	39,	39,	35,	35,	37,	39,	35,	39,	43,	43,	39,	43,	43,	39,	43,	43,	39,	43,	43,	39,	43,	47,	43,	46,	47,	47,	43,	39,	35,	39,	42,	43,	43,	39,	35,	31,	38,	42,	45,	43,	38,	38,	38,	38,	38,	46,	41,	38,	46,	46,	38,	34,	30,	35,	27,	30,	29,	42,	42,	28,	60,	47,	43,	39,	42,	42,	42,	42,	42,	42,	42,	43,	38,	34,	29,	43,	39,	35,	38,	38,	38,	38,	38,	38,	39,	34,	29,	47,	43,	39,	42,	42,	42,	42,	42,	42,	42,	43,	38,	34,	29,	43,	39,	35,	38,	38,	38,	38,	38,	38,	39,	34,	29,	37,	25,	39,	39,	39,	38,	25,	38,	26,	38,	26,	38,	42,	45,	46,	47,	43,	39,	43,	39,	43,	39,	43,	39,	43,	39,	42,	42,	37,	38,	42,	46,	42,	46,	42,	46,	42,	46,	47,	43,	39,	43,	39,	43,	39,	43,	39,	43,	39,	42,	42,	37,	38,	42,	46,	42,	46,	42,	46,	42,	46,	47,	43,	39,	43,	39,	43,	39,	43,	39,	43,	39,	42,	42,	43,	39,	35,	39,	35,	39,	35,	39,	35,	39,	35,	38,	43,	38,	43,	38,	43,	38,	38,	42,	45,	43,	38,	38,	38,	38,	38,	46,	41,	38,	46,	42,	38,	34,	30,	35,	27,	30,	27,	30,	42,	41,	28,	28,	60]

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 15
step = 1
sentences = []
next_notes = []
len_notes= 61

for i in range(0, len(Notes_ids) - maxlen, step):
    sentences.append(Notes_ids[i : i + maxlen])
    next_notes.append(Notes_ids[i + maxlen])

print("Number of sequences:", len(sentences))

x = np.zeros((len(sentences), maxlen, len_notes), dtype=bool)
y = np.zeros((len(sentences), len_notes), dtype=bool)

for i, sentence in enumerate(sentences):
    for t, noteid in enumerate(sentence):
        x[i, t, noteid] = 1
    y[i, next_notes[i]] = 1


# Define the model architecture
model = Sequential()

# Add the first LSTM layer with 256 units, input shape of (maxlen, len_notes), and return sequences
model.add(LSTM(256, input_shape=(maxlen, len_notes), return_sequences=True))
model.add(Dropout(0.3))  # Add dropout regularization

# Add the second LSTM layer with 512 units and return sequences
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))  # Add dropout regularization

# Add the third LSTM layer with 256 units
model.add(LSTM(256))


# Add a fully connected dense layer with 256 units
model.add(Dense(256))
model.add(Dropout(0.3))  # Add dropout regularization

# Add the output layer with len_notes units and softmax activation
model.add(Dense(len_notes))
model.add(Activation('softmax'))

# Compile the model with categorical crossentropy loss and rmsprop optimizer
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64") # Convert to numpy array and cast to float64
    preds = np.log(preds) / temperature  # Scale probabilities
    exp_preds = np.exp(preds) # Compute exponential of scaled probabilities
    preds = exp_preds / np.sum(exp_preds) # Normalize probabilities
    probas = np.random.multinomial(1, preds, 1) # Sample from multinomial distribution
    return np.argmax(probas) # Return the index with the highest probability


epochs = 20
batch_size = 100

# Select a random starting index for generating notes
start_index = random.randint(0, len(Notes_ids))

# Extract a sequence of notes from the starting index
sentences=Notes_ids[start_index : start_index + maxlen]

# Initialize x_pred for prediction with zeros
x_pred = np.zeros((1, maxlen, len_notes), dtype=bool)

# Convert the sequence of notes to a one-hot encoded format
for t, noteid in enumerate(sentences):
    x_pred[0, t, noteid] = 1

# Training loop
for epoch in range(epochs):
    # Fit the model to the training data
    model.fit(x, y, batch_size=batch_size, epochs=50)
    print()
    # Optional: Generate notes after each epoch
    # print("Generating notes after epoch: %d" % epoch)

# Optionally save the trained model
# model.save('my_model.h5')
    
# Iterate over different diversities
    for diversity in [0.5,1,2,4,6]:
        print("...Diversity:", diversity)
        generated=[]

        # Initialize sentences with a sequence of notes
        sentences=Notes_ids[start_index : start_index + maxlen]

        # Generate notes
        for i in range(1000):
            # Predict the next note based on the current sequence
            preds = model.predict(x_pred, verbose=0)[0]
            next_note = sample(preds, diversity)
            generated.append(next_note)
            
            # Append the predicted note to the sequence
            sentences.append(next_note)

            # Prepare input for the next prediction
            pred_sentence=sentences[i : i + maxlen]
            x_pred = np.zeros((1, maxlen, len_notes), dtype=bool)
            for t, noteid in enumerate(pred_sentence):
                x_pred[0, t, noteid] = 1

        # Print generated notes, sentences, and predicted sentences       
        print("...Generated: ", generated)
        print("...sentences: ", sentences)
        print("...pred sentence: ", pred_sentence)                
        print()

        # Save generated sentences to a file
        filename='predictions\epoch='+str(epoch)+'_diversity='+str(diversity)+'.txt'
        np.savetxt(filename,sentences)

# Save the model
model.save('my_model.keras')