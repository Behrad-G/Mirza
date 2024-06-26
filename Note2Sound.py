import time
import numpy as np
import pyaudio
import math
from scipy.io.wavfile import write


def sinewavegen(fs,duration,f):
    volume = 1  # range [0.0, 1.0]

    # generate samples, note conversion to float32 array
    samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)

    # explicitly convert to bytes sequence
    output_byte=(volume * samples)
    for i in range(0,len(output_byte)):
        output_byte[i]=output_byte[i]*math.exp(-2*i/fs) # Apply exponential decay to the signal
    
    return output_byte

fs = 44100  # sampling rate, Hz, must be integer
p = pyaudio.PyAudio()



def note_attributes(note_title):
    duration_dict = {
        "0Sol_1" : 2, "0Sol_2" : 1, "0Sol_3" : 0.5, "0Sol_4" : 0.25,
        "0La_1": 2, "0La_2": 1, "0La_3": 0.5, "0La_4": 0.25,
        "0Si_1": 2, "0Si_2": 1, "0Si_3": 0.5, "0Si_4": 0.25,
        "0Do_1":   2, "0Do_2":   1, "0Do_3":   0.5, "0Do_4":   0.25,
        "0Re_1": 2, "0Re_2": 1, "0Re_3": 0.5, "0Re_4": 0.25,
        "0Mi_1": 2, "0Mi_2": 1, "0Mi_3": 0.5, "0Mi_4": 0.25,
        "1Fa_1":   2, "1Fa_2":   1, "1Fa_3":   0.5, "1Fa_4":   0.25,
        "1Sol_1" : 2, "1Sol_2" : 1, "1Sol_3" : 0.5, "1Sol_4" : 0.25,
        "1La_1": 2, "1La_2": 1, "1La_3": 0.5, "1La_4": 0.25,
        "1Si_1": 2, "1Si_2": 1, "1Si_3": 0.5, "1Si_4": 0.25,
        "1Do_1":   2, "1Do_2":   1, "1Do_3":   0.5, "1Do_4":   0.25,
        "1Re_1": 2, "1Re_2": 1, "1Re_3": 0.5, "1Re_4": 0.25,
        "1Mi_1": 2, "1Mi_2": 1, "1Mi_3": 0.5, "1Mi_4": 0.25,
        "2Fa_1":   2, "2Fa_2":   1, "2Fa_3":   0.5, "2Fa_4":   0.25, 
        "2Sol_1" : 2, "2Sol_2" : 1, "2Sol_3" : 0.5, "2Sol_4" : 0.25    
    }

    frequency_dict = {
		"0Sol_1" : 196,"0Sol_2" : 196,"0Sol_3" : 196,"0Sol_4" : 196,
		"0La_1" : 213.8,"0La_2" : 213.8,"0La_3" : 213.8,"0La_4" : 213.8,
		"0Si_1" : 233.08,"0Si_2" : 233.08,"0Si_3" : 233.08,"0Si_4" : 233.08,
		"0Do_1" : 261.62,"0Do_2" : 261.62,"0Do_3" : 261.62,"0Do_4" : 261.62,
		"0Re_1" : 285.4,"0Re_2" : 285.4,"0Re_3" : 285.4,"0Re_4" : 285.4,
		"0Mi_1" : 311.2,"0Mi_2" : 311.2,"0Mi_3" : 311.2,"0Mi_4" : 311.2,
		"1Fa_1" : 349.2,"1Fa_2" : 349.2,"1Fa_3" : 349.2,"1Fa_4" : 349.2,
		"1Sol_1" : 392,"1Sol_2" : 392,"1Sol_3" : 392,"1Sol_4" : 392,
		"1La_1" : 427.6,"1La_2" : 427.6,"1La_3" : 427.6,"1La_4" : 427.6,
		"1Si_1" : 466.16,"1Si_2" : 466.16,"1Si_3" : 466.16,"1Si_4" : 466.16,
		"1Do_1" : 523.24,"1Do_2" : 523.24,"1Do_3" : 523.24,"1Do_4" : 523.24,
		"1Re_1" : 570.8,"1Re_2" : 570.8,"1Re_3" : 570.8,"1Re_4" : 570.8,
		"1Mi_1" : 622.4,"1Mi_2" : 622.4,"1Mi_3" : 622.4,"1Mi_4" : 622.4,
		"2Fa_1" : 698.4,"2Fa_2" : 698.4,"2Fa_3" : 698.4,"2Fa_4" : 698.4,
		"2Sol_1" : 784,"2Sol_2" : 784,"2Sol_3" : 784,"2Sol_4" : 784
    }

    return duration_dict[note_title],frequency_dict[note_title] 

output_byte=sinewavegen(fs,0.01,100)

# Open and read notes from file
with open("./notes/notes.txt", "r") as notes_file:
    notes = notes_file.read().split(',')

# Generate sine waves for each note and append to output
for note in notes:
    duration, f = note_attributes(note)
    output_byte = np.append(output_byte, sinewavegen(fs, duration, f), axis=None)

# Convert output byte array to bytes
output_bytes = output_byte.tobytes()

# Open PyAudio stream
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=fs,
                output=True)

# Play the sound
start_time = time.time()
stream.write(output_bytes)
print("Played sound for {:.2f} seconds".format(time.time() - start_time))

# Stop and close the stream
stream.stop_stream()
stream.close()

# Terminate PyAudio
p.terminate()

# Write output to a WAV file
write('./export/audio.wav', fs, output_byte)
