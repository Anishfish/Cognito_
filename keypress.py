from pynput import keyboard 
import time
from datetime import datetime
#from run import analysis
#from sendemail import send_p, send_a


## print(hand['f'])

global tame 
global cow 

hand = {
    "a": 'L', 'A': 'L',
    'b': 'R', 'B': 'R',
    'c': 'L', 'C': 'L',
    'd': 'L', 'D': 'L',
    'e': 'L', 'E': 'L',
    'f': 'L', 'F': 'L',
    'g': 'L', 'G': 'L',
    'h': 'R', 'H': 'R',
    'i': 'R', 'I': 'R',
    'j': 'R', 'J': 'R',
    'k': 'R', 'K': 'R',
    'l': 'R', 'L': 'R',
    'm': 'R', 'M': 'R',
    'n': 'R', 'N': 'R',
    'o': 'R', 'O': 'R',
    'p': 'R', 'P': 'R',
    'q': 'L', 'Q': 'L',
    'r': 'L', 'R': 'L',
    's': 'L', 'S': 'L',
    't': 'L', 'T': 'L',
    'u': 'R', 'U': 'R',
    'v': 'L', 'V': 'L',
    'w': 'L', 'W': 'L',
    'x': 'L', 'X': 'L',
    'y': 'R', 'Y': 'R',
    'z': 'L', 'Z': 'L'

        }

def k_press():
    t = {
        'UserKey': 'Patient1234',
        'Date': datetime.today().strftime('%d-%m-%y').replace("-", ""),
        'Hand':"",
        'Timestamp': datetime.today().strftime('%H:%M:%S'),
        'time': time.time(),
        'time_flight': time.time(),
        'released': True,
        'keys_pressed': 0,
        'keys': []
    }


        


    LOAD_THRESH = 1


    def load_to_csv(keys, fname='/Users/Sumeet/Desktop/CognitoMaster/presses.txt'):
        with open(fname, 'a') as f:
            for key_name, hold_time, release_time, Cow  in keys:
                Date = datetime.today().strftime('%d-%m-%y').replace("-", "")
                timestamp = datetime.today().strftime('%H:%M:%S')
                id = "Patient1234"
                k = str(key_name)
                if '\\' in k:
                    continue
                k = k.replace("'", "")
                k = k.replace("Key.", "")
                k = k.replace(')', 'parenleft')
                k = k.replace(',', 'comma')
                k = k.replace(':', 'colon')
                k = k.replace(';', 'semicolon')
                k = k.replace('.', 'period')
                k = k.replace('(', 'parenright')
                k = k.replace('?', 'question')
                k = k.replace('enter', 'Return')
                if k == 'shift':
                    k = 'Shift_L'
                k = k.replace('shift_r', 'Shift_R')
                k = k.replace('backspace', 'BackSpace')
                x = ""
                k = f.write(id + x*7 + Date +  x*4 + timestamp + " {Cow}, {hold_time},{round(release_time, 4)},{release_time}\n")
        
            
        




    def on_press(key):
        hand = {
    "a": 'L', 'A': 'L',
    'b': 'R', 'B': 'R',
    'c': 'L', 'C': 'L',
    'd': 'L', 'D': 'L',
    'e': 'L', 'E': 'L',
    'f': 'L', 'F': 'L',
    'g': 'L', 'G': 'L',
    'h': 'R', 'H': 'R',
    'i': 'R', 'I': 'R',
    'j': 'R', 'J': 'R',
    'k': 'R', 'K': 'R',
    'l': 'R', 'L': 'R',
    'm': 'R', 'M': 'R',
    'n': 'R', 'N': 'R',
    'o': 'R', 'O': 'R',
    'p': 'R', 'P': 'R',
    'q': 'L', 'Q': 'L',
    'r': 'L', 'R': 'L',
    's': 'L', 'S': 'L',
    't': 'L', 'T': 'L',
    'u': 'R', 'U': 'R',
    'v': 'L', 'V': 'L',
    'w': 'L', 'W': 'L',
    'x': 'L', 'X': 'L',
    'y': 'R', 'Y': 'R',
    'z': 'L', 'Z': 'L',
    'Key.space': 'S'


        }
        cow1 = str(key)
        cow2 = cow1.replace('"', '')
        cow3 = cow2.replace("'", "")
        if(cow3 in hand):
            global cow 
            cow = hand.get(cow3, "")
            print(cow)
        else: 
            pass
        if t['released']:
            t['time'] = time.time()
            t['released'] = False
            print(f'{key} pressed after {time.time() - t["time_flight"]}')
            global tame 
            tame = t['Date'].replace("-", "")
            print(tame)
       
                

    def on_release(key):
        print('{0} held for {1}s'.format(
            key, time.time() - t['time']))
        t['released'] = True
        t['keys_pressed'] += 1
        if t['keys_pressed'] % LOAD_THRESH == 0:
            load_to_csv(t['keys'])
            t['keys'] = []
        id = "Patient1234"
        timestamp = datetime.today().strftime('%H:%M:%S')
        t['keys'].append(
            (
                str(key),
                round(time.time() - t['time'], 4),
                round(time.time() - t['time_flight'], 4),
                cow
            )
        )
        if key == keyboard.Key.esc:
            # Stop listener
            return False

    with keyboard.Listener(
        on_press=on_press,
        on_release=on_release,
    ) as listener:
        listener.join()

k_press()

     
