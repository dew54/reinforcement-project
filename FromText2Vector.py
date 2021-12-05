#script che legge il csv di registrazione di gioco e converte in un vettore riga
#es su,giù,dx,sx,x
#premo su e sarà 10000

import csv
import pandas as pd

df=pd.read_csv('keys.csv', sep='\t')
print(df)

#up
df = df.replace('''['UP']''', "10000")
df = df.replace('''['UP', 'MOTION_UP']''', "10000")

#down
df = df.replace('''['DOWN']''', "01000")
df = df.replace('''['DOWN', 'MOTION_DOWN']''', "01000")

#right
df = df.replace('''['RIGHT']''', "00100")
df = df.replace('''['RIGHT', 'MOTION_RIGHT']''', "00100")

#left
df = df.replace('''['LEFT']''', "00010")
df = df.replace('''['LEFT', 'MOTION_LEFT']''', "00010")

#jump
df = df.replace('''['X']''', "00001")

#scrivo i risultati

df.to_csv('key_vector.csv', index=False)



print(df)
