import pandas

df = pandas.read_csv('Database\Data\instances.csv', skipfooter=120341)# On récupère que le signer 1

#for i in range (len(df["id"])):
#    if df.index

compteur = 0
for i in range(len(df)):
    if df.loc[i, "sign"]=="AVANCER":# AUSSI et AVANCER sont utilisés 22 et 14 fois dans la base de donnée téléchargée, ils constituent une bonne base pour l'apprentissage
        compteur += 1

print(compteur)