import os
import pandas as pd
from cryptography.fernet import Fernet

# 1. Clé Fernet
key_env = os.getenv('FERNET_KEY')
if not key_env:
    key = Fernet.generate_key()
    print("Nouvelle clé générée (à stocker en lieu sûr) :", key.decode())
else:
    key = key_env.encode()
cipher = Fernet(key)

# 2. Fonctions de chiffrement / déchiffrement
def encrypt_name(name: str) -> str:
    return cipher.encrypt(name.strip().encode()).decode()

def decrypt_name(token: str) -> str:
    return cipher.decrypt(token.encode()).decode()

# 3. Charger les CSV (colonnes “Nom” et “Prenom”)
df_noms = pd.read_csv('noms_marocains.csv', header=0)
df_prenoms = pd.read_csv('prenoms_marocains.csv', header=0)

# 4. Enlever d’éventuelles lignes d’en-tête “Nom” / “Prenom” en data
df_noms = df_noms[df_noms['Nom'] != 'Nom']
df_prenoms = df_prenoms[df_prenoms['Prenom'] != 'Prenom']

# 5. Créer uniquement la colonne chiffrée et supprimer l’original
df_noms['Nom_encrypted'] = df_noms['Nom'].apply(encrypt_name)
df_noms = df_noms[['Nom_encrypted']]

df_prenoms['Prenom_encrypted'] = df_prenoms['Prenom'].apply(encrypt_name)
df_prenoms = df_prenoms[['Prenom_encrypted']]

# 6. Sauvegarder sans la colonne originale
df_noms.to_csv('noms_marocains_encrypted.csv', index=False)
df_prenoms.to_csv('prenoms_marocains_encrypted.csv', index=False)

print("Fichiers générés : 'noms_marocains_encrypted.csv' et 'prenoms_marocains_encrypted.csv'.")
