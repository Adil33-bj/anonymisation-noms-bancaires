import os
import pandas as pd
from cryptography.fernet import Fernet

# 1. Charger la clé Fernet (doit être définie en variable d'environnement)
key_env = os.getenv('FERNET_KEY')
if not key_env:
    raise RuntimeError("ERREUR : la variable d'env FERNET_KEY n'est pas définie.")
key = key_env.encode()
cipher = Fernet(key)

# 2. Fonctions de déchiffrement
def decrypt_token(token: str) -> str:
    """Retourne le texte d’origine à partir du token chiffré."""
    return cipher.decrypt(token.encode()).decode()

# 3. Exemple A : simple test de décryptage sur vos fichiers de noms/prénoms
# -----------------------------------------------------------------------
# Lit le CSV contenant uniquement la colonne 'Nom_encrypted'
df_noms_enc = pd.read_csv('noms_marocains_encrypted.csv', header=0)
# On suppose que la colonne s’appelle exactement 'Nom_encrypted'
df_noms_enc['Nom_decrypted'] = df_noms_enc['Nom_encrypted'].apply(decrypt_token)

# Même chose pour les prénoms
df_prenoms_enc = pd.read_csv('prenoms_marocains_encrypted.csv', header=0)
df_prenoms_enc['Prenom_decrypted'] = df_prenoms_enc['Prenom_encrypted'].apply(decrypt_token)

# Sauvegarde des versions déchiffrées (facultatif)
df_noms_enc.to_csv('noms_marocains_decrypted_test.csv', index=False)
df_prenoms_enc.to_csv('prenoms_marocains_decrypted_test.csv', index=False)

print("✔ Fichiers de test générés :")
print("  - 'noms_marocains_decrypted_test.csv' (colonne 'Nom_decrypted')")
print("  - 'prenoms_marocains_decrypted_test.csv' (colonne 'Prenom_decrypted')")


# 4. Exemple B : fusion avec un CSV “principal”
# ---------------------------------------------
# Supposons que vous avez un fichier 'clients_banque_brut.csv' qui contient déjà
# deux colonnes chiffrées 'Nom_encrypted' et 'Prenom_encrypted', et d’autres colonnes.
# Ce script montre comment ajouter les colonnes en clair puis sauvegarder.

# Remplacez ce chemin par le nom de votre CSV “principal”
PRINCIPAL_CSV = 'clients_banque_brut.csv'

if os.path.isfile(PRINCIPAL_CSV):
    df_bank = pd.read_csv(PRINCIPAL_CSV, header=0)

    # Si vos colonnes s’appellent différemment, ajustez ici :
    if 'Nom_encrypted' not in df_bank.columns or 'Prenom_encrypted' not in df_bank.columns:
        raise KeyError("Les colonnes 'Nom_encrypted' et/ou 'Prenom_encrypted' sont introuvables dans le fichier.")

    # 4.1. Déchiffrer
    df_bank['Nom'] = df_bank['Nom_encrypted'].apply(decrypt_token)
    df_bank['Prenom'] = df_bank['Prenom_encrypted'].apply(decrypt_token)

    # 4.2. (Optionnel) Supprimer les colonnes chiffrées
    # df_bank = df_bank.drop(columns=['Nom_encrypted', 'Prenom_encrypted'])

    # 4.3. Sauvegarder le nouveau CSV complet avec noms/prénoms en clair
    df_bank.to_csv('clients_banque_decrypted.csv', index=False)
    print("✔ Fichier fusionné et déchiffré : 'clients_banque_decrypted.csv'")
else:
    print(f"⚠ Le fichier '{PRINCIPAL_CSV}' n'existe pas ; j’ai seulement testé le décryptage des listes de noms/prénoms.")

