import pandas as pd
import unicodedata
import re

# Étape 1 : lire le fichier brut
with open("prenoms.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Étape 2 : fonction de nettoyage
def clean_prenom(word):
    word = word.strip()
    word = unicodedata.normalize("NFKD", word)  # supprime accents si besoin
    word = ''.join(c for c in word if c.isalpha())  # garde que lettres
    return word.capitalize()

# Étape 3 : split + filtrage
prenoms = [clean_prenom(word) for word in raw_text.split()]

# Étape 4 : filtres logiques
mots_interdits = {"names", "popular", "unisex", "girls", "for", "only", "first", "abc", "allah"}
prenoms = [
    p for p in prenoms 
    if 2 < len(p) < 20 
    and p.lower() not in mots_interdits
    and not re.search(r'(abc|test|allah)', p.lower())
]

# Étape 5 : dédoublonnage + tri
prenoms = sorted(set(prenoms))

# Étape 6 : export en CSV
df = pd.DataFrame(prenoms, columns=["Prenom"])
df.to_csv("prenoms_marocains.csv", index=False)

print("Fichier nettoyé exporté : prenoms_marocains.csv")
