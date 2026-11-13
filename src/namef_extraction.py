import re

# Remplace 'noms_raw.txt' par le chemin vers ton fichier brut
input_filepath = 'noms.txt'
# Le fichier où on va sauvegarder la liste nettoyée
output_filepath = 'clean_noms.txt'

# On stocke les noms uniques dans un set
names_set = set()

# 1) Pattern pour les lignes numérotées tabulé : par ex. "2\tAlaoui\t127,833..."
num_list_pattern = re.compile(r'^(?:\d+)\t([A-Za-zÀ-ÖØ-öø-ÿ\'\-\s]+)\t')

# 2) Pattern pour les noms listés sous chaque lettre (un mot capitalisé, pas juste "A", "B", etc.)
single_name_pattern = re.compile(r'^[A-ZÀ-ÖØ-öø-ÿ][a-zà-öø-ÿ\'\-\s]+$')

# 3) Pattern pour les définitions du type "1. Abergel A Judeo-Spanish..."
def_pattern = re.compile(r'^\d+\. ?([A-Z][A-Za-zÀ-ÖØ-öø-ÿ\'\-\s]+)')

with open(input_filepath, 'r', encoding='utf-8') as infile:
    for raw_line in infile:
        line = raw_line.strip()
        if not line:
            continue

        # Cas 1 : ligne numérotée avec tab
        m1 = num_list_pattern.match(line)
        if m1:
            name = m1.group(1).strip()
            names_set.add(name)
            continue

        # Cas 2 : paragraphe de définition "1. Abergel A Judeo …"
        m3 = def_pattern.match(line)
        if m3:
            full = m3.group(1).strip()
            # On coupe dès qu'on rencontre une lettre minuscule
            name = re.split(r'\s+[a-z]', full)[0].strip()
            names_set.add(name)
            continue

        # Cas 3 : nom isolé sous une lettre (ex : "Abergel", "Aamara"...)
        if single_name_pattern.match(line) and len(line) > 1:
            names_set.add(line)
            continue

# Écriture finale, trié alphabétiquement
with open(output_filepath, 'w', encoding='utf-8') as outfile:
    for nm in sorted(names_set):
        outfile.write(nm + '\n')

print(f"Extraction terminée : {len(names_set)} noms uniques enregistrés dans '{output_filepath}'.")
