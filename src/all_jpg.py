import os

# Defina os rótulos das classes
labels = {'acerola': 0, 'amora': 1, 'bacuri': 2, 'banana': 3, 'caja': 4}

# Caminho base para as imagens
base_path = 'C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic-haralick/images/'

# Loop para processar as imagens de todas as classes
for class_name, label in labels.items():
    class_path = os.path.join(base_path, class_name)

    if not os.path.exists(class_path):
        print(f"A pasta '{class_path}' não existe. Pulando para a próxima classe.")
        continue

    # Renomear arquivos para .jpg
    for imagem_nome in os.listdir(class_path):
        if not imagem_nome.endswith(".jpg"):
            old_path = os.path.join(class_path, imagem_nome)
            new_nome = f"{label}_{imagem_nome[:-4]}.jpg"  # Prefixo com o label e extensão .jpg
            new_path = os.path.join(class_path, new_nome)
            
            try:
                os.rename(old_path, new_path)
                print(f"Arquivo renomeado: {old_path} -> {new_path}")
            except Exception as e:
                print(f"Erro ao renomear o arquivo: {old_path} -> {new_path}. Erro: {e}")
