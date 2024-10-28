import os
import cv2
import mahotas
import pandas as pd

# rótulos das classes
labels = {'Acerola': 0, 'Amora': 1, 'Bacuri': 2, 'Banana': 3, 'Caja': 4, 
          'Caju': 5, 'Goiaba': 6, 'Graviola': 7, 'Mamao': 8, 'Manga': 9,
          'Maracuja': 10, 'Pinha': 11}

# lista para armazenar as características
all_features = []

# loop para processar as imagens de todas as classes
for class_name, label in labels.items():
    class_path = f'C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic-haralick/images/{class_name}'

    if not os.path.exists(class_path):
        print(f"A pasta '{class_path}' não existe. Pulando para a próxima classe.")
        continue

    # extrai características para cada imagem na classe
    for imagem_nome in os.listdir(class_path):
        if imagem_nome.endswith(".jpg"):
            imagem_path = os.path.join(class_path, imagem_nome)
            image_cv2 = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(image_cv2, (256, 256))

            # características Haralick
            features = mahotas.features.haralick(resized_image, compute_14th_feature=True, return_mean=True).reshape(1, 14)

            # DataFrame temporário com as características e o rótulo
            temp_df = pd.DataFrame(features, columns=[f'Haralick_{i}' for i in range(14)])
            temp_df['Label'] = label

            all_features.append(temp_df)

# combinação de todas as características em um único DataFrame
if all_features:
    df = pd.concat(all_features, ignore_index=True)

    # verificar se o DataFrame não está vazio
    if not df.empty:
        print("DataFrame não está vazio. Salvando arquivo CSV...")

        # salvar o DataFrame em um arquivo CSV
        output_path = 'C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic-haralick/etc/features.csv'
        try:
            df.to_csv(output_path, index=False, sep=';')
            print(f"Arquivo CSV salvo com sucesso em {output_path}")
        except Exception as e:
            print(f"Erro ao salvar o arquivo CSV: {e}")
    else:
        print("DataFrame está vazio. Nenhum arquivo CSV será salvo.")
else:
    print("Nenhuma característica foi extraída. Nenhum arquivo CSV será salvo.")
