# Borrar regiões em vídeos

Aplicativo gráfico para selecionar regiões poligonais fixas e desfocá-las durante um
intervalo de frames. O preview e o processamento usam OpenCV; o FFmpeg exporta AVI
com vídeo I420 (YUV 4:2:0) sem compressão. O OpenCV escreve os frames e o FFmpeg
apenas copia o fluxo de vídeo e preserva o áudio original, quando houver.

## Requisitos

- Python 3.10 ou superior
- FFmpeg instalado e disponível no `PATH` do Windows (`ffmpeg -version`)
- Pacotes Python listados em `requirements.txt`
- Tkinter (incluído nas instalações padrão do Python para Windows)

Instale os pacotes e abra o aplicativo:

```powershell
python -m pip install -r video_anonymization\requirements.txt
python video_anonymization\blur_regions.py
```

## Como usar

1. Abra um vídeo e navegue pela barra de frames ou pelos botões de avançar/voltar.
2. No frame em que o objeto/região está visível, clique em **Usar frame atual como início**.
3. Clique nos vértices do polígono sobre o vídeo; use pelo menos três pontos.
4. Navegue até o frame final desejado e clique em **Usar frame atual como fim**.
5. Clique em **Adicionar região**. Repita para adicionar outras regiões e intervalos.
6. Clique em **Exportar vídeo** e escolha onde salvar o AVI.

O polígono permanece exatamente na mesma posição e formato em todos os frames entre
o início e o fim selecionados; não há tracking automático.

Ao exportar, o aplicativo também procura na pasta do vídeo pelo arquivo
`*_landmarks.h5` criado por `process_video_optimized.py` e grava nele o grupo
`blur_regions`. Esse grupo contém os intervalos inclusivos de frames e os vértices
dos polígonos em coordenadas de pixel (origem no canto superior esquerdo). Os
polígonos são uma representação compacta da máscara: rasterizá-los com OpenCV
recria os pixels cobertos, sem armazenar uma máscara completa para cada frame.
O `process_video_optimized.py` grava os polígonos dinâmicos usados no desfoque
automático de olhos/sobrancelhas, nariz, boca e contorno externo do rosto. Eles
incluem a largura do traço do contorno e o polígono excluído da parte inferior do
rosto. Eles são salvos diretamente em `blur_regions/face_blur`. Ao exportar no
aplicativo, os polígonos manuais são adicionados ao mesmo grupo raiz
(`blur_regions/frame_ranges` e `blur_regions/polygon_points`), preservando os
padrão. Para arquivos HDF5 antigos sem os polígonos do processador,
`save_std_blur_regions.py` os reconstrói a partir dos landmarks salvos, sem
decodificar o vídeo.
Se não houver um HDF5 compatível ou houver vários candidatos sem correspondência
pelo nome do vídeo, a exportação é interrompida para evitar editar o arquivo errado.

### Exemplo da estrutura HDF5

Depois de exportar pelo aplicativo, a estrutura relevante fica assim. `F` é o
número de frames, `M` o número de polígonos manuais e `N` o número de polígonos
padrão salvos (normalmente seis por frame com face detectada):

```text
video_landmarks.h5
├── atributos do arquivo
│   ├── video_filename = "video.avi"
│   ├── frame_width = 640
│   ├── frame_height = 480
│   ├── fps = 30.0
│   ├── coordinate_system = "MediaPipe normalized face coordinates"
│   ├── coordinate_order = "x, y, z"
│   ├── z_definition = "Relative depth estimate; not a physical distance."
│   ├── landmark_count = 478
│   ├── frame_indexing = "Zero-based; dataset row i matches video frame i"
│   └── frame_count = F
├── landmarks                  float32 [F, 478, 3]
│   └── description = "Normalized x, y, z coordinates for the first face in each video frame."
├── face_detected              bool    [F]
│   └── description = "True for a new detection; False when previous landmarks are reused."
├── source_frame               int64   [F]
│   └── description = "Frame index where this row's landmarks were last detected."
└── blur_regions/
    ├── atributos do grupo
    │   ├── schema_version = 1
    │   ├── coordinate_system = "Zero-based video pixel coordinates (x, y), origin at top left"
    │   ├── frame_indexing = "Zero-based frame indices; start_frame and end_frame are inclusive"
    │   ├── frame_width = 640
    │   ├── frame_height = 480
    │   └── description = "Each polygon is blurred on every frame in its inclusive frame range. Rasterizing the polygon reproduces the selected pixel mask."
    ├── frame_ranges             int64   [M, 2]
    ├── polygon_offsets          uint64  [M + 1]
    ├── polygon_points           int32   [total_de_pontos_manuais, 2]
    └── face_blur/
        ├── atributos do grupo
        │   ├── schema_version = 1
        │   ├── coordinate_system = "Zero-based video pixel coordinates (x, y), origin at top left"
        │   ├── frame_indexing = "Zero-based frame indices"
        │   ├── description = "Frame-specific polygons used to build the anonymization mask. Polygons are in full-frame pixel coordinates. Rasterize include polygons, apply stroke_width_px to outside_face_ring, then subtract exclude polygons. The final mask is feathered with a 15x15 Gaussian."
        │   ├── outside_face_ring = "The face-oval polygon is drawn as a stroke with stroke_width_px; the face interior is removed."
        ├── frame_indices           int64   [N]
        ├── region_names            bytes   [N]
        ├── region_roles            bytes   [N]
        ├── stroke_width_px         int32   [N]
        ├── polygon_offsets         uint64  [N + 1]
        └── polygon_points          int32   [total_de_pontos_padrão, 2]
```

Exemplos ilustrativos dos valores:

```text
landmarks[0, 0, :] = [0.52, 0.31, -0.04]  # x, y e z normalizados
face_detected[:4] = [true, false, true, false]
source_frame[:4] = [0, 0, 2, 2]

blur_regions/frame_ranges[:2] = [[12, 35], [40, 55]]  # início e fim inclusivos
blur_regions/polygon_offsets[:3] = [0, 4, 8]
blur_regions/polygon_points[:8] =
  [[10, 10], [30, 10], [30, 30], [10, 30],
   [50, 50], [70, 50], [70, 70], [50, 70]]

blur_regions/face_blur/frame_indices[:7] =
  [0, 0, 0, 0, 0, 0, 1]
blur_regions/face_blur/region_names[:7] =
  [left_eye_and_eyebrow, right_eye_and_eyebrow, nose, mouth,
   outside_face_ring, lower_face_exclusion]
blur_regions/face_blur/region_roles[:7] =
  [include, include, include, include, include, exclude, include]
blur_regions/face_blur/stroke_width_px[:7] = [0, 0, 0, 0, 18, 0, 0]
blur_regions/face_blur/polygon_offsets[:8] = [0, 4, 8, 11, 15, 20, 24, 28]
blur_regions/face_blur/polygon_points[:4] = [[210, 160], [220, 150], [230, 160], [220, 170]]
```

Em cada grupo, `polygon_offsets[i:i+2]` seleciona os pontos do polígono `i` em
`polygon_points`. Para `face_blur`, o registro `i` também usa
`frame_indices[i]`, `region_names[i]`, `region_roles[i]` e
`stroke_width_px[i]`. Polígonos `include` formam a máscara; `exclude` remove
pixels dela. `outside_face_ring` é desenhado como contorno, usando sua espessura,
e não como um polígono preenchido. O
`read_landmarks.py --show-blur-regions`
destaca tanto os polígonos manuais quanto esses polígonos padrão.
Se for executado apenas `save_std_blur_regions.py`, o HDF5 pode conter somente
`blur_regions/face_blur`; os arrays e atributos das regiões manuais são criados
quando o aplicativo exporta as regiões manuais.

Para assistir ao vídeo anonimizado e destacar em vermelho os pixels cobertos pelas
regiões salvas, use o leitor de landmarks:

```powershell
python face_detection\read_landmarks.py "C:\dados\video_borrado.avi" --show-blur-regions --landmarks "C:\dados\video_landmarks.h5"
```

O vídeo anonimizado e o HDF5 bastam para consultar as regiões; o vídeo original
não é necessário. Se o HDF5 não tiver o mesmo nome-base do vídeo, informe-o com
`--landmarks`.

Para preencher os polígonos padrão do processador em HDF5s antigos, sem abrir ou
decodificar os vídeos, use:

```powershell
python video_anonymization\save_std_blur_regions.py "C:\dados\video.avi"
python video_anonymization\save_std_blur_regions.py --dataset "C:\dados\dataset"
```

O comando de dataset percorre recursivamente as subpastas. Os HDF5s devem conter
`landmarks`, `frame_width` e `frame_height`; para reconstruir polígonos que ainda
não foram salvos, o ambiente também precisa do MediaPipe usado pelo processador.
Os polígonos padrão são armazenados em `blur_regions/face_blur`. Se estiverem
no antigo grupo `blur_regions/process_video` ou no grupo raiz
`process_video_blur_regions`, o utilitário os move para esse local.
Como as coordenadas armazenadas são `float32`, a conversão para pixels pode
arredondar pontos raros em relação à máscara original.

O codec I420 não comprime o vídeo. Assim, vídeos de origem AVI já sem compressão
podem ter tamanho final semelhante ao original. Vídeos de origem comprimidos (como
MP4/H.264) podem gerar arquivos muito maiores; isso é esperado ao salvar sem
compressão. O formato YUV 4:2:0 também reduz a resolução de cor.
