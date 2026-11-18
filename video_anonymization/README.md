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
Se não houver um HDF5 compatível ou houver vários candidatos sem correspondência
pelo nome do vídeo, a exportação é interrompida para evitar editar o arquivo errado.

Para assistir ao vídeo anonimizado e destacar em vermelho os pixels cobertos pelas
regiões salvas, use o leitor de landmarks:

```powershell
python face_detection\read_landmarks.py "C:\dados\video_borrado.avi" --show-blur-regions --landmarks "C:\dados\video_landmarks.h5"
```

O vídeo anonimizado e o HDF5 bastam para consultar as regiões; o vídeo original
não é necessário. Se o HDF5 não tiver o mesmo nome-base do vídeo, informe-o com
`--landmarks`.

O codec I420 não comprime o vídeo. Assim, vídeos de origem AVI já sem compressão
podem ter tamanho final semelhante ao original. Vídeos de origem comprimidos (como
MP4/H.264) podem gerar arquivos muito maiores; isso é esperado ao salvar sem
compressão. O formato YUV 4:2:0 também reduz a resolução de cor.
