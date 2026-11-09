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

O codec I420 não comprime o vídeo. Assim, vídeos de origem AVI já sem compressão
podem ter tamanho final semelhante ao original. Vídeos de origem comprimidos (como
MP4/H.264) podem gerar arquivos muito maiores; isso é esperado ao salvar sem
compressão. O formato YUV 4:2:0 também reduz a resolução de cor.
