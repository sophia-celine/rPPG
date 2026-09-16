import argparse
import shutil
from pathlib import Path
import subprocess
import sys


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def main():

    parser = argparse.ArgumentParser(
        description="Face restoration with CodeFormer."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Pasta contendo as imagens originais/anônimas."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Pasta onde as imagens restauradas serão salvas."
    )

    parser.add_argument(
        "--w",
        type=float,
        default=0.7,
        help="Fidelity weight do CodeFormer. Default: 0.7"
    )

    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    # Diretório temporário onde o CodeFormer irá gerar os resultados
    temp_dir = output_dir / "_codeformer_temp"

    # Verificações
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Pasta de entrada não encontrada:\n{input_dir}"
        )

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"O caminho de entrada não é uma pasta:\n{input_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Cria diretório temporário limpo
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(parents=True)

    # Localização do inference_codeformer.py
    codeformer_dir = Path(__file__).resolve().parent
    codeformer_script = codeformer_dir / "inference_codeformer.py"

    if not codeformer_script.exists():
        raise FileNotFoundError(
            "Não encontrei inference_codeformer.py.\n"
            f"Esperado em:\n{codeformer_script}"
        )

    print("=" * 70)
    print("CODEFORMER - FACE RESTORATION")
    print("=" * 70)

    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"w     : {args.w}")
    print()

    # ------------------------------------------------------------
    # Verifica CUDA
    # ------------------------------------------------------------

    print("Verificando CUDA...")

    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print()
        print("AVISO: CUDA não está disponível.")
        print("O CodeFormer poderá executar em CPU e ficar muito lento.")
        print()

    print("=" * 70)
    print()

    # ------------------------------------------------------------
    # Executa CodeFormer
    # ------------------------------------------------------------

    command = [
        sys.executable,
        str(codeformer_script),

        "-w",
        str(args.w),

        "--input_path",
        str(input_dir),

        "--output_path",
        str(temp_dir),
    ]

    print("Executando CodeFormer...")
    print()

    result = subprocess.run(command)

    if result.returncode != 0:
        raise RuntimeError(
            "\nO CodeFormer terminou com erro.\n"
            f"Return code: {result.returncode}"
        )

    # ------------------------------------------------------------
    # Localiza os resultados
    # ------------------------------------------------------------

    restored_dir = temp_dir / "final_results"

    if not restored_dir.exists():
        raise FileNotFoundError(
            "O CodeFormer terminou, mas a pasta final_results "
            "não foi encontrada:\n"
            f"{restored_dir}"
        )

    restored_files = list(restored_dir.glob("*.png"))

    if len(restored_files) == 0:
        raise RuntimeError(
            "Nenhuma imagem restaurada foi encontrada."
        )

    # ------------------------------------------------------------
    # Mapeia os arquivos originais
    # ------------------------------------------------------------

    original_files = sorted(
        [
            f
            for f in input_dir.iterdir()
            if f.is_file() and f.suffix in IMAGE_EXTENSIONS
        ]
    )

    print()
    print("=" * 70)
    print("ORGANIZANDO RESULTADOS")
    print("=" * 70)

    processed = 0
    missing = 0

    # ------------------------------------------------------------
    # Salva com os mesmos nomes dos arquivos originais
    # ------------------------------------------------------------

    for original_file in original_files:

        stem = original_file.stem

        # CodeFormer salva o resultado como:
        #
        # original_name.png
        #
        restored_file = restored_dir / f"{stem}.png"

        destination = output_dir / original_file.name

        if not restored_file.exists():

            print(
                f"[NÃO ENCONTRADO] {original_file.name}"
            )

            missing += 1
            continue

        # --------------------------------------------------------
        # Se a extensão original for PNG:
        # simplesmente copia o PNG.
        # --------------------------------------------------------

        if original_file.suffix.lower() == ".png":

            shutil.copy2(
                restored_file,
                destination
            )

        # --------------------------------------------------------
        # Se for JPG/JPEG:
        # converte PNG -> JPG para preservar o nome/extensão.
        # --------------------------------------------------------

        else:

            import cv2

            image = cv2.imread(
                str(restored_file),
                cv2.IMREAD_COLOR
            )

            if image is None:
                print(
                    f"[ERRO] Não foi possível ler "
                    f"{restored_file}"
                )
                missing += 1
                continue

            success = cv2.imwrite(
                str(destination),
                image,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

            if not success:
                print(
                    f"[ERRO] Não foi possível salvar "
                    f"{destination}"
                )
                missing += 1
                continue

        print(
            f"[OK] {original_file.name}"
        )

        processed += 1

    # ------------------------------------------------------------
    # Remove resultados temporários
    # ------------------------------------------------------------

    shutil.rmtree(temp_dir)

    # ------------------------------------------------------------
    # Resumo
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINALIZADO")
    print("=" * 70)

    print(f"Imagens encontradas : {len(original_files)}")
    print(f"Imagens processadas  : {processed}")
    print(f"Não processadas      : {missing}")
    print()
    print(f"Resultados salvos em:")
    print(output_dir)


if __name__ == "__main__":
    main()