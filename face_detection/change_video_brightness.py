from pathlib import Path

import cv2


def change_video_brightness(input_path, output_path=None, brightness=50):
    """Brighten a video and save it as an AVI encoded with I420 (YUV 4:2:0)."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Vídeo de entrada não encontrado: {input_path}")

    if output_path is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_brightened.avi"
        )
    else:
        output_path = Path(output_path)

    if output_path.suffix.lower() != ".avi":
        raise ValueError("O caminho de saída deve ter a extensão .avi")
    if output_path.resolve() == input_path.resolve():
        raise ValueError("O caminho de saída deve ser diferente do vídeo de entrada")
    if brightness < 0:
        raise ValueError("O valor de brilho deve ser maior ou igual a zero")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"Propriedades inválidas no vídeo: {input_path}")
    if width % 2 or height % 2:
        capture.release()
        raise ValueError(
            "O codec I420 exige largura e altura pares; "
            f"vídeo recebido: {width}x{height}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"I420")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"Não foi possível criar o vídeo de saída I420: {output_path}"
        )

    frame_count = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            brightened_frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness)
            writer.write(brightened_frame)
            frame_count += 1
    finally:
        capture.release()
        writer.release()

    if frame_count == 0:
        raise RuntimeError(f"Nenhum frame foi lido do vídeo: {input_path}")

    print(f"Vídeo clareado salvo em: {output_path}")
    return output_path


if __name__ == "__main__":
    video_path = input("Caminho do vídeo: ").strip()
    change_video_brightness(video_path)