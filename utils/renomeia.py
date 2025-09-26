"""Utility script to normalize image filenames and COCO annotations.

This mirrors the CLI logic under `dataset/all/renomeia.py` but defaults to the
repository-level dataset path (`dataset/all/train`). Use it when you want to
rename images and/or update `_annotations.coco.json` from the project root.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Normaliza nomes de arquivos e anotações para apenas números.")
    parser.add_argument("--images", action="store_true", help="Renomeia os arquivos de imagem.")
    parser.add_argument("--annotations", action="store_true", help="Atualiza o arquivo COCO.")
    parser.add_argument(
        "--dataset-dir",
        default="./dataset/all/train",
        help="Diretório onde estão as imagens (padrão: dataset/all/train).",
    )
    parser.add_argument(
        "--annotations-file",
        default="_annotations.coco.json",
        help="Nome do arquivo COCO (padrão: _annotations.coco.json).",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    annotations_path = Path(args.annotations_file)
    if not annotations_path.is_absolute():
        annotations_path = dataset_dir / annotations_path

    if not args.images and not args.annotations:
        args.images = args.annotations = True

    if args.images:
        rename_images_to_numeric(dataset_dir)

    if args.annotations:
        normalize_annotations_to_numeric(annotations_path)


def rename_images_to_numeric(dataset_dir: Path) -> Dict[str, str]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {dataset_dir}")

    image_suffixes = {".jpg", ".jpeg", ".png"}
    number_pattern = re.compile(r"^(\d+)")

    rename_candidates = []
    new_name_counts: Dict[str, int] = {}
    rename_map: Dict[str, str] = {}

    for image_path in dataset_dir.iterdir():
        if not image_path.is_file():
            continue

        suffix = image_path.suffix.lower()
        if suffix not in image_suffixes:
            continue

        match = number_pattern.match(image_path.name)
        if not match:
            print(f"Ignorando {image_path.name}: não começa com número")
            continue

        new_name = f"{match.group(1)}{suffix}"

        if new_name == image_path.name:
            continue

        rename_candidates.append((image_path, new_name))
        new_name_counts[new_name] = new_name_counts.get(new_name, 0) + 1

    for image_path, new_name in rename_candidates:
        old_name = image_path.name

        if new_name_counts[new_name] > 1:
            print(f"Ignorando {old_name}: mais de um arquivo geraria {new_name}")
            continue

        target_path = dataset_dir / new_name
        if target_path.exists() and target_path != image_path:
            print(f"Ignorando {old_name}: destino {new_name} já existe")
            continue

        image_path.rename(target_path)
        rename_map[old_name] = new_name
        print(f"Renomeado: {old_name} -> {new_name}")

    if not rename_map:
        print("Nenhuma imagem renomeada.")

    return rename_map


def normalize_annotations_to_numeric(annotations_path: Path) -> None:
    if not annotations_path.exists():
        raise FileNotFoundError(f"Arquivo de anotações não encontrado: {annotations_path}")

    number_pattern = re.compile(r"^(\d+)")

    with annotations_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    images = data.get("images", [])
    updates = 0

    for image in images:
        file_name = image.get("file_name")
        if not file_name:
            continue

        file_path = Path(file_name)
        match = number_pattern.match(file_path.name)
        if not match:
            print(f"Ignorando {file_name}: não começa com número")
            continue

        suffix = file_path.suffix.lower() or ".jpg"
        new_name = f"{match.group(1)}{suffix}"

        if file_path.name == new_name:
            continue

        if file_path.parent == Path("."):
            image["file_name"] = new_name
        else:
            image["file_name"] = str(file_path.with_name(new_name))
        updates += 1

    if updates:
        with annotations_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        print(f"Atualizadas {updates} entradas em {annotations_path.name}")
    else:
        print("Nenhuma entrada de anotação precisou ser atualizada.")


if __name__ == "__main__":
    main()
