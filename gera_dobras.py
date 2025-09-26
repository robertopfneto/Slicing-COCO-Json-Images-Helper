# Baseado no código de Artur Karaźniewicz
# Disponível aqui: https://github.com/akarazniewicz/cocosplit
#
# O número de folds padrão é 5 
# O percentual a ser usado para validação é 0.3 (30%)
# Os arquivos .json resultantes são salvos na pasta ./dataset/all/filesJSON
#
# Para mudar estes valores basta passar valores diferentes como parâmetros

import argparse
import json
import math
import os
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import funcy

parser = argparse.ArgumentParser(
    description='Divide o conjunto de anotações para permitir aplicação de validação cruzada em dobras'
)
parser.add_argument(
    '-annotations',
    default='./dataset/all/train/_annotations.coco.json',
    metavar='coco_annotations',
    type=str,
    help='Caminho para o arquivo com as anotações',
    required=False,
)
parser.add_argument(
    '-json',
    default='./dataset/all/filesJSON/',
    type=str,
    help='Pasta para os arquivos resultantes',
    required=False,
)
parser.add_argument(
    '-folds',
    default='5',
    dest='folds',
    type=int,
    help='Número de dobras a ser usado',
    required=False,
)
parser.add_argument(
    '-valperc',
    default='0.3',
    dest='valperc',
    type=float,
    help='Percentual a ser usado para validação durante o treinamento',
    required=False,
)
parser.add_argument(
    '--having-annotations',
    dest='having_annotations',
    action='store_true',
    help='Ignora imagens que não tenham nenhuma anotação',
)
parser.add_argument(
    '--seed',
    dest='seed',
    type=int,
    default=42,
    help='Semente para reprodutibilidade dos sorteios',
)

args = parser.parse_args()

NEGATIVE_KEEP_FRACTION = 0.10


def save_coco(file, info, licenses, images, annotations, categories):
    sanitized_images = []
    for image in images:
        item = deepcopy(image)
        file_name = item.get('file_name', '')
        item['file_name'] = os.path.basename(file_name)
        sanitized_images.append(item)

    payload = {
        'info': info,
        'licenses': licenses,
        'images': sanitized_images,
        'annotations': annotations,
        'categories': categories,
    }

    Path(file).parent.mkdir(parents=True, exist_ok=True)
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump(payload, coco, indent=2, sort_keys=True)


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def build_annotation_lookup(annotations: List[Dict]) -> Dict[int, List[Dict]]:
    lookup: Dict[int, List[Dict]] = defaultdict(list)
    for annotation in annotations:
        lookup[int(annotation['image_id'])].append(annotation)
    return lookup


def get_group_id(image: Dict) -> str:
    file_name = image.get('file_name', '')
    stem = Path(file_name).stem
    if '_tile_' in stem:
        return stem.split('_tile_')[0]
    return stem


def build_images_by_group(images: List[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for image in images:
        groups[get_group_id(image)].append(image)
    return dict(groups)


def split_group_ids(groups: List[str], fraction: float, rng: random.Random) -> Tuple[List[str], List[str]]:
    if not groups or fraction <= 0:
        return groups.copy(), []
    if len(groups) <= 1:
        return groups.copy(), []

    shuffled = groups.copy()
    rng.shuffle(shuffled)

    split_count = math.ceil(len(shuffled) * fraction)
    split_count = max(1, split_count)
    if split_count >= len(shuffled):
        split_count = len(shuffled) - 1

    val_groups = shuffled[:split_count]
    train_groups = shuffled[split_count:]
    return train_groups, val_groups


def expand_groups(group_ids: List[str], images_by_group: Dict[str, List[Dict]]) -> List[Dict]:
    images: List[Dict] = []
    for gid in group_ids:
        images.extend(images_by_group[gid])
    return images


def select_training_subset(
    images: List[Dict],
    annotations_lookup: Dict[int, List[Dict]],
    rng: random.Random,
    keep_fraction: float = NEGATIVE_KEEP_FRACTION,
) -> Tuple[List[Dict], Dict[str, int]]:
    positives = [img for img in images if annotations_lookup.get(int(img['id']))]
    negatives = [img for img in images if not annotations_lookup.get(int(img['id']))]

    if not negatives:
        selected_negatives: List[Dict] = []
    else:
        keep_count = math.ceil(len(negatives) * keep_fraction)
        keep_count = max(1, keep_count)
        keep_count = min(keep_count, len(negatives))
        selected_negatives = rng.sample(negatives, keep_count)

    selected = positives + selected_negatives
    rng.shuffle(selected)

    stats = {
        'positives': len(positives),
        'negatives_total': len(negatives),
        'negatives_kept': len(selected_negatives),
    }

    return selected, stats


def build_folds(group_ids: List[str], folds: int, seed: int) -> List[List[str]]:
    if len(group_ids) < folds:
        raise ValueError('Número de dobras maior que o número de imagens originais disponíveis.')

    rng = random.Random(seed)
    shuffled = group_ids.copy()
    rng.shuffle(shuffled)

    fold_sizes = [len(shuffled) // folds] * folds
    for i in range(len(shuffled) % folds):
        fold_sizes[i] += 1

    fold_groups: List[List[str]] = []
    idx = 0
    for size in fold_sizes:
        fold_groups.append(shuffled[idx: idx + size])
        idx += size

    return fold_groups


def prepare_dataset(args):
    with open(args.annotations, 'rt', encoding='UTF-8') as annotations_file:
        coco = json.load(annotations_file)

    info = coco['info']
    licenses = coco['licenses']
    images = coco['images']
    annotations = coco['annotations']
    categories = coco['categories']

    annotations_lookup = build_annotation_lookup(annotations)
    images_by_group = build_images_by_group(images)

    if args.having_annotations:
        images_by_group = {
            gid: group_images
            for gid, group_images in images_by_group.items()
            if any(annotations_lookup.get(int(img['id'])) for img in group_images)
        }

    if not images_by_group:
        raise ValueError('Nenhuma imagem disponível após aplicar os filtros solicitados.')

    return info, licenses, annotations, categories, annotations_lookup, images_by_group


def prepare_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for existing in path.glob('*'):
        if existing.is_file():
            existing.unlink()


def geraUma(args):
    info, licenses, annotations, categories, annotations_lookup, images_by_group = prepare_dataset(args)

    output_dir = Path(args.json)
    prepare_output_dir(output_dir)

    group_ids = list(images_by_group.keys())
    rng_seed = args.seed

    train_groups, test_groups = split_group_ids(group_ids, 0.2, random.Random(rng_seed))
    train_groups, val_groups = split_group_ids(train_groups, args.valperc, random.Random(rng_seed + 1))

    train_images = expand_groups(train_groups, images_by_group)
    val_images = expand_groups(val_groups, images_by_group)
    test_images = expand_groups(test_groups, images_by_group)

    train_subset, train_stats = select_training_subset(train_images, annotations_lookup, random.Random(rng_seed + 2))

    save_coco(
        str(output_dir / 'fold_1_train.json'),
        info,
        licenses,
        train_subset,
        filter_annotations(annotations, train_subset),
        categories,
    )
    save_coco(
        str(output_dir / 'fold_1_val.json'),
        info,
        licenses,
        val_images,
        filter_annotations(annotations, val_images),
        categories,
    )
    save_coco(
        str(output_dir / 'fold_1_test.json'),
        info,
        licenses,
        test_images,
        filter_annotations(annotations, test_images),
        categories,
    )

    print(
        "Salvou {} tiles em {} ({} grupos; {} com anotação, {} de {} negativos)".format(
            len(train_subset),
            'fold_1_train.json',
            len(train_groups),
            train_stats['positives'],
            train_stats['negatives_kept'],
            train_stats['negatives_total'],
        )
    )
    print("Salvou {} tiles em {} ({} grupos)".format(len(val_images), 'fold_1_val.json', len(val_groups)))
    print("Salvou {} tiles em {} ({} grupos)".format(len(test_images), 'fold_1_test.json', len(test_groups)))


def main(args):
    if args.folds == 1:
        geraUma(args)
        return

    info, licenses, annotations, categories, annotations_lookup, images_by_group = prepare_dataset(args)

    output_dir = Path(args.json)
    prepare_output_dir(output_dir)

    group_ids = list(images_by_group.keys())
    total_groups = len(group_ids)
    total_tiles = sum(len(group) for group in images_by_group.values())

    print(f'Total de imagens originais (grupos): {total_groups}')
    print(f'Total de tiles: {total_tiles}')

    fold_groups = build_folds(group_ids, args.folds, args.seed)
    print('Distribuição de grupos por dobra:', [len(fold) for fold in fold_groups])

    for i, test_groups in enumerate(fold_groups):
        print('---------------------------')
        print('Processando Dobra ', i + 1)

        other_groups = [gid for j, fold in enumerate(fold_groups) if j != i for gid in fold]
        train_groups, val_groups = split_group_ids(other_groups, args.valperc, random.Random(args.seed + i + 1))

        train_images = expand_groups(train_groups, images_by_group)
        val_images = expand_groups(val_groups, images_by_group)
        test_images = expand_groups(test_groups, images_by_group)

        train_subset, train_stats = select_training_subset(train_images, annotations_lookup, random.Random(args.seed + i + 101))

        arq_treino = output_dir / f'fold_{i + 1}_train.json'
        arq_val = output_dir / f'fold_{i + 1}_val.json'
        arq_teste = output_dir / f'fold_{i + 1}_test.json'

        save_coco(
            str(arq_treino),
            info,
            licenses,
            train_subset,
            filter_annotations(annotations, train_subset),
            categories,
        )
        save_coco(
            str(arq_val),
            info,
            licenses,
            val_images,
            filter_annotations(annotations, val_images),
            categories,
        )
        save_coco(
            str(arq_teste),
            info,
            licenses,
            test_images,
            filter_annotations(annotations, test_images),
            categories,
        )

        print(
            "Salvou {} tiles em {} ({} grupos; {} com anotação, {} de {} negativos)".format(
                len(train_subset),
                arq_treino.name,
                len(train_groups),
                train_stats['positives'],
                train_stats['negatives_kept'],
                train_stats['negatives_total'],
            )
        )
        print(
            "Salvou {} tiles em {} ({} grupos)".format(
                len(val_images),
                arq_val.name,
                len(val_groups),
            )
        )
        print(
            "Salvou {} tiles em {} ({} grupos)".format(
                len(test_images),
                arq_teste.name,
                len(test_groups),
            )
        )


if __name__ == "__main__":
    main(args)
