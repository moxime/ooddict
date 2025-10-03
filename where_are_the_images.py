import os
import tarfile
import os.path as p
import logging

source_files = ['/tmp/train', '/tmp/train_3']
target_file = './data/benchmark_imglist/imagenet/train_imagenet.txt'


available_nodes = {}

for source_file in source_files:
    with open(source_file) as f:
        for line in f:
            node = p.splitext(line)[0]
            if node in available_nodes:
                logging.info('{} in {} already in from {}'.format(node,
                                                                  source_file,
                                                                  *available_nodes[node]))
                available_nodes[node].append(source_file)
            else:
                available_nodes[node] = [source_file]

node_by_files = {tuple(_): [n for n in available_nodes if available_nodes[n] == _]
                 for _ in available_nodes.values()}


for _ in node_by_files:
    print(*_, len(node_by_files[_]))

needed_nodes = {}

with open(target_file) as f:

    for line in f:
        path = line.split()[0]
        image_name = p.basename(path)
        node = image_name.split('_')[0]

        if node not in available_nodes:
            logging.error('{} not available'.format(node))
            continue

        needed_nodes[node] = available_nodes[node]


assert all('/tmp/train' in _ for _ in needed_nodes.values())


def organize_inet_nodes(root='/mnt/Data/ImageNet/train', dry_run=True):

    jpeg_files = [_ for _ in os.listdir(root) if _.lower.endswith('.jpeg')]
    logging.info('Found {} JPEGs'.format(len(jpeg_files)))

    for f in jpeg_files:

        node, image = f.plit('_')
        node_path = p.join(root, node)
        origin_path = p.join(root, f)
        target_path = p.join(node_path, image)

        logging.info('{} -> {}'.format(origin_path, target_path))
        if dry_run:
            continue
        if p.exists(node_path):
            logging.info('{} already exists'.format(node))
        else:
            os.mkdir(node_path)

        os.rename(origin_path, target_path)
