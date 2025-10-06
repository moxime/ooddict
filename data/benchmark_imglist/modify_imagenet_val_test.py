import os
from os import path


def remove_nodes(list_file, dry_run=True):

    backup = list_file + '.bak'

    if not path.exists(backup):
        os.rename(list_file, backup)

    with open(list_file, 'w') as targetf:

        with open(backup) as sourcef:

            for line in sourcef:
                image_path, *other = line.split()
                image_node, image_name = path.split(image_path)
                image_dir = path.split(image_node)[0]

                new_image_path = path.join(image_dir, image_name)

                if dry_run:
                    print(image_path, '->', new_image_path)
                    continue

                new_line = ' '.join([new_image_path, *other, '\n'])

                targetf.write(new_line)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('-x', action='store_true')

    args = parser.parse_args()

    remove_nodes(args.file, dry_run=not args.x)
