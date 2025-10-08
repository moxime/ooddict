from openood.pipelines import get_pipeline
from openood.utils import launch, setup_config


def main(config):

    pipeline = get_pipeline(config)
    pipeline.run()


if __name__ == '__main__':

    import sys
    print('Path', sys.path)
    print('forceantialias in sys.modules?', 'forceantialias' in sys.modules)
    try:
        import forceantialias
        print('Imported from:', forceantialias.__file__)
    except Exception as e:
        print('Import raised:', repr(e))

    config = setup_config()

    launch(
        main,
        config.num_gpus,
        num_machines=config.num_machines,
        machine_rank=config.machine_rank,
        dist_url='auto',
        args=(config, ),
    )
