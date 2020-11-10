import argparse


class ArgumentParser(argparse.ArgumentParser):
    def _print_message(self, message, file=None):
        raise Exception(message)

    def exit(self, *args, **kwargs):
        return

    def error(self, message):
        raise Exception(str(message).replace('--', ''))

    def parse_kwargs(self, kwargs: dict) -> argparse.Namespace:
        # Flatten dict to list
        args = []
        for k, v in kwargs.items():
            args.append(f'--{k}')
            args.append(str(v))
        return super().parse_args(args)
