from argparse import ArgumentParser, Namespace


class Parser(ArgumentParser):
    def error(self, message):
        raise Exception(str(message).replace('--', ''))

    def parse_kwargs(self, kwargs: dict) -> Namespace:
        # Flatten dict to list
        args = []
        for k, v in kwargs.items():
            args.append(f'--{k}')
            args.append(str(v))
        return super().parse_args(args)
