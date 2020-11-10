from ext import ArgumentParser


def test_parser():
    expected = [
        dict(
            input=dict(arg_1='value 1'),
            output='s',
        )
    ]
    parser = ArgumentParser()
    parser.add_argument('--seed', default=None, type=int)
    data = dict(seed='23')
    args = parser.parse_kwargs(data)
    print(args)
    args = parser.parse_kwargs({})
    print(args)
