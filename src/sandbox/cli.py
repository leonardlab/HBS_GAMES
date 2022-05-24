import click
from sandbox.fibonacci import Fibonacci


@click.command()
@click.argument("number", type=int)
def cli(number: int) -> None:
    """CLI entrypoint for Fibonacci."""
    print(Fibonacci.get_values(number))
    print(Fibonacci.calculate(number))


# pylint: disable=no-value-for-parameter

if __name__ == "__main__":
    cli()
