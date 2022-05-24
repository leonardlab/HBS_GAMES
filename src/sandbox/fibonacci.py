from typing import List


class Fibonacci:
    """Fibonacci sequence methods."""

    @staticmethod
    def calculate(nth: int) -> int:
        """Calculate the n-th Fibonacci number.

        An extended description of Fibonacci numbers can go here.

        Parameters
        ----------
        nth
            Number of Fibonacci sequence to calculate.

        Returns
        -------
        int
            Value of n-th Fibonacci number.
        """

        num_a, num_b = 0, 1

        for _ in range(nth):
            num_a, num_b = num_b, num_a + num_b

        return num_a

    @staticmethod
    def get_values(nth: int) -> List:
        """Returns the Fibonacci sequence up until n-th number.

        Parameters
        ----------
        nth
            Number of Fibonacci sequence to calculate.

        Returns
        -------
        list
            Fibonacci numbers.
        """

        return [Fibonacci.calculate(i) for i in range(1, nth + 1)]
