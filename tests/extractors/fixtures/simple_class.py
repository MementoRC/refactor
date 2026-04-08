"""A simple module for testing."""

CONSTANT = 42
logger = None


class Animal:
    """Base animal class."""

    species: str = "unknown"

    def __init__(self, name: str, age: int = 0):
        """Initialize animal.

        :param name: The animal's name
        :param age: The animal's age in years
        """
        self.name = name
        self.age = age

    def speak(self) -> str:
        """Make the animal speak."""
        return f"{self.name} says hello"

    @property
    def is_adult(self) -> bool:
        """Whether the animal is adult."""
        return self.age >= 1


class Dog(Animal):
    """A dog."""

    def speak(self) -> str:
        return f"{self.name} says woof"
