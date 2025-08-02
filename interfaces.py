
"""

simple functions and classes to interface with 
not-so-well documented library results

"""

import statistics
from typing import Iterable, NamedTuple, Self

import easyocr


class Point(NamedTuple):
    """classic numeric 2D coordinate"""
    x: float
    y: float

    def is_close_to(self, other: Self, threshold: int) -> bool:
        cond1 = abs(self.x - other.x)
        cond2 = abs(self.y - other.y)
        return (cond1 <= threshold) and (cond2 <= threshold)


class Box(NamedTuple):
    """basic bounding box"""
    p1: Point
    p2: Point
    p3: Point
    p4: Point

    @classmethod
    def from_tuple(cls, t: tuple[Point, Point, Point, Point]) -> Self:
        return cls(*t)

    def mean(self) -> Point:
        """returns the point in the middle of the box"""
        # NOTE: maybe rename to "middle ?"
        t = tuple(self)
        mean_x = statistics.mean(p.x for p in t)
        mean_y = statistics.mean(p.y for p in t)
        return Point(mean_x, mean_y)

    @classmethod
    def from_points(cls, points: list[Point]) -> Self:
        """ from a set of points, get the box that surrounds them all """
        min_x = min(p.x for p in points)
        min_y = min(p.y for p in points)
        max_x = max(p.x for p in points)
        max_y = max(p.y for p in points)
        return cls.from_minmax(min_x, min_y, max_x, max_y)

    @classmethod
    def from_minmax(cls, min_x: float, min_y: float, max_x: float, max_y: float) -> Self:
        return cls(
            Point(min_x, min_y),
            Point(max_x, min_y),
            Point(min_x, max_y),
            Point(max_x, max_y)
        )


class ReadText(NamedTuple):
    """
    custom type for what easyocr.Reader().readtext() returns

    ## SEE
    https://deepwiki.com/JaidedAI/EasyOCR/3-basic-usage
    """
    box: Box
    text: str
    confidence: float


def read_page(reader: easyocr.Reader, image: bytes) -> list[ReadText]:
    """ this only adds type hints and freezes the "detail" param """
    # results = [
    #     (
    #         [[700, 288], [748, 288], [748, 346], [700, 346]], # 4 corners of the box, in a list
    #         '2',
    #         0.35551235377397816),
    #     ... # you get a list of these
    # ]
    return reader.readtext(image, detail=1)


class FoundText(NamedTuple):
    coordinates: Box
    text: str
    line_count: int


class FormattedText(NamedTuple):
    coordinates: Box
    text: str
    formatted: str




