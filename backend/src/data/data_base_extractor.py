from abc import ABC, abstractmethod
from typing import Union
import pycountry

from .data_pydentic import MarketData, MacroData, NewsData

FETCHER_RESULT_TYPE =   Union[MarketData, MacroData, NewsData]

class BaseFetcher(ABC):
    """
    Base class for data fetchers.
    Accepts either a ticker string or a pycountry country object.
    Returns one of the defined pydantic response models.
    """

    def __init__(self, ticker: str | None = None, country: pycountry.db.Country | None = None):
        if ticker is None and country is None:
            raise ValueError("Provide either a ticker or a country.")
        if ticker is not None and country is not None:
            raise ValueError("Provide either a ticker or a country, not both.")

        self.ticker = ticker
        self.country = country

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def fetch(self) -> FETCHER_RESULT_TYPE:
        """abstract method"""
        pass