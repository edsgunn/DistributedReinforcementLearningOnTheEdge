import random
from typing import List, Tuple
from Environment import Environment
import Enum

class Suit(Enum):
    Diamonds = 1
    Clubs = 2
    Hearts = 3
    Spades = 4

class CardNumber(Enum):
    Ace = 1
    Two = 2
    Three = 3
    Four = 4 
    Five = 5
    Six = 6
    Seven = 7
    Eight = 8
    Nine = 9
    Ten = 10
    Jack = 11
    Queen = 12
    King = 13

class Card:
    
    def __init__(self, number: CardNumber, suit: Suit):
        self.number = number
        self.suit = suit

    def getNumber(self) -> CardNumber:
        return self.number

    def getSuit(self) -> Suit:
        return self.suit

    def getCard(self) -> Tuple[CardNumber, Suit]:
        return (self.number, self.suit)

class DeckOfCards:

    def __init__(self):
        self.deck: List[Card] = []
        for suit in Suit:
            for number in CardNumber:
                card = Card(number, suit)
                self.deck.append(card)
        random.shuffle(self.deck)

    def draw(self) -> Card:
        return self.deck.pop()

class BlackJack(Environment):
    
    def __init__(self) -> None:
        self.deck = DeckOfCards()
        self.playersHand: List[Card] = [self.deck.draw() for _ in range(2)]
        self.dealersHand: List[Card] = [self.deck.draw() for _ in range(2)]

    def handToState(hand: List[Card]) -> Tuple[int, int]:
        cardTotal = 0
        numberOfAces = 0
        for card in hand:
            number = card.getNumber()
            if card.getNumber == CardNumber.Ace:
                numberOfAces += 1
            else:
                cardTotal += number.value

        for _ in range(numberOfAces):
            if 21-cardTotal > 10:
                cardTotal += 11
            else:
                cardTotal += 1

        return (cardTotal, numberOfAces)

    def getObservableState(self):
        return self.handToState(self.playersHand)

        