import random
from typing import List, Tuple
from SingleAgentTests.Environment import Environment
from enum import Enum
from Common.Types import Action, ActionSet, State

class Suit(Enum):
    Diamonds = 0
    Clubs = 1
    Hearts = 2
    Spades = 3

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
    Jack = 10
    Queen = 10
    King = 10

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
        self.possibleActions: ActionSet = [0,1]
        self.gameOver = False
        self.suitTranslation = ["D", "C", "H", "S"]

    def handToState(self, hand: List[Card]) -> Tuple[int, int]:
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

    def getObservableState(self) -> State:
        return self.handToState(self.playersHand)

    def getPossibleActions(self) -> ActionSet:
        return self.possibleActions

    def getAllPossibleStateActions(self) -> List[Tuple(State, Action)]:
        return [((cardTotal, aces), action) for cardTotal in range(30) for aces in range(5) for action in range(2)] 

    def step(self, action: Action) -> bool:
        if action == 0:
            self.gameOver = True
        elif action == 1:
            self.playersHand.append(self.deck.draw())
            if self.handToState(self.playersHand)[0] > 21:
                self.gameOver = True

        if self.gameOver:
            while self.handToState(self.dealersHand)[0] < 17:
                self.dealersHand.append(self.deck.draw())
            return False
        
        return True

    def getReward(self) -> float:
        if self.gameOver:
            dealerTotal = self.handToState(self.dealersHand)[0]
            playerTotal = self.handToState(self.playersHand)[0]
            print(f"Game Over\nPlayer: {playerTotal}, Dealer: {dealerTotal}")
            if playerTotal > 21:
                if dealerTotal > 21:
                    print("Player and dealer Bust")
                    return 0
                else:
                    print("Player bust")
                    return -1
            else:
                if dealerTotal > 21 or playerTotal > dealerTotal:
                    print("Player wins")
                    return 1
                elif playerTotal == dealerTotal:
                    print("Player and dealer equal")
                    return 0
                else:
                    print("Player loses")
                    return -1
        return 0

    def printHand(self, hand: List[Card]):
        text = ""
        for card in hand:
            text += str(card.getNumber().value)
            text += self.suitTranslation[card.getSuit().value]
            text += "\n"

        print(text)

                