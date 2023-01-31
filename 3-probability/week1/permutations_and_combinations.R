library(gtools)

# construct a deck of cards
suits <- c('Diamonds', 'Clubs', 'Hearts', 'Spades')
numbers <- c('Ace', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 
            'Nine', 'Ten', 'Jack', 'Queen', 'King')
deck <- expand.grid(number = numbers, suit = suits)
deck <- paste(deck$number, deck$suit)

# probability of drawing a king
kings <- paste('King', suits)
mean(deck %in% kings)

# probability of drawing a second king given that the first card is a king
hands <- permutations(52, 2, v = deck)
first_card <- hands[,1]
second_card <- hands[,2]
mean(first_card %in% kings & second_card %in% kings)/mean(first_card %in% kings)

# probability of a natural 21 (ace + face card or 10) in blackjack
# we do not consider all possible permutations here because ace + ten is same 
# as ten + ace
aces <- paste('Ace', suits)
facecards <- c("King", "Queen", "Jack", "Ten")
facecards <- expand.grid(number = facecard, suit = suits)
facecards <- paste(facecard$number, facecard$suit)

hands <- combinations(52, 2, v = deck)
first_card <- hands[,1]
second_card <- hands[,2]

mean((first_card %in% aces & second_card %in% facecards) | 
       (first_card %in% facecards & second_card %in% aces))

# monte carlo simulation for one hand of blackjack

hand <- sample(deck, 2)
hand
set.seed(2022 - 9 - 29)

n <- 10000
results <- replicate(n, {
  hand <- sample(deck, 2)
  (hand[1] %in% aces & hand[2] %in% facecards) | (hand[2] %in% aces & hand[1] %in% facecards)
})
mean(results)
