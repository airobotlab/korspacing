from korspacing import KorSpacing

rules = {
    '아버지 가방에': '아버지가 방에', 
    '아 버지가방': '아버지가 방',     
}

# spacing = KorSpacing(rules=rules)
spacing = KorSpacing()

result = spacing('아버지가방에들어가신다')
print(result)