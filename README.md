# einstein.py
Solve Einstein's riddle

### Notes
- Conditions are poorly implemented
- There are a lot of repeatable flag checks & sets

### Output
```
if not on left of WHITE then impossible GREEN
if not on right of GREEN then impossible WHITE
if index 2 then value MILK
if not value TEA then impossible DANE
if not value COFE then impossible GREEN
if not on right of GREEN then impossible WHITE
if index 0 then value NOR
if not value BRIT then impossible RED
if not value SWEDE then impossible DOG
if not value DANE then impossible TEA
if not value BEER then impossible BMAST
if not value GER then impossible PRINCE
if not next to NOR then impossible BLUE
if not value RED then impossible BRIT
if not on right of GREEN then impossible WHITE
if not value RED then impossible BRIT
if on left of WHITE then value GREEN
if value RED then value BRIT
if not value SWEDE then impossible DOG
if value GREEN then value COFE
if not value TEA then impossible DANE
if value YELLOW then value DUN
if not value PAL then impossible BIRD
if not next to DUN then impossible HORSE
if not value DOG then impossible SWEDE
if not value BIRD then impossible PAL
if not value BMAST then impossible BEER
if not value BEER then impossible BMAST
if not value GER then impossible PRINCE
if not next to WATER then impossible BLEND
if value PAL then value BIRD
if not next to BLEND then impossible CAT
if value BMAST then value BEER
if value TEA then value DANE
if value PRINCE then value GER
if value SWEDE then value DOG

Color                       : Nation                  : Drink                    : Cigar                      : Pet                    
YELLOW                      : NOR                     : WATER                    : DUN                        : CAT                    
BLUE                        : DANE                    : TEA                      : BLEND                      : HORSE                  
RED                         : BRIT                    : MILK                     : PAL                        : BIRD                   
GREEN                       : GER                     : COFE                     : PRINCE                     : FISH                   
WHITE                       : SWEDE                   : BEER                     : BMAST                      : DOG                    
```