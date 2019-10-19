#!/usr/bin/env python3

'''
https://udel.edu/~os/riddle.html

The situation

    There are 5 houses in five different colors.
    In each house lives a person with a different nationality.

    These five owners drink a certain type of beverage, smoke a
    certain brand of cigar and keep a certain pet.  No owners have the
    same pet, smoke the same brand of cigar or drink the same
    beverage.

The question is: Who owns the fish?

Hints:
    the Brit lives in the red house
    the Swede keeps dogs as pets
    the Dane drinks tea
    the green house is on the left of the white house
    the green house's owner drinks coffee
    the person who smokes Pall Mall rears birds
    the owner of the yellow house smokes Dunhill
    the man living in the center house drinks milk
    the Norwegian lives in the first house
    the man who smokes blends lives next to the one who keeps cats
    the man who keeps horses lives next to the man who smokes Dunhill
    the owner who smokes BlueMaster drinks beer
    the German smokes Prince
    the Norwegian lives next to the blue house
    the man who smokes blend has a neighbor who drinks water

Solution:
    House   #1      #2      #3      #4      #5
    Color   Yellow  Blue    Red     Green   White
    Natl    Norweg  Dane    Brit    German  Swede
    Bevg    Water   Tea     Milk    Coffee  Beer
    Smokes  Dunhill Blends  PallM   Prince  BlueM
    Pet     Cat     Horse   Birds   Fish    Dogs
'''

import enum
import itertools
import operator
import re
import sys
from functools import reduce


# +---------+
# | General |
# +---------+

# TODO: Do not use _decompose()
def e2s(flag):
    if flag.name is not None:
        return flag.name
    members, _ = enum._decompose(type(flag), flag.value)
    return '|'.join(m.name for m in members)


POWERS = [2**x for x in range(31)]


def singlebit(flag):
    return flag.value in POWERS


class Entry:

    SEPARATOR = ' : '

    def __init__(self, model):
        def full(cls):
            return reduce(operator.or_, cls, cls(0))
        self._flags = [full(cls) for cls in model]
        self.width = [len(s) for s in map(e2s, self._flags)]

    def __str__(self):
        def f(tup):
            index, flags = tup
            return e2s(flags).ljust(self.width[index])
        en = enumerate(self._flags)
        return Entry.SEPARATOR.join(map(f, en))

    def __repr__(self):
        return 'Entry: {}'.format(
            ', '.join(map(str, (x.value for x in self._flags)))
        )

    def _index(self, flag):
        '''
        Return the index at which self._flags holds flags of the same
        type.
        '''
        for i, p in enumerate(self._flags):
            if type(p) == type(flag):
                return i
        raise ValueError

    # Accessors

    def flag_by_type(self, cls):
        for p in self._flags:
            if type(p) == cls:
                return p
        raise ValueError

    # Predicates

    def eq(self, flag):
        return self.flag_by_type(type(flag)) == flag

    def isset(self, flag):
        return flag in self.flag_by_type(type(flag))

    # Manipulators

    def assign(self, flag):
        i = self._index(flag)
        old = self._flags[i]
        self._flags[i] &= flag
        return self._flags[i] != old

    def unset(self, flag):
        i = self._index(flag)
        old = self._flags[i]
        self._flags[i] &= ~flag
        return self._flags[i] != old


class Rule:

    def __init__(self, cond, then):
        self.cond = cond
        self.then = then

    def apply(self, matrix):
        '''Return True on successful application and actual change'''
        results = self.cond(matrix)
        if results:
            entries = matrix.enum(results)
            change = self.then(entries)
            if change:
                # matrix.norm() works also, but is generally not reliable
                while matrix.norm():
                    pass
                return True
        return False

    def __str__(self):
        return '{} {}'.format(self.cond.__name__, self.then.__name__)


class Matrix:

    def __init__(self):
        self._model = []
        self._entries = []

    # Building

    def flag(self, name, values=None):
        if values is None:
            flag = name
        else:
            flag = enum.Flag(name, values)
        self._model.append(flag)
        return flag

    def alloc(self, num):
        for _ in range(num):
            self._entries.append(Entry(self._model))

    # Modifiers

    def norm(self):
        '''
        First: if there is only one entry that may have a particular
        value, then set that value.

        XXX: This function has the nasty misfeature of setting anew
        single-bit values!

        Second: if there's a mask that has only one flag set, then
        clear # that flag for all other entries.
        '''
        change = False
        # 1
        for cls in self._model:
            occs = {}           # occurrences
            memo = {}           # keeps last entry with this flag
            for index, entry in self.enum():
                for flag in cls:
                    if entry.isset(flag):
                        occs.setdefault(flag, 0)
                        occs[flag] += 1
                        memo[flag] = index
            for flag, occ in occs.items():
                if occ == 1:
                    index = memo[flag]
                    change |= self._entries[index].assign(flag)
        # 2
        for index, entry in self.enum():
            for flag in entry._flags:
                if singlebit(flag):
                    change |= self.unset(flag, index)
        return change

    def unset(self, flag, exceptfor):
        change = False
        for index, entry in self.enum():
            if index != exceptfor:
                change |= entry.unset(flag)
        return change

    # Accessors

    def check(self, index):
        '''Check if index is valid'''
        return 0 <= index < len(self._entries)

    def entries(self):
        return self._entries

    def enum(self, xs=None):
        if xs:
            return list(zip(itertools.count(), self._entries, xs))
        else:
            return list(enumerate(self._entries))

    def find(self, flag):
        '''
        Find the first index of an entry that has this flag singly
        set.
        '''
        for index, entry in self.enum():
            if entry.eq(flag):
                return index
        return -1

    def length(self):
        return len(self._entries)

    def model(self):
        return self._model

    # Print

    def pr(self):
        if not self._entries:
            return
        one = self._entries[0]

        def f(tup):
            index, flag = tup
            classname = type(flag).__name__
            width = one.width[index]
            just = classname.ljust(one.width[index])
            return just

        en = enumerate(one._flags)
        line = Entry.SEPARATOR.join(map(f, en))
        print(line)
        for e in self._entries:
            print(e)


# +------------+
# | Conditions |
# +------------+

def cond(fn, name=''):
    def clojure(data):
        def test(matrix):
            return [fn(data, index, entry)
                    for (index, entry) in matrix.enum()]
        test.__name__ = 'if {} {}'.format(name, getattr(data, 'name', data))
        return test
    return clojure


def cond_needle(fn, name=''):
    def clojure(flag):
        def test(matrix):
            pass
        test.__name__ = 'if {} {}'.format(name, flag.name)
        return test
    return clojure


def cond2(fn, name=''):
    def clojure(flag):
        def test(matrix):
            return fn(flag, matrix)
        test.__name__ = 'if {} {}'.format(name, flag.name)
        return test
    return clojure


def index(needle, index, _):
    return needle == index


def value(flag, _, entry):
    return entry.eq(flag)


def not_value(flag, _, entry):
    actual = entry.flag_by_type(type(flag))
    return singlebit(actual) and actual != flag


def on_left(flag, matrix):
    needle = matrix.find(flag)
    if needle >= 0:
        left = needle - 1
        return [index == left for (index, _) in matrix.enum()]
    return None


def on_right(flag, matrix):
    needle = matrix.find(flag)
    if needle >= 0:
        right = needle + 1
        return [index == right for (index, _) in matrix.enum()]
    return None


def not_on_left(flag, matrix):
    res = [False] * matrix.length()
    for index, entry in matrix.enum():
        if not entry.isset(flag):
            left = index - 1
            if matrix.check(left):
                res[left] = True
        # The last one can't be on the left of anything
        res[-1] = True
        return res


def not_on_right(flag, matrix):
    res = [False] * matrix.length()
    for index, entry in matrix.enum():
        if not entry.isset(flag):
            right = index + 1
            if matrix.check(right):
                res[right] = True
    # The first one can't be on the right of anything.
    res[0] = True
    return res


def not_next_to(flag, matrix):
    res = [True] * matrix.length()
    for index, entry in matrix.enum():
        if entry.isset(flag):
            left = index - 1
            if matrix.check(left):
                res[left] = False
            right = index + 1
            if matrix.check(right):
                res[right] = False
    return res


if_index = cond(index, 'index')
if_value = cond(value, 'value')
if_not_value = cond(not_value, 'not value')
if_on_left = cond2(on_left, 'on left of')
if_on_right = cond2(on_right, 'on right of')
if_not_on_left = cond2(not_on_left, 'not on left of')
if_not_on_right = cond2(not_on_right, 'not on right of')
if_not_next_to = cond2(not_next_to, 'not next to')


# +------+
# | Then |
# +------+

def then(fn, name=''):
    def clojure(flag):
        def apply(entries):
            change = False
            for _, entry, test in entries:
                if test:
                    change |= fn(entry, flag)
            return change
        apply.__name__ = 'then {} {}'.format(name, flag.name)
        return apply
    return clojure


then_value = then(Entry.assign, 'value')
then_impossible = then(Entry.unset, 'impossible')


# +-------+
# | Rules |
# +-------+

class RuleBook:

    def __init__(self, model):
        self._rules = []
        self._flags = {}
        for m in model:
            for flag in m:
                self._flags[flag.name] = flag

    def _make(self, cond, a, then, b):
        rule = Rule(cond(a), then(b))
        self._rules.append(rule)

    def rule(self, s):
        p = re.compile(
            r'if '
            r'(?P<cond>index|next to|on left of|value) '
            r'(?P<a>\w+) '
            r'then '
            r'(?P<then>possible|value) '
            r'(?P<b>[A-Z]+)'
        )
        match = p.match(s)
        if not match:
            raise ValueError(s)
        cond = match['cond']
        then = match['then']
        a = self._flags.get(match['a'], match['a'])
        b = self._flags[match['b']]
        if cond == 'value' and then == 'value':
            self._make(if_value, a, then_value, b)
            self._make(if_value, b, then_value, a)
            self._make(if_not_value, a, then_impossible, b)
            self._make(if_not_value, b, then_impossible, a)
        elif cond == 'next to' and then == 'possible':
            self._make(if_not_next_to, a, then_impossible, b)
            self._make(if_not_next_to, b, then_impossible, a)
        elif cond == 'index' and then == 'value':
            self._make(if_index, int(a), then_value, b)
        elif cond == 'on left of' and then == 'value':
            self._make(if_on_left, a, then_value, b)
            self._make(if_on_right, b, then_value, a)
            self._make(if_not_on_left, a, then_impossible, b)
            self._make(if_not_on_right, b, then_impossible, a)

    def rules(self):
        return self._rules


# +------+
# | Main |
# +------+

def main():
    m = Matrix()
    m.flag('Color', 'RED GREEN WHITE YELLOW BLUE')
    m.flag('Nation', 'BRIT SWEDE DANE NOR GER')
    m.flag('Drink', 'TEA COFE MILK BEER WATER')
    m.flag('Cigar', 'PAL DUN BLEND BMAST PRINCE')
    m.flag('Pet', 'DOG BIRD CAT HORSE FISH')
    m.alloc(5)                  # 5 men

    book = RuleBook(m.model())
    book.rule('if value BRIT then value RED')
    book.rule('if value SWEDE then value DOG')
    book.rule('if value DANE then value TEA')
    book.rule('if on left of WHITE then value GREEN')
    book.rule('if value GREEN then value COFE')
    book.rule('if value PAL then value BIRD')
    book.rule('if value YELLOW then value DUN')
    book.rule('if index 2 then value MILK')
    book.rule('if index 0 then value NOR')
    book.rule('if next to CAT then possible BLEND')
    book.rule('if next to DUN then possible HORSE')
    book.rule('if value BMAST then value BEER')
    book.rule('if value GER then value PRINCE')
    book.rule('if next to BLUE then possible NOR')
    book.rule('if next to WATER then possible BLEND')

    verbose = '-v' in sys.argv

    while 1:
        for rule in book.rules():
            if rule.apply(m):
                print(rule)
                if verbose:
                    m.pr()
                    print()
                break
        else:
            break

    if not verbose:
        print()
        m.pr()


if __name__ == '__main__':
    main()
