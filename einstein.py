#!/usr/bin/env python3

'''
https://udel.edu/~os/riddle.html

The situation:
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
import functools
import itertools
import operator
import re
import sys


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
            return functools.reduce(operator.or_, cls, cls(0))
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

    # Accessors

    def _index_by_type(self, cls):
        for i, p in enumerate(self._flags):
            if type(p) == cls:
                return i
        raise ValueError

    def flag_by_type(self, cls):
        index = self._index_by_type(cls)
        return self._flags[index]

    # Predicates

    def eq(self, flag):
        return self.flag_by_type(type(flag)) == flag

    def isset(self, flag):
        return flag in self.flag_by_type(type(flag))

    # Manipulators

    def assign(self, flag):
        i = self._index_by_type(type(flag))
        old = self._flags[i]
        self._flags[i] &= flag
        return self._flags[i] != old

    def unset(self, flag):
        i = self._index_by_type(type(flag))
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
                matrix.norm()
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
        First: if there is only one entry across the matrix that may
        have a particular value, then set that value.

        XXX: This function has the nasty misfeature of setting anew
        single-bit values!

        Second: if there's a mask that has only one flag set, then
        clear that flag for all other entries.
        '''
        change = False
        # 1
        for cls in self._model:
            occs = {}           # occurrences
            memo = {}           # keep last entry with this flag set
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
                    change |= self._unset(flag, exceptfor=index)
        return change

    def _unset(self, flag, exceptfor):
        change = False
        for index, entry in self.enum():
            if index != exceptfor:
                change |= entry.unset(flag)
        return change

    # Accessors

    def check(self, index):
        '''Check if index is valid'''
        return 0 <= index < len(self._entries)

    def enum(self, xs=None):
        if xs:
            return zip(itertools.count(), self._entries, xs)
        else:
            return enumerate(self._entries)

    def find(self, flag):
        '''
        Find the index of the entry that has this single-bit flag set.
        '''
        for index, entry in self.enum():
            if entry.eq(flag):
                return index
        return -1

    @property
    def length(self):
        return len(self._entries)

    @property
    def model(self):
        return self._model

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

def cond(initial=False, flip=None, needle=False):
    def wrapper(fn):
        @functools.wraps(fn)
        def clojure(data):
            def test(matrix):
                # Each condition returns an array of bools indicating
                # whether the respective entry passed the test or not
                ary = [initial] * matrix.length
                # Hard-coded index to flip
                if flip is not None:
                    ary[flip] = not initial
                # The aggregated function may also need a needle
                needle_ = matrix.find(data) if needle else None
                # For each entry, call the aggregated function and
                # flip any indices it returns
                for index, entry in matrix.enum():
                    res = fn(data, index, entry, needle_)
                    # Flip indices returned
                    for j in filter(matrix.check, res or ()):
                        ary[j] = not initial
                return ary
            # The name of the resulting test function is made out of
            # the original function name and the data held in clojure.
            # `data' may optionally be a flag, which has a .name attr.
            test.__name__ = '{} {}'.format(
                fn.__name__.replace('_', ' '),
                getattr(data, 'name', data)
            )
            return test
        return clojure
    return wrapper


@cond()
def if_index(searched, index, entry, *args):
    return (index,) if searched == index else None


@cond()
def if_value(flag, index, entry, *args):
    return (index,) if entry.eq(flag) else None


@cond()
def if_not_value(flag, index, entry, *arsg):
    actual = entry.flag_by_type(type(flag))
    if singlebit(actual) and actual != flag:
        return (index,)


@cond(flip=-1)
def if_not_on_left_of(flag, index, entry, *args):
    if not entry.isset(flag):
        return (index - 1,)


@cond(flip=0)
def if_not_on_right_of(flag, index, entry, *args):
    if not entry.isset(flag):
        return (index + 1,)


@cond(initial=True)
def if_not_next_to(flag, index, entry, *args):
    if entry.isset(flag):
        return (index - 1, index + 1)


@cond(needle=True)
def if_on_left_of(flag, index, entry, needle):
    return (index,) if (index == needle - 1) else None


@cond(needle=True)
def if_on_right_of(flag, index, entry, needle):
    return (index,) if (needle >= 0 and index == needle + 1) else None


# +------+
# | Then |
# +------+

def then(fn, name):
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
        self._flags = {}
        for m in model:
            for flag in m:
                self._flags[flag.name] = flag
        self._pattern = re.compile(
            r'if '
            r'(?P<cond>index|next to|on left of|value) '
            r'(?P<a>\w+) '
            r'then '
            r'(?P<then>possible|value) '
            r'(?P<b>[A-Z]+)'
        )
        self._rules = []

    def _make(self, cond, a, then, b):
        rule = Rule(cond(a), then(b))
        self._rules.append(rule)

    def apply(self, matrix):
        for rule in self._rules:
            if rule.apply(matrix):
                return rule

    def rule(self, s):
        match = self._pattern.match(s)
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
            self._make(if_on_left_of, a, then_value, b)
            self._make(if_on_right_of, b, then_value, a)
            self._make(if_not_on_left_of, a, then_impossible, b)
            self._make(if_not_on_right_of, b, then_impossible, a)

    @property
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

    book = RuleBook(m.model)
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

    verbose = '-v' in sys.argv or '--verbose' in sys.argv

    while 1:
        rule = book.apply(m)
        if not rule:
            break
        print(rule)
        if verbose:
            m.pr()
            print()

    if not verbose:
        print()
        m.pr()


if __name__ == '__main__':
    main()
