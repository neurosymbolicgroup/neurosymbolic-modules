from dreamcoder.domains.scan.make_tasks import tstring
from dreamcoder.type import arrow

def root_primitive(word):
    return Primitive(word, tstring, word)

concat = Primitive('concat', arrow(tstring, tstring, tstring), lambda s1: lambda
        s2: s1 + s2)
