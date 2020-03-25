from modules import *

def binary_search(l, key):
    start = 0
    end = len(l) -1
    while start <= end:
        mid = int((start + end)/2)
        if key == l[mid]:
            return mid
        elif key < l[mid]:
            end = mid - 1
        elif key > l[mid]:
            start = mid + 1
    return -1

## module-based binary search

def module_binary_search_func():
    set_len = SetVar('length', ListUnit('l', ListOp.LEN))
    set_start = SetVar('start', Value(0))
    set_end = SetVar('end', ArithUnit(GetVar('length'), Value(1), Operation.MINUS))

    while_cond = BoolUnit(GetVar('start'), GetVar('end'), Comparison.LESSTHANEQ)

    #inside loop
    sum_start_end = ArithUnit(GetVar('start'), GetVar('end'), Operation.PLUS)
    mid_val = ExtraOps(ArithUnit(sum_start_end, Value(2), Operation.DIVIDEDBY), 'to_int')
    set_mid = SetVar('mid', mid_val)

    if_cond_1 = BoolUnit(GetVar('key'),
                       ListUnit('l', ListOp.GETIND, GetVar('mid')), Comparison.EQUALS)

    if_cond_2 = BoolUnit(GetVar('key'),
                       ListUnit('l', ListOp.GETIND, GetVar('mid')), Comparison.LESSTHAN)

    if_body_1 = Return(GetVar('mid'))
    if_body_2 = SetVar('end', ArithUnit(GetVar('mid'), Value(1), Operation.MINUS))
    if_body_3 = SetVar('start', ArithUnit(GetVar('mid'), Value(1), Operation.PLUS))

    if_statement = If(if_cond_1, if_body_1, If(if_cond_2, if_body_2, if_body_3))
    while_body = Function([set_mid, if_statement])

    while_statement = While(while_cond, while_body)
    final_return = Return(Value(-1))

    return Function([set_len, set_start, set_end, while_statement, final_return])

def main():
    l = [-1, 1, 2, 3, 3, 4, 5, 7]
    key = 4

    print('list:', l)
    print('key: %d' % key)

    print('expected index: %d' % binary_search(l, key))

    module_binary_search = module_binary_search_func()

    inp = Variables({'l': l, 'key': key})
    module_binary_search.execute(inp)

if __name__ == "__main__": main()

