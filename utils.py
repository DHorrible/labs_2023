from types import FunctionType
from llist import dllist

def dll_remove_if(l: dllist, f: FunctionType) -> None:
    node = l.first
    while node is not None:
        if f(node.value):
            removed_node = node 
            node = node.next
            l.remove(removed_node)
        else:
            node = node.next
