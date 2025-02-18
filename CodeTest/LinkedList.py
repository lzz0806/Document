class LinkNode:

    def __init__(self, x):
        self.val = x
        self.next = None


def createLinkedList(array) -> LinkNode:
    if array is None or len(array) == 0:
        return None

    head = LinkNode(array[0])
    cur = head
    for i in range(1, len(array)):
        cur.next = LinkNode(array[i])
        cur = cur.next

    return head


def print_linked_list(head: LinkNode):
    res = ''
    p = head
    while p:
        res += f"{p.val} -> "
        p = p.next
    print(res[:-4])


def add_node_to_head(head: LinkNode, val: int):
    new_head = LinkNode(val)
    new_head.next = head
    head = new_head
    print_linked_list(head)


def add_node_to_tail(head: LinkNode, val: int):
    p = head
    while p.next:
        p = p.next
    p.next = LinkNode(val)
    print_linked_list(head)


def add_node_to_linked(head: LinkNode, val: int, nums: int):

    p = head
    for i in range(nums-1):
        p = p.next
    pre_head = p.next
    p.next = LinkNode(val)
    p.next.next = pre_head
    print_linked_list(head)

def del_node_to_linked(head: LinkNode, nums: int):
    p = head
    for i in range(nums-2):
        p = p.next
    p.next = p.next.next
    print_linked_list(head)

if __name__ == '__main__':
    linked_head = createLinkedList([1, 2, 3, 4, 5])

    # add_node_to_head(linked_head, 6)
    # add_node_to_tail(linked_head, 444)
    # add_node_to_linked(linked_head, 445, 2)
    del_node_to_linked(linked_head, 4)
