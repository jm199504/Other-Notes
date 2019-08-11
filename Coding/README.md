```javascript
class Node:
    def __init__(self,data=None,next = None):
        self.data = data
        self.next = next
 
def rev(link):
    pre = link
    cur = link.next
    pre.next = None
    while cur:
        temp = cur.next
        cur.next = pre
        pre =cur
        cur = temp
    return pre
 
if __name__ == '__main__':
    link = Node(1, Node(2, Node(3, Node(4, Node(5, Node(6, Node(7, Node(8, Node(9)))))))))
    root = rev(link)
    while root:
        print(root.data)
        root =root.next
```