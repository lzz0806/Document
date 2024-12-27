class TreeNode:

    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class BinaryTree:

    def __init__(self):
        self.res = []

    def pre_order(self, root: TreeNode):
        # 二叉树前序遍历，根左右
        pass

    def in_order(self, root: TreeNode):
        # 二叉树中序遍历，左根右
        pass

    def post_order(self, root: TreeNode):
        # 二叉树后续遍历，左右根
        pass

    def level_order_1(self, root: TreeNode):
        # 层序排序
        if not root:
            return []
        queue = [root]
        while queue:
            node = queue.pop(0)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            self.res.append(node.val)
        return self.res

    def level_order_2(self, root: TreeNode):
        # 层序遍历,记录每一层的值
        if not root:
            return []
        queue = [root]
        while queue:
            # 前处理，可以定义列表作为记录每一层值的变量
            tmp = []
            n = len(queue)
            for i in range(n):
                node = queue.pop(0)
                tmp.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            # 后续处理，例如可以将一层的值加入新列表，也可以记录当前层的高度
            self.res.append(tmp)
        return self.res



if __name__ == '__main__':
    tree = TreeNode(3)
    tree.left = TreeNode(9)
    tree.right = TreeNode(20)
    tree.left.left = TreeNode(15)
    tree.left.right = TreeNode(7)
    tree.right.left = TreeNode(22)
    tree.right.right = TreeNode(35)
    binary_tree = BinaryTree()
    # print(binary_tree.level_order_1(tree))
    print(binary_tree.level_order_2(tree))