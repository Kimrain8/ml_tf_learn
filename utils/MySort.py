# 用显性的二叉树来检查数据是否正确：经检验没有问题
import numpy as np

#使用二叉树的方法进行排序：
class MyArrSort:
    def __init__():
        pass
    @staticmethod
    def arrSort(arr,opt=1):
        """arr :需要排序的数组；opt：选项"""
        #先从arr的中间位置取出一个数值
        rst = {}
        k = len(arr)//2
        #rst的每个节点有7个元素：
        #元素1：左节点序号
        #元素2：当前节点的数值
        #元素3：当前节点数值出现的个数
        #元素4：右节点的序号
        #元素5：小于当前节点值的节点个数
        #元素6：上层的节点是第几个(在此rst中)
        #元素7：在原数组中排列的序号(先删除此元素)
        
        g=0
        rst[g]=[None,arr[k],0,None,0,-1] #[None,arr[k],0,None,0,-1,""] #小于当前数据的左节点序号
    
        for u,v in enumerate(arr):
            j = 0
            while rst.get(j) is not None: #
                if v == rst[j][1]:
                    rst[j][2] = rst[j][2] + 1
                    # rst[j][6]=rst[j][6] + "," if len(rst[j][6]) >= 1 else ""  + str(u)
                    j=-1
                elif v < rst[j][1]:
                    rst[j][4] = rst[j][4] + 1  #记录小于此节点值的个数
                    if rst[j][0] is None:
                        g = g+1
                        rst[g]=[None,v,1,None,0,j] #[None,v,1,None,0,j,str(u)]
                        rst[j][0] = g
                        j=-1
                    else:
                        j = rst[j][0]
    
                else:
                    if rst[j][3] is None:
                        g = g+1
                        rst[g]=[None,v,1,None,0,j] #[None,v,1,None,0,j,str(u)]
                        rst[j][3] = g
                        j=-1
                    else:
                        j = rst[j][3]
        return rst

    @staticmethod
    def arrSort2(arr,l):
        """
        arr: 经过myArrSort函数排序后的结果
        l: 表示原数组长度
        """
        j = 0
        rst = np.tile([None],l)
        seq = []
        seq.append(arr[0][4] + arr[0][2]-1)
        for i in np.arange(arr[0][2]):
            rst[seq[j]-i] = arr[j][1]
        j = j+1
        while j<len(arr):
            if arr[j][5]>=0: #表示arr中第j个元素有上层元素
                if arr[arr[j][5]][0] == j:
                    # 上一层节点排序位置序号 - 小于上一层节点的个数 - 上一层重复次数 + 小于本结点的个数 + 本节点的重复数
                    seq.append(seq[arr[j][5]] - arr[arr[j][5]][4] - arr[arr[j][5]][2] + arr[j][4] + arr[j][2])
                    for i in np.arange(arr[j][2]):
                        rst[seq[j]-i] = arr[j][1]
                elif arr[arr[j][5]][3] == j:
                    #上层节点应该位于第几个 + 小于本节点的个数 + 本节点的重复数
                    seq.append(seq[arr[j][5]] + arr[j][4] + arr[j][2])
                    for i in np.arange(arr[j][2]):
                        rst[seq[j]-i] = arr[j][1]
            j=j+1
        return rst
    
    #然后将树状结构的数据  转换为最后满足要求数据顺序（序号）：未完成
    @staticmethod
    def arrSort3(arr,l):
        """
        返回排好序的序号:rst
        """
        j = 0
        rst = []
        rst.append(arr[0][4] + arr[0][2]-1) 
        j = j+1
        while j<len(arr):
            if arr[j][5]>=0: #表示arr中第j个元素有上层元素
                if arr[arr[j][5]][0] == j:
                    # 上一层节点排序位置序号 - 小于上一层节点的个数 - 上一层重复次数 + 小于本结点的个数 + 本节点的重复数
                    rst.append(rst[arr[j][5]] - arr[arr[j][5]][4] - arr[arr[j][5]][2] + arr[j][4] + arr[j][2])
                elif arr[arr[j][5]][3] == j:
                    #上层节点应该位于第几个 + 小于本节点的个数 + 本节点的重复数
                    rst.append(rst[arr[j][5]] + arr[j][4] + arr[j][2])
            j=j+1
        rst2 = np.tile([None],l)
        j = 0
        while j<len(arr):
            rst2[rst[j]] = r[j][1]
            j=j+1
        return rst2

# 手动定义二叉树结构（示例）
class TreeNode:
    def __init__(self, value, feature=None, threshold=None, left=None, right=None):
        self.value = value          # 节点值（如果是叶子节点）
        self.feature = feature      # 分裂特征（非叶子节点，有多少个）
        self.threshold = threshold  # 分裂阈值（非叶子节点）
        self.left = left            # 左子树
        self.right = right          # 右子树
        self.node_id = 0
    

    # 对排序的结果数据生成一棵树：（参数data就是 myArrSort 返回的结果）
    @staticmethod
    def genTree(data):
        nodes = []
        for k,v in data.items():
            nodes.append(TreeNode(v[1],feature=v[2],threshold=v[1]))
        for k,v in data.items():
            if (-1 if v[0] is None else v[0]) >= 0 :
                nodes[k].left = nodes[v[0]]
            if (-1 if v[3] is None else v[3]) >= 0:
                nodes[k].right = nodes[v[3]]
        return nodes[0]


    # 对树转换成可是识别的graphviz的数据：
    # 递归生成 DOT 数据
    @staticmethod
    def tree_to_dot(node, parent=None, edge_label=''):
        nid = 0
        def tree_to_dot1(node, parent=None, edge_label=''): #, node_id=0
            # global node_id
            # nid=self.node_id
            nonlocal nid
            if node is None:
                return ''
            
            current_id = f'node_{nid}'
            nid += 1
            
            if node.value is not None:
                label = f'{node.value}*{node.feature}'
            else:  #非叶子节点全部作为分类标准，所以没有value
                label = f'Feature {node.feature} ≤ {node.threshold:.2f}'
            
            dot =  f'{current_id} [label="{label}"];\n'
            
            if parent is not None:
                dot += f'{parent} -> {current_id} [labeldistance=2.5, labelangle=45, headlabel="{edge_label}"];\n'
            
            if node.left is not None:
                dot += tree_to_dot1(node.left, current_id, 'Yes')
            if node.right is not None:
                dot += tree_to_dot1(node.right, current_id, 'No')
            return dot
        return tree_to_dot1(node, parent, edge_label)

# if __name__=="__main__":
#     #先对数组进行排序：
#     arr = [7,4,2,2,6,8,1,6,5,6,8,9]
#     r = myArrSort(arr)
#     print(r.values())
#     print(myArrSort2(r,len(arr)))
    
#     #在对排序后的二叉树进行 绘制树形图
#     tree1 = TreeNode.genTree(r)
#     dot_data = "digraph G {\n" + TreeNode.tree_to_dot(tree1) + "}"
    
#     # 生成 PNG 图像（前两行的方式有点问题）
#     # graph = pydotplus.graph_from_dot_data(dot_data)
#     # Image(graph.create_png())
#     src = Source(dot_data)
#     Image(src.pipe(format='png'))  # 输出PNG
#     # 或者：直接渲染DOT数据
#     # src = Source(dot_data)
#     # src.render(filename='tree', format='png', cleanup=True)  # 生成tree.png
#     # Image(filename='tree.png')  # 在Jupyter中显示